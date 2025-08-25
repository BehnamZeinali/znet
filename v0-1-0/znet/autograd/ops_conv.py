# znet/autograd/ops_conv.py
# znet/autograd/ops_conv.py
from __future__ import annotations
import torch as th
from .engine import Function
from .tensor import Tensor

# ---------- helpers ----------
def _pair(x):
    return (x, x) if isinstance(x, int) else tuple(int(v) for v in x)

def _to_int(x):
    return int(x)

def _eff_kernel(k, d):
    k, d = int(k), int(d)
    return 1 + (k - 1) * d

def _pad2d(x: th.Tensor, padH: int, padW: int) -> th.Tensor:
    if padH == 0 and padW == 0:
        return x
    N, C, H, W = x.shape
    out = th.zeros((N, C, H + 2*padH, W + 2*padW), dtype=x.dtype, device=x.device)
    out[:, :, padH:padH+H, padW:padW+W] = x
    return out

def _im2col_windows(x_pad: th.Tensor, kH, kW, strH, strW, dilH, dilW):
    """
    Returns:
      win: view of shape (N, C, OH, OW, kH, kW) respecting dilation/stride
      OH, OW: output spatial sizes
    Implementation uses as_strided to first make windows of size (eff_kH, eff_kW),
    then slices them with steps (dilH, dilW) to realize the true (kH,kW) dilation.
    """
    kH = _to_int(kH); kW = _to_int(kW)
    strH = _to_int(strH); strW = _to_int(strW)
    dilH = _to_int(dilH); dilW = _to_int(dilW)

    N, C, Hpad, Wpad = x_pad.shape
    eff_kH = _eff_kernel(kH, dilH)
    eff_kW = _eff_kernel(kW, dilW)

    # output sizes
    if Hpad < eff_kH or Wpad < eff_kW:
        OH = OW = 0
    else:
        OH = (Hpad - eff_kH) // strH + 1
        OW = (Wpad - eff_kW) // strW + 1
    if OH <= 0 or OW <= 0:
        raise ValueError(f"Invalid conv config yields non-positive output: (OH,OW)=({OH},{OW})")

    sN, sC, sH, sW = x_pad.stride()  # in elements
    # windows over (eff_kH, eff_kW) with strides stepping by (strH, strW)
    out_size   = (N, C, OH, OW, eff_kH, eff_kW)
    out_stride = (sN, sC, sH*strH, sW*strW, sH, sW)
    win_eff = th.as_strided(x_pad, size=out_size, stride=out_stride)

    # realize dilation by sub-sampling inside the window
    if dilH != 1 or dilW != 1:
        win = win_eff[:, :, :, :, ::dilH, ::dilW]
    else:
        win = win_eff

    if win.shape[-2:] != (kH, kW):
        raise RuntimeError(
            f"Internal: window size {tuple(win.shape[-2:])} != kernel {(kH, kW)}; "
            f"eff=({_eff_kernel(kH,dilH)},{_eff_kernel(kW,dilW)}), dil=({dilH},{dilW})"
        )
    return win, OH, OW

# ---------- Conv2d op ----------
class Conv2d(Function):
    @staticmethod
    def forward(ctx, x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
        strH, strW = _pair(stride)
        padH, padW = _pair(padding)
        dilH, dilW = _pair(dilation)
        G = _to_int(groups)

        if x.ndim != 4: raise ValueError(f"x (N,C,H,W) expected, got {tuple(x.shape)}")
        if w.ndim != 4: raise ValueError(f"weight (C_out,Cg,kH,kW) expected, got {tuple(w.shape)}")
        if b is not None and b.ndim != 1: raise ValueError(f"bias (C_out,) expected, got {tuple(b.shape)}")

        N, C_in, H, W = x.shape
        C_out, Cg, kH, kW = w.shape
        if C_in % G or C_out % G:
            raise ValueError("in/out channels must be divisible by groups")
        if Cg != C_in // G:
            raise ValueError(f"weight second dim must be C_in//groups; got {Cg}")

        # padding
        x_pad = _pad2d(x, padH, padW)

        # im2col windows: (N, C_in, OH, OW, kH, kW)
        win, OH, OW = _im2col_windows(x_pad, kH, kW, strH, strW, dilH, dilW)

        if G == 1:
            # cols: (N*OH*OW, C_in*kH*kW)
            cols = win.permute(0, 2, 3, 1, 4, 5).reshape(N*OH*OW, C_in*kH*kW)
            # weights: (C_in*kH*kW, C_out)
            Wmat = w.reshape(C_out, C_in*kH*kW).transpose(0, 1)
            out2d = cols @ Wmat                                     # (N*OH*OW, C_out)
        else:
            Cg = C_in // G; Og = C_out // G
            # (N, OH, OW, G, Cg*kH*kW)
            cols_g = win.reshape(N, C_in, OH, OW, kH, kW) \
                       .reshape(N, G, Cg, OH, OW, kH, kW) \
                       .permute(0, 3, 4, 1, 2, 5, 6) \
                       .reshape(N*OH*OW, G, Cg*kH*kW)                 # (N*OH*OW, G, Cg*kH*kW)
            # (G, Og, Cg*kH*kW)
            Wg = w.reshape(G, Og, Cg, kH, kW).reshape(G, Og, Cg*kH*kW)
            # out per group: (N*OH*OW, G, Og)
            out_g2d = th.einsum("ngd,god->ngo", cols_g, Wg)
            # merge groups: (N*OH*OW, C_out)
            out2d = out_g2d.reshape(N*OH*OW, C_out)

        # back to (N, C_out, OH, OW)
        out = out2d.reshape(N, OH, OW, C_out).permute(0, 3, 1, 2)
        if b is not None:
            out = out + b.view(1, -1, 1, 1)

        # save for backward
        ctx.save_for_backward(x, w, (b if b is not None else th.tensor([], device=x.device)))
        ctx.meta.update({
            "stride": (strH, strW),
            "padding": (padH, padW),
            "dilation": (dilH, dilW),
            "groups": G,
            "in_shape": (N, C_in, H, W),
            "out_shape": (N, C_out, OH, OW),
            "k_shape": (kH, kW),
            "has_bias": (b is not None),
        })
        return out

    def backward(self, grad_out):
        x, w, _ = self.ctx.saved_tensors
        strH, strW = self.ctx.meta["stride"]
        padH, padW = self.ctx.meta["padding"]
        dilH, dilW = self.ctx.meta["dilation"]
        G          = self.ctx.meta["groups"]
        (N, C_in, H, W)     = self.ctx.meta["in_shape"]
        (N2, C_out, OH, OW) = self.ctx.meta["out_shape"]
        (kH, kW)            = self.ctx.meta["k_shape"]
        has_bias            = bool(self.ctx.meta["has_bias"])
        assert N == N2

        # ---- dB ----
        gB = grad_out.sum(dim=(0, 2, 3)) if has_bias else None

        # ---- dW ----
        # --- dW (use same col layout as forward to avoid index misalignment) ---
        # ---- dW ----
        # --- dW (Torch tensors) ---
        # assumes: N, C_in, C_out, H, W, OH, OW, kH, kW, strH, strW, dilH, dilW, G
        #          and tensors x, w, grad_out are already torch.Tensors

        # padding (pure tensor, no F.pad)
        if padH or padW:
            N_, C_, H_, W_ = x.shape
            x_pad = th.zeros((N_, C_, H_ + 2*padH, W_ + 2*padW),
                            dtype=x.dtype, device=x.device)
            x_pad[:, :, padH:padH+H_, padW:padW+W_] = x
        else:
            x_pad = x

        # im2col windows (must be your torch version of _im2col_windows)
        win, OH_chk, OW_chk = _im2col_windows(x_pad, kH, kW, strH, strW, dilH, dilW)
        assert (OH, OW) == (OH_chk, OW_chk)

        if G == 1:
            # win: (N, C_in, OH, OW, kH, kW)
            # go : (N, C_out, OH, OW)
            gW = th.einsum("n c h w r s, n o h w -> o c r s", win, grad_out)
        else:
            Cg = C_in // G
            Og = C_out // G
            # reshape per-group
            win_g = win.reshape(N, G, Cg, OH, OW, kH, kW)          # (N, G, Cg, OH, OW, kH, kW)
            go_g  = grad_out.reshape(N, G, Og, OH, OW)              # (N, G, Og, OH, OW)
            gW_g  = th.einsum("n g c h w r s, n g o h w -> g o c r s", win_g, go_g)  # (G, Og, Cg, kH, kW)
            gW    = gW_g.reshape(C_out, Cg, kH, kW)                 # (C_out, Cg, kH, kW)



        # ---- dX (transpose conv via explicit upsample + im2col + einsum) ----
        # 1) insert zeros between grad_out samples (upsampling by stride)
        upH = (OH - 1) * strH + 1
        upW = (OW - 1) * strW + 1
        go_up = th.zeros((N, C_out, upH, upW), dtype=grad_out.dtype, device=grad_out.device)
        go_up[:, :, ::strH, ::strW] = grad_out

        # 2) flip spatial and swap out/in channels within groups
        Og = C_out // G; Cg = C_in // G
        w_g = w.reshape(G, Og, Cg, kH, kW)
        w_flip = w_g.flip(-1, -2)                               # (..., kH, kW) -> flipped
        # swap out/in -> shape (G, Cg, Og, kH, kW)
        w_T_g = w_flip.permute(0, 2, 1, 3, 4)
        # combine groups: (C_in, Og, kH, kW)
        w_T = w_T_g.reshape(C_in, Og, kH, kW)

        # 3) compute padding for transpose
        eff_kH = _eff_kernel(kH, dilH)
        eff_kW = _eff_kernel(kW, dilW)
        base_padH = (kH - 1) * dilH - padH
        base_padW = (kW - 1) * dilW - padW
        if base_padH < 0 or base_padW < 0:
            raise ValueError(f"Backward conv requires non-negative base transpose padding, got "
                             f"({base_padH},{base_padW}).")
        out_padH = (H + 2*padH - eff_kH) % strH
        out_padW = (W + 2*padW - eff_kW) % strW

        # asymmetric padding on bottom/right for output_padding
        pad_top, pad_bottom = base_padH, base_padH + out_padH
        pad_left, pad_right = base_padW, base_padW + out_padW
        if pad_top or pad_bottom or pad_left or pad_right:
            Hup, Wup = go_up.shape[-2:]
            go_pad = th.zeros((N, C_out, Hup + pad_top + pad_bottom, Wup + pad_left + pad_right),
                              dtype=go_up.dtype, device=go_up.device)
            go_pad[:, :, pad_top:pad_top+Hup, pad_left:pad_left+Wup] = go_up
        else:
            go_pad = go_up

        # 4) im2col on go_pad with stride=1 (still honoring dilation)
        win_dx, Hx, Wx = _im2col_windows(go_pad, kH, kW, 1, 1, dilH, dilW)
        if (Hx, Wx) != (H, W):
            raise RuntimeError(f"dx windows produced ({Hx},{Wx}) but expected input spatial ({H},{W}).")

        if G == 1:
            # gX[n,i,h,w] = sum_{o,r,s} win_dx[n,o,h,w,r,s] * w_T[i,o,r,s]
            gX = th.einsum("n o h w r s, i o r s -> n i h w", win_dx, w_T)
        else:
            # grouped
            win_dx_g = win_dx.reshape(N, G, Og, H, W, kH, kW)
            w_T_g    = w_T.reshape(G, Cg, Og, kH, kW)
            gX_g = th.einsum("n g o h w r s, g c o r s -> n g c h w", win_dx_g, w_T_g)
            gX = gX_g.reshape(N, C_in, H, W)

        return (gX, gW, gB) if has_bias else (gX, gW)

def conv2d(x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
    if not isinstance(x, Tensor): x = Tensor(x)
    if not isinstance(w, Tensor): w = Tensor(w)
    if b is not None and not isinstance(b, Tensor): b = Tensor(b)
    return Conv2d.apply(x, w, b, stride, padding, dilation, groups)







# # znet/autograd/ops_conv.py
# from __future__ import annotations
# import torch as th
# import torch.nn.functional as F
# from .engine import Function
# from .tensor import Tensor

# # ---------- helpers ----------
# def _pair(x):
#     return (x, x) if isinstance(x, int) else tuple(int(v) for v in x)

# def _eff_kernel(k, d):
#     k = int(k); d = int(d)
#     return 1 + (k - 1) * d

# # ---------- Conv2d op ----------
# class Conv2d(Function):
#     @staticmethod
#     def forward(ctx, x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
#         """
#         x: (N, C_in, H, W)
#         w: (C_out, C_in/groups, kH, kW)
#         b: (C_out,) or None
#         stride, padding, dilation: int or (int,int)
#         groups: int
#         """
#         if x.ndim != 4: raise ValueError(f"x (N,C,H,W) expected, got {tuple(x.shape)}")
#         if w.ndim != 4: raise ValueError(f"w (C_out,Cg,kH,kW) expected, got {tuple(w.shape)}")
#         if b is not None and b.ndim != 1: raise ValueError(f"bias (C_out,) expected, got {tuple(b.shape)}")

#         strH, strW = _pair(stride)
#         padH, padW = _pair(padding)
#         dilH, dilW = _pair(dilation)
#         G = int(groups)

#         N, C_in, H, W = x.shape
#         C_out, Cg, kH, kW = w.shape
#         if C_in % G or C_out % G: raise ValueError("in/out channels must be divisible by groups")
#         if Cg != C_in // G: raise ValueError(f"weight second dim must be C_in//groups; got {Cg}")

#         bias = b if isinstance(b, th.Tensor) else None
#         out = F.conv2d(x, w, bias=bias, stride=(strH, strW), padding=(padH, padW),
#                        dilation=(dilH, dilW), groups=G)

#         # Save for backward
#         ctx.save_for_backward(x, w, (b if isinstance(b, th.Tensor) else th.tensor([], device=x.device)))
#         ctx.meta.update({
#             "stride": (strH, strW),
#             "padding": (padH, padW),
#             "dilation": (dilH, dilW),
#             "groups": G,
#             "in_shape": (N, C_in, H, W),
#             "out_shape": tuple(out.shape),            # (N, C_out, OH, OW)
#             "k_shape": (kH, kW),
#             "has_bias": (b is not None),
#         })
#         return out

#     def backward(self, grad_out):
#         x, w, _ = self.ctx.saved_tensors
#         strH, strW = self.ctx.meta["stride"]
#         padH, padW = self.ctx.meta["padding"]
#         dilH, dilW = self.ctx.meta["dilation"]
#         G          = self.ctx.meta["groups"]
#         (N, C_in, H, W)     = self.ctx.meta["in_shape"]
#         (_, C_out, OH, OW)  = (None,) + self.ctx.meta["out_shape"]
#         (kH, kW)            = self.ctx.meta["k_shape"]
#         has_bias            = bool(self.ctx.meta["has_bias"])

#         # --- dB ---
#         gB = grad_out.sum(dim=(0, 2, 3)) if has_bias else None

#         # --- dW ---
#         # Unfold input into columns to accumulate weight gradients
#         cols = F.unfold(x, kernel_size=(kH, kW),
#                         dilation=(dilH, dilW),
#                         padding=(padH, padW),
#                         stride=(strH, strW))                        # (N, C_in*kH*kW, OH*OW)
#         go = grad_out.reshape(N, C_out, OH * OW)                    # (N, C_out, L)

#         Cg = C_in // G
#         Og = C_out // G
#         gW = th.zeros_like(w)                                       # (C_out, Cg, kH, kW)
#         Ckk = Cg * kH * kW

#         if G == 1:
#             # sum over N and positions L
#             gW_2d = th.einsum('nol,ncl->oc', go, cols)              # (C_out, C_in*kH*kW)
#             gW = gW_2d.view(C_out, C_in // G, kH, kW)
#         else:
#             for g in range(G):
#                 cols_g = cols[:, g*Ckk:(g+1)*Ckk, :]                # (N, Cg*kH*kW, L)
#                 go_g   = go[:, g*Og:(g+1)*Og, :]                    # (N, Og, L)
#                 gW_g2d = th.einsum('nol,ncl->oc', go_g, cols_g)     # (Og, Cg*kH*kW)
#                 gW[g*Og:(g+1)*Og, :, :, :] = gW_g2d.view(Og, Cg, kH, kW)

#         # --- dX (conv-transpose) ---
#         # Compute output_padding to guarantee exact (H, W)
#         eff_kH = _eff_kernel(kH, dilH)
#         eff_kW = _eff_kernel(kW, dilW)
#         out_padH = (H + 2*padH - eff_kH) % strH
#         out_padW = (W + 2*padW - eff_kW) % strW

#         gX = F.conv_transpose2d(
#             grad_out, w, bias=None,
#             stride=(strH, strW),
#             padding=(padH, padW),
#             output_padding=(out_padH, out_padW),
#             groups=G,
#             dilation=(dilH, dilW)
#         )  # -> (N, C_in, H, W)

#         return (gX, gW, gB) if has_bias else (gX, gW)

# def conv2d(x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
#     if not isinstance(x, Tensor): x = Tensor(x)
#     if not isinstance(w, Tensor): w = Tensor(w)
#     if b is not None and not isinstance(b, Tensor): b = Tensor(b)
#     return Conv2d.apply(x, w, b, stride, padding, dilation, groups)
