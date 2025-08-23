# znet/autograd/ops_conv.py
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .engine import Function

# ---------- helpers ----------
def _pair(x):
    return (x, x) if isinstance(x, int) else tuple(x)

def _as_int_pair(v):
    if isinstance(v, (tuple, list, np.ndarray)) and len(v) == 2:
        return int(v[0]), int(v[1])
    return int(v), int(v)

def _to_int(x):
    if isinstance(x, np.ndarray):
        if x.size != 1:
            raise TypeError(f"Expected scalar-like, got array with shape {x.shape}")
        return int(x.item())
    return int(x)

def _eff_kernel(k, d):
    return 1 + (_to_int(k) - 1) * _to_int(d)

def _im2col_windows(x_pad, kH, kW, strH, strW, dilH, dilW):
    kH, kW, strH, strW, dilH, dilW = map(_to_int, (kH, kW, strH, strW, dilH, dilW))
    N, C, Hpad, Wpad = x_pad.shape
    eff_kH = _eff_kernel(kH, dilH)
    eff_kW = _eff_kernel(kW, dilW)
    win = sliding_window_view(x_pad, (eff_kH, eff_kW), axis=(2, 3))
    win = win[:, :, ::strH, ::strW, :, :]
    OH, OW = win.shape[2], win.shape[3]
    if dilH != 1 or dilW != 1:
        win = win[..., ::dilH, ::dilW]
    if win.shape[-2:] != (kH, kW):
        raise RuntimeError(
            f"Internal error: window size {win.shape[-2:]} != kernel {(kH, kW)}; "
            f"eff=({_eff_kernel(kH,dilH)},{_eff_kernel(kW,dilW)}), dil=({dilH},{dilW})"
        )
    return win, OH, OW

# ---------- Conv2d op ----------
class Conv2d(Function):
    @staticmethod
    def forward(ctx, x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
        strH, strW = _as_int_pair(_pair(stride))
        padH, padW = _as_int_pair(_pair(padding))
        dilH, dilW = _as_int_pair(_pair(dilation))
        G = _to_int(groups)

        if x.ndim != 4: raise ValueError(f"x (N,C,H,W) expected, got {x.shape}")
        if w.ndim != 4: raise ValueError(f"weight (C_out,Cg,kH,kW) expected, got {w.shape}")
        if b is not None and b.ndim != 1: raise ValueError(f"bias (C_out,) expected, got {b.shape}")

        N, C_in, H, W = x.shape
        C_out, Cg, kH, kW = w.shape
        if C_in % G or C_out % G: raise ValueError("in/out channels must be divisible by groups")
        if Cg != C_in // G: raise ValueError(f"weight second dim must be C_in//groups; got {Cg}")

        x_pad = np.pad(x, ((0,0),(0,0),(padH,padH),(padW,padW)), mode="constant") if (padH or padW) else x
        win, OH, OW = _im2col_windows(x_pad, kH, kW, strH, strW, dilH, dilW)
        cols = win.transpose(0,2,3,1,4,5).reshape(N*OH*OW, C_in*kH*kW)

        if G == 1:
            Wmat = w.reshape(C_out, C_in*kH*kW).T
            out2d = cols @ Wmat
        else:
            Cg = C_in // G; Og = C_out // G
            cols_g = win.reshape(N, G, Cg, OH, OW, kH, kW).transpose(0,3,4,1,2,5,6)\
                        .reshape(N*OH*OW, G, Cg*kH*kW)
            Wg = w.reshape(G, Og, Cg, kH, kW).reshape(G, Og, Cg*kH*kW)
            out_g2d = np.einsum("ngd,god->ngo", cols_g, Wg, optimize=True)
            out2d = out_g2d.reshape(N*OH*OW, C_out)

        out = out2d.reshape(N, OH, OW, C_out).transpose(0,3,1,2)
        if b is not None: out = out + b[None,:,None,None]

        ctx.save_for_backward(x, w, (b if b is not None else np.array([])))
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
        strH, strW = _as_int_pair(self.ctx.meta["stride"])
        padH, padW = _as_int_pair(self.ctx.meta["padding"])
        dilH, dilW = _as_int_pair(self.ctx.meta["dilation"])
        G          = _to_int(self.ctx.meta["groups"])
        (N, C_in, H, W)     = self.ctx.meta["in_shape"]
        (N2, C_out, OH, OW) = self.ctx.meta["out_shape"]
        (kH, kW)            = _as_int_pair(self.ctx.meta["k_shape"])
        has_bias            = bool(self.ctx.meta["has_bias"])
        assert N == N2

        # --- dB ---
        gB = grad_out.sum(axis=(0,2,3)) if has_bias else None

        # --- dW ---
        x_pad = np.pad(x, ((0,0),(0,0),(padH,padH),(padW,padW)), mode="constant") if (padH or padW) else x
        win, OH_chk, OW_chk = _im2col_windows(x_pad, kH, kW, strH, strW, dilH, dilW)
        assert (OH, OW) == (OH_chk, OW_chk)
        if G == 1:
            gW = np.einsum("n c h w r s, n o h w -> o c r s", win, grad_out, optimize=True)
        else:
            Cg = C_in // G; Og = C_out // G
            win_g = win.reshape(N, G, Cg, OH, OW, kH, kW)
            go_g  = grad_out.reshape(N, G, Og, OH, OW)
            gW_g  = np.einsum("n g c h w r s, n g o h w -> g o c r s", win_g, go_g, optimize=True)
            gW    = gW_g.reshape(C_out, Cg, kH, kW)

        # --- dX via conv-transpose ---
        upH = (OH - 1) * strH + 1
        upW = (OW - 1) * strW + 1
        go_up = np.zeros((N, C_out, upH, upW), dtype=grad_out.dtype)
        go_up[:, :, ::strH, ::strW] = grad_out

        # flipped, channel-swapped weights
        Og = C_out // G; Cg = C_in // G
        w_g = w.reshape(G, Og, Cg, kH, kW)
        w_flip = w_g[..., ::-1, ::-1]
        w_T = np.transpose(w_flip, (0,2,1,3,4)).reshape(C_in, Og, kH, kW)

        # transpose conv padding: p' = dil*(k-1) - p  and **output padding**:
        eff_kH = _eff_kernel(kH, dilH)
        eff_kW = _eff_kernel(kW, dilW)
        base_padH = (kH - 1) * dilH - padH
        base_padW = (kW - 1) * dilW - padW
        if base_padH < 0 or base_padW < 0:
            raise ValueError(
                f"Backward conv requires non-negative base transpose padding, got ({base_padH},{base_padW})."
            )
        out_padH = (H + 2*padH - eff_kH) % strH
        out_padW = (W + 2*padW - eff_kW) % strW

        # asymmetric pad on bottom/right to realize output_padding
        pad_top, pad_bottom = base_padH, base_padH + out_padH
        pad_left, pad_right = base_padW, base_padW + out_padW
        if pad_top or pad_bottom or pad_left or pad_right:
            go_pad = np.pad(go_up, ((0,0),(0,0),(pad_top,pad_bottom),(pad_left,pad_right)), mode="constant")
        else:
            go_pad = go_up

        win_dx, Hx, Wx = _im2col_windows(go_pad, kH, kW, 1, 1, dilH, dilW)  # stride=1
        if (Hx, Wx) != (H, W):
            raise RuntimeError(f"dx windows produced ({Hx},{Wx}) but expected input spatial ({H},{W}).")

        if G == 1:
            gX = np.einsum("n o h w r s, i o r s -> n i h w", win_dx, w_T, optimize=True)
        else:
            win_dx_g = win_dx.reshape(N, G, Og, H, W, kH, kW)
            w_T_g    = w_T.reshape(G, Cg, Og, kH, kW)
            gX_g = np.einsum("n g o h w r s, g c o r s -> n g c h w", win_dx_g, w_T_g, optimize=True)
            gX = gX_g.reshape(N, C_in, H, W)

        return (gX, gW, gB) if has_bias else (gX, gW)

def conv2d(x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
    return Conv2d.apply(x, w, b, stride, padding, dilation, groups)
