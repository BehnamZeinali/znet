# znet/autograd/ops_conv_2.py
import mlx.core as mx
from .engine import Function

# ---------- helpers ----------
def _pair(x):
    return (x, x) if isinstance(x, int) else tuple(x)

def _as_int_pair(v):
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return int(v[0]), int(v[1])
    return int(v), int(v)

def _to_int(x):
    if hasattr(x, "item"):
        return int(x.item())
    return int(x)

def _eff_kernel(k, d):
    return 1 + (_to_int(k) - 1) * _to_int(d)

def _pad2d(x, padH, padW):
    if padH == 0 and padW == 0:
        return x
    return mx.pad(x, ((0, 0), (0, 0), (padH, padH), (padW, padW)))

def _im2col_mx(x_pad, kH, kW, strH, strW, dilH, dilW):
    """
    x_pad: (N, C, Hpad, Wpad)
    Returns:
      cols: (N*OH*OW, C*kH*kW)
      win:  (N, C, OH, OW, kH, kW)
      OH, OW
    """
    N, C, Hpad, Wpad = x_pad.shape
    eff_kH = _eff_kernel(kH, dilH)
    eff_kW = _eff_kernel(kW, dilW)

    OH = (Hpad - eff_kH) // strH + 1
    OW = (Wpad - eff_kW) // strW + 1
    if OH <= 0 or OW <= 0:
        raise ValueError(
            f"Invalid output spatial shape OH={OH}, OW={OW} "
            f"(Hpad={Hpad}, Wpad={Wpad}, eff=({eff_kH},{eff_kW}), stride=({strH},{strW}))"
        )

    patches = []
    for r in range(kH):
        r0 = r * dilH
        r_slice = slice(r0, r0 + OH * strH, strH)
        for s in range(kW):
            s0 = s * dilW
            s_slice = slice(s0, s0 + OW * strW, strW)
            patches.append(x_pad[:, :, r_slice, s_slice])  # (N, C, OH, OW)

    win = mx.stack(patches, axis=-1)                  # (N, C, OH, OW, kH*kW)
    win = mx.reshape(win, (N, C, OH, OW, kH, kW))     # (N, C, OH, OW, kH, kW)
    cols = mx.reshape(mx.transpose(win, (0, 2, 3, 1, 4, 5)), (N * OH * OW, C * kH * kW))
    return cols, win, OH, OW

def _upsample_with_zeros(g, strH, strW):
    """
    Insert (strH-1) zeros between rows and (strW-1) zeros between cols.
    Output size: upH=(OH-1)*strH+1, upW=(OW-1)*strW+1
    """
    N, C, OH, OW = g.shape
    g_exp = mx.reshape(g, (N, C, OH, 1, OW, 1))  # (N,C,OH,1,OW,1)
    pad_h = (0, strH - 1) if strH > 1 else (0, 0)
    pad_w = (0, strW - 1) if strW > 1 else (0, 0)
    g_pad = mx.pad(g_exp, ((0, 0), (0, 0), (0, 0), pad_h, (0, 0), pad_w))  # (N,C,OH,strH,OW,strW)
    up_full = mx.reshape(g_pad, (N, C, OH * strH, OW * strW))
    upH = (OH - 1) * strH + 1
    upW = (OW - 1) * strW + 1
    return up_full[:, :, :upH, :upW]

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

        x_pad = _pad2d(x, padH, padW)

        if G == 1:
            cols, _, OH, OW = _im2col_mx(x_pad, kH, kW, strH, strW, dilH, dilW)
            Wmat = mx.reshape(w, (C_out, C_in * kH * kW))
            out2d = cols @ mx.transpose(Wmat, (1, 0))  # (N*OH*OW, C_out)
        else:
            cols, _, OH, OW = _im2col_mx(x_pad, kH, kW, strH, strW, dilH, dilW)
            Cg = C_in // G
            Og = C_out // G
            cols_g = mx.reshape(cols, (N * OH * OW, G, Cg * kH * kW))
            cols_g = mx.transpose(cols_g, (1, 0, 2))           # (G, N*, CgK)
            Wg = mx.reshape(w, (G, Og, Cg, kH, kW))
            Wg = mx.reshape(Wg, (G, Og, Cg * kH * kW))         # (G, Og, CgK)
            out_g = cols_g @ mx.transpose(Wg, (0, 2, 1))       # (G, N*, Og)
            out2d = mx.reshape(mx.transpose(out_g, (1, 0, 2)), (N * OH * OW, C_out))

        out = mx.reshape(out2d, (N, OH, OW, C_out))
        out = mx.transpose(out, (0, 3, 1, 2))                  # (N, C_out, OH, OW)
        if b is not None:
            out = out + mx.reshape(b, (1, -1, 1, 1))

        ctx.save_for_backward(x, w, (b if b is not None else mx.array([])))
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
        gB = mx.sum(grad_out, axis=(0, 2, 3)) if has_bias else None

        # --- dW ---
        x_pad = _pad2d(x, padH, padW)
        cols, _, OH_chk, OW_chk = _im2col_mx(x_pad, kH, kW, strH, strW, dilH, dilW)
        if (OH, OW) != (OH_chk, OW_chk):
            raise RuntimeError("Internal OH/OW mismatch.")
        go2d = mx.reshape(mx.transpose(grad_out, (0, 2, 3, 1)), (N * OH * OW, C_out))
        if G == 1:
            gWmat = mx.transpose(cols, (1, 0)) @ go2d           # (C_in*kH*kW, C_out)
            gW = mx.reshape(mx.transpose(gWmat, (1, 0)), (C_out, C_in, kH, kW))
        else:
            Cg = C_in // G; Og = C_out // G
            cols_g = mx.reshape(cols, (N * OH * OW, G, Cg * kH * kW))
            cols_g = mx.transpose(cols_g, (1, 0, 2))            # (G, N*, CgK)
            go_g   = mx.reshape(go2d, (N * OH * OW, G, Og))
            go_g   = mx.transpose(go_g, (1, 0, 2))              # (G, N*, Og)
            gW_g   = (mx.transpose(cols_g, (0, 2, 1)) @ go_g)   # (G, CgK, Og)
            gW     = mx.reshape(mx.transpose(gW_g, (0, 2, 1)), (G * Og, Cg, kH, kW))

        # --- dX via conv-transpose ---
        # 1) upsample grad_out
        go_up = _upsample_with_zeros(grad_out, strH, strW)      # (N, C_out, upH, upW)

        # 2) compute transpose-conv padding (include output_padding!) and pad asymmetrically
        eff_kH = _eff_kernel(kH, dilH)
        eff_kW = _eff_kernel(kW, dilW)
        base_padH = (kH - 1) * dilH - padH
        base_padW = (kW - 1) * dilW - padW
        if base_padH < 0 or base_padW < 0:
            raise ValueError(
                f"Backward conv requires non-negative base transpose padding, got ({base_padH},{base_padW})."
            )
        out_padH = (H + 2 * padH - eff_kH) % strH
        out_padW = (W + 2 * padW - eff_kW) % strW

        pad_top, pad_bottom = base_padH, base_padH + out_padH
        pad_left, pad_right = base_padW, base_padW + out_padW
        if pad_top or pad_bottom or pad_left or pad_right:
            go_pad = mx.pad(go_up, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)))
        else:
            go_pad = go_up

        # 3) extract windows (stride=1) and compute gX
        _, win_dx, Hx, Wx = _im2col_mx(go_pad, kH, kW, 1, 1, dilH, dilW)
        if (Hx, Wx) != (H, W):
            raise RuntimeError(f"dx windows produced ({Hx},{Wx}) but expected ({H},{W}).")

        if G == 1:
            # cols over go_pad per (h,w): (N*H*W, C_out*kH*kW)
            cols_dx = mx.reshape(mx.transpose(win_dx, (0, 2, 3, 1, 4, 5)), (N * H * W, C_out * kH * kW))
            # flip spatial and reshape weights for matmul
            w_flip = w[..., ::-1, ::-1]                           # (C_out, C_in, kH, kW)
            wT_mat = mx.reshape(w_flip, (C_out * kH * kW, C_in))  # (C_out*kH*kW, C_in)
            gX2d = cols_dx @ wT_mat                               # (N*H*W, C_in)
            gX = mx.reshape(gX2d, (N, H, W, C_in))
            gX = mx.transpose(gX, (0, 3, 1, 2))                   # (N, C_in, H, W)
        else:
            # Grouped transposed conv via batch matmul
            Og = C_out // G; Cg = C_in // G
            # win_dx: (N, C_out, H, W, kH, kW) -> (N, G, Og, H, W, kH, kW)
            win_dx_g = mx.reshape(win_dx, (N, G, Og, H, W, kH, kW))
            # Build cols per pixel: (G, N*H*W, Og*kH*kW)
            cols_dx_g = mx.transpose(win_dx_g, (0, 3, 4, 1, 2, 5, 6))  # (N,H,W,G,Og,kH,kW)
            cols_dx_g = mx.reshape(cols_dx_g, (N * H * W, G, Og * kH * kW))
            cols_dx_g = mx.transpose(cols_dx_g, (1, 0, 2))             # (G, N*, OgK)

            w_g = mx.reshape(w, (G, Og, Cg, kH, kW))
            w_flip = w_g[..., ::-1, ::-1]                              # (G, Og, Cg, kH, kW)
            w_T_g = mx.reshape(mx.transpose(w_flip, (0, 1, 3, 4, 2)), (G, Og * kH * kW, Cg))  # (G, OgK, Cg)

            gX_g2d = cols_dx_g @ w_T_g                                 # (G, N*, Cg)
            gX_g2d = mx.transpose(gX_g2d, (1, 0, 2))                   # (N*, G, Cg)
            gX = mx.reshape(gX_g2d, (N, H, W, G, Cg))
            gX = mx.transpose(gX, (0, 3, 4, 1, 2))
            gX = mx.reshape(gX, (N, C_in, H, W))

        return (gX, gW, gB) if has_bias else (gX, gW)

def conv2d(x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
    return Conv2d.apply(x, w, b, stride, padding, dilation, groups)
