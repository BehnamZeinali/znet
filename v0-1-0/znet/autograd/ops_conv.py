# znet/autograd/ops_conv.py
import mlx.core as mx
from .engine import Function

# --------- small helpers ----------
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

def _flip_hw(w):  # flip spatial dims (… , KH, KW, …)
    return w[:, ::-1, ::-1, :]
def _upsample2d(x, stride_hw):
    """
    Insert (s-1) zeros between samples *and* pad (s-1) zeros at the end so the
    result has shape (N, H*s, W*s, C). This matches conv_transpose geometry.
    """
    sH, sW = stride_hw
    if sH == 1 and sW == 1:
        return x

    N, H, W, C = x.shape
    # First: classic repeat -> size (H1, W1) = ((H-1)*s+1, (W-1)*s+1)
    H1 = (H - 1) * sH + 1
    W1 = (W - 1) * sW + 1
    xr = mx.repeat(x, sH, axis=1)
    xr = mx.repeat(xr, sW, axis=2)
    xr = xr[:, :H1, :W1, :]  # trim tail if any

    # Then: pad trailing rows/cols to reach H*s and W*s
    Hs = H * sH
    Ws = W * sW
    if Hs > H1:
        pad_bottom = mx.zeros((N, Hs - H1, W1, C), dtype=x.dtype)
        xr = mx.concatenate([xr, pad_bottom], axis=1)
    if Ws > W1:
        pad_right = mx.zeros((N, xr.shape[1], Ws - W1, C), dtype=x.dtype)
        xr = mx.concatenate([xr, pad_right], axis=2)

    # Finally: zero out the interleaved slots, keep only indices % s == 0
    mh = (mx.arange(Hs) % sH == 0).astype(x.dtype)  # (Hs,)
    mw = (mx.arange(Ws) % sW == 0).astype(x.dtype)  # (Ws,)
    xr = xr * mh[None, :, None, None] * mw[None, None, :, None]
    return xr

def _pad2d(x, pad_hw):
    """Symmetric padding on H,W axes for NHWC."""
    pH, pW = pad_hw
    if pH == 0 and pW == 0:
        return x
    # pad spec is ((N0,N1),(H0,H1),(W0,W1),(C0,C1))
    return mx.pad(x, ((0,0), (pH,pH), (pW,pW), (0,0)))
def _ceil_div(a, b):
    # integer ceil(a/b) for possibly negative a
    return -((-a) // b)

def _floor_div(a, b):
    # integer floor(a/b) for possibly negative a
    return a // b

def _valid_out_range(H_in, H_out, stride, offset):
    """
    For mapping h_in = offset + stride * h_out, find the integer range of h_out
    such that 0 <= h_in < H_in. Returns (h_out_start, h_out_end) with end exclusive.
    """
    # Lower bound: h_out >= ceil_div(-offset, stride)
    lo = max(0, _ceil_div(-offset, stride))
    # Upper bound: h_out <= floor_div((H_in - 1 - offset), stride)
    hi = min(H_out, _floor_div(H_in - 1 - offset, stride) + 1)
    return lo, hi

# ---------- Conv2d op (PyTorch-like; MLX backend) ----------
class Conv2d(Function):
    @staticmethod
    def forward(ctx, x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
        # Caller layout:
        #   x: (N,C,H,W)         w: (C_out, C_in/groups, KH, KW)   b: (C_out,)
        strH, strW = _as_int_pair(_pair(stride))
        padH, padW = _as_int_pair(_pair(padding))
        dilH, dilW = _as_int_pair(_pair(dilation))
        G = _to_int(groups)

        if x.ndim != 4: raise ValueError(f"x (N,C,H,W) expected, got {x.shape}")
        if w.ndim != 4: raise ValueError(f"weight (C_out, Cg, KH, KW) expected, got {w.shape}")
        if b is not None and b.ndim != 1: raise ValueError(f"bias (C_out,) expected, got {b.shape}")

        N, C_in, H, W = x.shape
        C_out, Cg, KH, KW = w.shape
        if C_in % G or C_out % G:
            raise ValueError("in/out channels must be divisible by groups")
        if Cg != C_in // G:
            raise ValueError(f"weight second dim must be C_in//groups; got {Cg}")

        # ---- Reorder to MLX layout ----
        x_nhwc = mx.transpose(x, (0, 2, 3, 1))            # NCHW -> NHWC
        # w: (O, Cin_g, KH, KW) -> (O, KH, KW, Cin_g)
        w_oihw = mx.transpose(w, (0, 2, 3, 1))

        # ---- Forward ----
        y_nhwc = mx.conv2d(
            x_nhwc, w_oihw,
            stride=(strH, strW),            # NOTE: 'stride', not 'strides'
            padding=(padH, padW),
            dilation=(dilH, dilW),
            groups=G
        )
        if b is not None:
            y_nhwc = y_nhwc + b[None, None, None, :]

        y = mx.transpose(y_nhwc, (0, 3, 1, 2))            # NHWC -> NCHW

        # save for backward (original layouts + meta)
        ctx.save_for_backward(x, w, (b if b is not None else mx.array([])))
        ctx.meta.update({
            "stride": (strH, strW),
            "padding": (padH, padW),
            "dilation": (dilH, dilW),
            "groups": G,
            "has_bias": (b is not None),
            "KH": KH, "KW": KW,
        })
        return y

    # --- inside your Conv2d(Function) ---
    def backward(self, grad_out):
        # Saved from forward
        x, w, b = self.ctx.saved_tensors
        strH, strW = self.ctx.meta["stride"]
        padH, padW = self.ctx.meta["padding"]
        dilH, dilW = self.ctx.meta["dilation"]
        G          = int(self.ctx.meta["groups"])
        has_bias   = bool(self.ctx.meta["has_bias"])
        KH         = int(self.ctx.meta["KH"])
        KW         = int(self.ctx.meta["KW"])

        # Shapes
        N, Cin, Hin, Win = x.shape
        Cout, CinG, _, _ = w.shape
        assert Cin % G == 0 and Cout % G == 0
        CinG  = Cin // G
        CoutG = Cout // G

        # NHWC views for MLX primitives
        x_nhwc  = mx.transpose(x,       (0, 2, 3, 1))   # (N,Hin,Win,Cin)
        go_nhwc = mx.transpose(grad_out,(0, 2, 3, 1))   # (N,Hout,Wout,Cout)
        # weights to MLX layout (O, KH, KW, CinG)
        w_oihw  = mx.transpose(w, (0, 2, 3, 1))

        # ===== gB =====
        gB = mx.sum(go_nhwc, axis=(0, 1, 2)) if has_bias else None  # (Cout,)

        # ===== gX (conv-transpose via conv2d) =====
        # upsample by stride (insert zeros)
        go_up = _upsample2d(go_nhwc, (strH, strW))  # (N, Hup, Wup, Cout)

        # flip kernel spatially and swap in/out per group
        w_flip = _flip_hw(w_oihw)                   # (Cout, KH, KW, CinG)
        parts = []
        for g in range(G):
            o0, o1 = g*CoutG, (g+1)*CoutG
            i0, i1 = g*CinG,  (g+1)*CinG
            wg = w_flip[o0:o1, :, :, :]             # (CoutG, KH, KW, CinG)
            parts.append(mx.transpose(wg, (3,1,2,0)))  # (CinG, KH, KW, CoutG)
        w_T = mx.concatenate(parts, axis=0)         # (Cin, KH, KW, Cout/G)

        # padding for conv-transpose equivalence
        padT_h = (KH - 1) * dilH - padH
        padT_w = (KW - 1) * dilW - padW
        go_up_padded = _pad2d(go_up, (padT_h, padT_w))

        gX_nhwc = mx.conv2d(
            go_up_padded, w_T,
            stride=(1, 1),
            padding=(0, 0),
            dilation=(dilH, dilW),
            groups=G
        )
        gX = mx.transpose(gX_nhwc, (0, 3, 1, 2))    # (N,Cin,Hin,Win)

        # ===== gW (accumulate with exact slice bounds; no boolean masks) =====
        Hout = go_nhwc.shape[1]
        Wout = go_nhwc.shape[2]
        # Build in (Cout, KH, KW, CinG) then transpose back
        gW_rows = []  # each (Cout, 1, KW, CinG)
        for ky in range(KH):
            off_h = ky * dilH - padH
            h_out0, h_out1 = _valid_out_range(Hin, Hout, strH, off_h)
            row_cols = []
            for kx in range(KW):
                off_w = kx * dilW - padW
                w_out0, w_out1 = _valid_out_range(Win, Wout, strW, off_w)

                Lh = h_out1 - h_out0
                Lw = w_out1 - w_out0
                if Lh <= 0 or Lw <= 0:
                    row_cols.append(mx.zeros((Cout, 1, 1, CinG), dtype=w.dtype))
                    continue

                # input slice starts and bounded stops to match Lh/Lw exactly
                h_in0 = off_h + strH * h_out0
                w_in0 = off_w + strW * w_out0
                h_stop = h_in0 + strH * Lh
                w_stop = w_in0 + strW * Lw

                # (N, Lh, Lw, Cout), (N, Lh, Lw, Cin)
                go_sel = go_nhwc[:, h_out0:h_out1, w_out0:w_out1, :]
                x_sel  = x_nhwc[:,  h_in0:h_stop: strH, w_in0:w_stop: strW, :]

                # Contract per group -> (CoutG, CinG)
                parts = []
                for g in range(G):
                    i0, i1 = g*CinG,  (g+1)*CinG
                    o0, o1 = g*CoutG, (g+1)*CoutG
                    Xg = x_sel[:, :, :, i0:i1].reshape((-1, CinG))     # (N*Lh*Lw, CinG)
                    Gg = go_sel[:, :, :, o0:o1].reshape((-1, CoutG))   # (N*Lh*Lw, CoutG)
                    GW = mx.transpose(Gg) @ Xg                          # (CoutG, CinG)
                    parts.append(GW)
                GW_all = mx.concatenate(parts, axis=0)                  # (Cout, CinG)

                row_cols.append(GW_all[:, None, None, :])               # (Cout,1,1,CinG)

            gW_rows.append(mx.concatenate(row_cols, axis=2))            # (Cout,1,KW,CinG)

        gW_oihw = mx.concatenate(gW_rows, axis=1)                       # (Cout,KH,KW,CinG)
        gW = mx.transpose(gW_oihw, (0, 3, 1, 2))                        # (Cout,CinG,KH,KW)

        return (gX, gW, gB) if has_bias else (gX, gW)



      

def conv2d(x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
    return Conv2d.apply(x, w, b, stride, padding, dilation, groups)
