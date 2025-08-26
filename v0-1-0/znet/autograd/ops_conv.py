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

        # Use MLX's conv2d (groups supported)
        out = mx.conv2d(x, w, strides=(strH, strW), padding=(padH, padW),
                        dilation=(dilH, dilW), groups=G)
        if b is not None:
            out = out + b[None, :, None, None]

        ctx.save_for_backward(x, w, (b if b is not None else mx.array([])))
        ctx.meta.update({
            "stride": (strH, strW),
            "padding": (padH, padW),
            "dilation": (dilH, dilW),
            "groups": G,
            "in_shape": (N, C_in, H, W),
            "out_shape": out.shape,
            "k_shape": (kH, kW),
            "has_bias": (b is not None),
        })
        return out

    def backward(self, grad_out):
        x, w, b = self.ctx.saved_tensors
        strH, strW = _as_int_pair(self.ctx.meta["stride"])
        padH, padW = _as_int_pair(self.ctx.meta["padding"])
        dilH, dilW = _as_int_pair(self.ctx.meta["dilation"])
        G          = _to_int(self.ctx.meta["groups"])
        has_bias   = bool(self.ctx.meta["has_bias"])

        # Use MLX's conv2d_backward to compute gradients
        gX, gW = mx.conv2d_backward(x, w, grad_out,
                                    strides=(strH, strW),
                                    padding=(padH, padW),
                                    dilation=(dilH, dilW),
                                    groups=G)
        gB = mx.sum(grad_out, axis=(0,2,3)) if has_bias else None

        return (gX, gW, gB) if has_bias else (gX, gW)

def conv2d(x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
    return Conv2d.apply(x, w, b, stride, padding, dilation, groups)
