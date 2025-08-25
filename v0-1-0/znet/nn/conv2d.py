
# znet/nn/conv.py  (or your existing path)
from __future__ import annotations
import torch as th
from .module import Module
from ..autograd.tensor import Tensor
from ..autograd.ops_conv import conv2d

def _pair(x): return (x, x) if isinstance(x, int) else tuple(int(v) for v in x)

# minimal dtype resolver (accepts numpy dtypes/strings/torch dtypes)
def _resolve_dtype(dtype):
    if isinstance(dtype, th.dtype):
        return dtype
    if dtype is None:
        return th.float32
    try:
        import numpy as np
        return {
            np.float16: th.float16, np.float32: th.float32, np.float64: th.float64,
            np.int8: th.int8, np.int16: th.int16, np.int32: th.int32, np.int64: th.int64,
            np.uint8: th.uint8, np.bool_: th.bool, bool: th.bool
        }[np.dtype(dtype).type]
    except Exception:
        s = str(dtype).lower()
        return {
            "half": th.float16, "float16": th.float16, "fp16": th.float16,
            "float": th.float32, "float32": th.float32, "fp32": th.float32,
            "double": th.float64, "float64": th.float64, "fp64": th.float64,
            "int8": th.int8, "int16": th.int16, "int32": th.int32, "int64": th.int64,
            "uint8": th.uint8, "bool": th.bool
        }.get(s, th.float32)

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, dtype=th.float32):
        super().__init__()
        kH, kW = _pair(kernel_size)
        sH, sW = _pair(stride)
        pH, pW = _pair(padding)
        dH, dW = _pair(dilation)
        G = int(groups)

        if in_channels % G != 0 or out_channels % G != 0:
            raise ValueError("in_channels and out_channels must be divisible by groups")

        tdtype = _resolve_dtype(dtype)

        # Kaiming-like init matching your formula: U(-sqrt(1/fan_in), +sqrt(1/fan_in))
        fan_in = (in_channels // G) * kH * kW
        bound = (1.0 / max(1, fan_in)) ** 0.5

        w = th.empty(out_channels, in_channels // G, kH, kW, dtype=tdtype)
        w.uniform_(-bound, bound)
        self.add_parameter("weight", Tensor(w, requires_grad=True, dtype=tdtype))

        if bias:
            b = th.zeros(out_channels, dtype=tdtype)
            self.add_parameter("bias", Tensor(b, requires_grad=True, dtype=tdtype))
        else:
            self.add_parameter("bias", None)

        # store hyperparams
        self.stride   = (sH, sW)
        self.padding  = (pH, pW)
        self.dilation = (dH, dW)
        self.groups   = G

    def forward(self, x: Tensor) -> Tensor:
        return conv2d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups
        )
