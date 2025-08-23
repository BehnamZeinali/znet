import numpy as np
from .module import Module
from ..autograd.tensor import Tensor
from ..autograd.ops_conv import conv2d

def _pair(x): return (x, x) if isinstance(x, int) else tuple(x)

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, dtype=np.float32):
        super().__init__()
        kH, kW = _pair(kernel_size)
        sH, sW = _pair(stride)
        pH, pW = _pair(padding)
        dH, dW = _pair(dilation)
        G = int(groups)

        if in_channels % G != 0 or out_channels % G != 0:
            raise ValueError("in_channels and out_channels must be divisible by groups")

        # Kaiming-like init
        fan_in = (in_channels // G) * kH * kW
        bound = np.sqrt(1.0 / max(1, fan_in))
        w = np.random.uniform(-bound, bound, size=(out_channels, in_channels // G, kH, kW)).astype(dtype)
        self.add_parameter("weight", Tensor(w, requires_grad=True, dtype=dtype))

        if bias:
            self.add_parameter("bias", Tensor(np.zeros((out_channels,), dtype=dtype), requires_grad=True))
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
