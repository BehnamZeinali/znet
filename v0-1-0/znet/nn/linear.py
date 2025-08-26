import mlx.core as mx
from .module import Module
from ..autograd.tensor import Tensor

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, dtype=mx.float32):
        super().__init__()
        # He init: N(0,1) * sqrt(2 / fan_in)
        scale = (2.0 / max(1, in_features)) ** 0.5
        w = mx.random.normal(shape=(out_features, in_features), dtype=dtype) * scale
        self.add_parameter("weight", Tensor(w, requires_grad=True, dtype=dtype))
        if bias:
            self.add_parameter("bias", Tensor(mx.zeros((out_features,), dtype=dtype), requires_grad=True))
        else:
            self.add_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., in_features), weight.T: (in_features, out_features)
        y = x @ self.weight.T
        if getattr(self, "bias", None) is not None:
            y = y + self.bias
        return y
