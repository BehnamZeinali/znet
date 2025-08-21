# znet/nn/linear.py
import numpy as np
from .module import Module
from ..autograd.tensor import Tensor

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, dtype=np.float32):
        super().__init__()
        # Keep your (out_features, in_features) layout; use Tensor ctor to own the backend array
        w = (np.random.randn(out_features, in_features).astype(dtype)
             * np.sqrt(2.0 / max(1, in_features)))
        self.add_parameter("weight", Tensor(w, requires_grad=True, dtype=dtype))
        if bias:
            self.add_parameter("bias", Tensor(np.zeros((out_features,), dtype=dtype), requires_grad=True))
        else:
            self.add_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        # Compose **Tensor ops** only; this keeps you engine-agnostic
        # x: (..., in_features), weight.T: (in_features, out_features)
        y = x @ self.weight.T             # uses your Matmul Function
        if getattr(self, "bias", None) is not None:
            y = y + self.bias             # Add Function with broadcasting
        return y
