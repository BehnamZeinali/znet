import numpy as np
from .module import Module
from ..autograd.tensor import Tensor
class ReLU(Module):
    def forward(self, x: Tensor):
        out_data = np.maximum(0, x.data)
        out = Tensor(out_data, requires_grad=x.requires_grad)

        def _grad_fn(grad_output):
            return grad_output * (x.data > 0).astype(np.float32)

        out._backward = lambda: None  # Just a placeholder, not used in this design
        out._prev = [x]
        out._grad_fn = _grad_fn
        return out
