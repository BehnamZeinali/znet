from .module import Module
from ..autograd.tensor import Tensor
from ..autograd.ops_activation import relu as relu_op

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return relu_op(x)

