from .module import Module
from ..autograd.ops_softmax import softmax 
from ..autograd.tensor import Tensor

class Softmax(Module):
    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor):
        return softmax(x, self.axis)
