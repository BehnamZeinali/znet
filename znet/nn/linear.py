import numpy as np
from .module import Module
from ..autograd.tensor import Tensor
class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Tensor(np.random.randn(out_features, in_features) * np.sqrt(2. / in_features), requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)
        self._parameters = [
            {'value': self.weight.data, 'grad': self.weight.grad},
            {'value': self.bias.data, 'grad': self.bias.grad},
        ]

    def forward(self, x: Tensor):
        out_data = x.data @ self.weight.data.T + self.bias.data
        out = Tensor(out_data, requires_grad=x.requires_grad or self.weight.requires_grad)

        def _grad_fn(grad_output):
            self.weight.grad += grad_output.T @ x.data
            self.bias.grad += grad_output.sum(axis=0)
            return grad_output @ self.weight.data

        out.set_backward(_grad_fn, [x])
        return out