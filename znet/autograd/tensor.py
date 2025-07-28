import numpy as np
class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self.requires_grad = requires_grad

        self._backward = lambda: None
        self._prev = []
        self._grad_fn = None

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        if grad is None:
            grad = np.ones_like(self.data)

        self.grad += grad

        self._backward()
        for t in self._prev:
            t.backward(t._grad_fn(self.grad))

    def set_backward(self, fn, prev):
        self._grad_fn = fn
        self._prev = prev