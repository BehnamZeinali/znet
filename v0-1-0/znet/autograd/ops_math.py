import mlx.core as mx
from .engine import Function
from .utils import unbroadcast

class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a + b
    def backward(self, grad_out):
        a, b = self.ctx.saved_tensors
        return unbroadcast(grad_out, a.shape), unbroadcast(grad_out, b.shape)

class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a - b
    def backward(self, grad_out):
        a, b = self.ctx.saved_tensors
        return unbroadcast(grad_out, a.shape), unbroadcast(-grad_out, b.shape)

class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a * b
    def backward(self, grad_out):
        a, b = self.ctx.saved_tensors
        ga = grad_out * b
        gb = grad_out * a
        return unbroadcast(ga, a.shape), unbroadcast(gb, b.shape)

class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a / b
    def backward(self, grad_out):
        a, b = self.ctx.saved_tensors
        ga = grad_out / b
        gb = -grad_out * a / (b ** 2)
        return unbroadcast(ga, a.shape), unbroadcast(gb, b.shape)

# Functional wrappers (optional)
def add(a, b): return Add.apply(a, b)
def sub(a, b): return Sub.apply(a, b)
def mul(a, b): return Mul.apply(a, b)
def div(a, b): return Div.apply(a, b)
