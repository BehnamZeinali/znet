from .engine import Function
from .utils import unbroadcast_like as unbroadcast



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

# class Matmul(Function):
#     @staticmethod
#     def forward(ctx, a, b):
#         if a.ndim < 2 or b.ndim < 2:
#             raise RuntimeError("matmul expects tensors with rank >= 2 on both inputs")
#         y = a @ b  # NumPy matmul handles broadcasting over leading dims
#         ctx.a = a; ctx.b = b
#         return y

#     def backward(self, grad_out):
#         a, b = self.ctx.a, self.ctx.b
#         # Raw grads in broadcasted shape
#         ga = grad_out @ np.swapaxes(b, -1, -2)
#         gb = np.swapaxes(a, -1, -2) @ grad_out
#         # Reduce back over broadcasted batch dims
#         ga = unbroadcast(ga, a.shape)
#         gb = unbroadcast(gb, b.shape)
#         return ga, gb


# def matmul(a, b): return Matmul.apply(a, b)