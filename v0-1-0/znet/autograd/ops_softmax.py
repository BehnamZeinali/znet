import numpy as np
from .engine import Function

class Softmax(Function):
    @staticmethod
    def forward(ctx, x, axis=-1):
        m = np.max(x, axis=axis, keepdims=True)
        e = np.exp(x - m)
        y = e / np.sum(e, axis=axis, keepdims=True)
        ctx.save_for_backward(y)
        ctx.meta["axis"] = axis
        return y

    def backward(self, g_out):
        (y,) = self.ctx.saved_tensors
        axis = self.ctx.meta["axis"]
        # J^T g = y * (g - <g,y>)
        gy = g_out * y
        s = np.sum(gy, axis=axis, keepdims=True)
        g_in = (g_out - s) * y
        return g_in   # ✅ ONLY return grad for the Tensor input

def softmax(input, axis=-1):
    return Softmax.apply(input , axis)

