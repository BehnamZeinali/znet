# znet/autograd/ops_view.py
import mlx.core as mx
from .engine import Function
from .utils import unbroadcast

class Reshape(Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        # new_shape may be list/tuple/array — normalize
        new_shape = tuple(new_shape) if isinstance(new_shape, (list, tuple)) else (new_shape,)
        ctx.meta = {"old_shape": x.shape}
        return mx.reshape(x, new_shape)
    def backward(self, grad_out):
        old_shape = self.ctx.meta["old_shape"]
        return mx.reshape(grad_out, old_shape), None  # None for non-tensor arg

class Transpose(Function):
    @staticmethod
    def forward(ctx, x):
        # NumPy-like transpose: reverse axes by default
        axes = tuple(range(len(x.shape)))[::-1]
        ctx.meta = {"axes": axes}
        return mx.transpose(x, axes=axes)
    def backward(self, grad_out):
        axes = self.ctx.meta["axes"]
        # inverse permutation = same as axes for a simple reverse
        return mx.transpose(grad_out, axes=axes),

class Flatten(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.meta = {"old_shape": x.shape}
        return mx.reshape(x, (-1,))
    def backward(self, grad_out):
        return mx.reshape(grad_out, self.ctx.meta["old_shape"]),

class Sum(Function):
    @staticmethod
    def forward(ctx, x, axis=None, keepdims=False):
        # Normalize axis
        if axis is not None:
            if isinstance(axis, int):
                axis = (axis,)
            else:
                axis = tuple(axis)
        ctx.meta = {"axis": axis, "keepdims": bool(keepdims), "in_shape": x.shape}
        return mx.sum(x, axis=axis, keepdims=keepdims)

    def backward(self, grad_out):
        axis     = self.ctx.meta["axis"]
        keepdims = self.ctx.meta["keepdims"]
        in_shape = self.ctx.meta["in_shape"]

        g = grad_out
        if axis is None:
            # sum over all elements → each input position receives grad_out
            g = mx.ones(in_shape, dtype=grad_out.dtype) * g
            return g, None, None

        # bring g to have reduced axes present if keepdims=False
        if not keepdims:
            if isinstance(axis, tuple):
                for ax in sorted(axis):
                    g = mx.expand_dims(g, axis=ax)
            else:
                g = mx.expand_dims(g, axis=axis)
        # broadcast to input shape
        g = mx.broadcast_to(g, in_shape)
        return g, None, None
