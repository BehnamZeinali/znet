# znet/autograd/ops_view.py
import numpy as np
from .engine import Function
from .utils import unbroadcast

class Reshape(Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        # new_shape may be list/tuple/np array — normalize
        new_shape = tuple(new_shape) if isinstance(new_shape, (list, tuple, np.ndarray)) else (new_shape,)
        ctx.meta = {"old_shape": np.shape(x)}
        return np.reshape(x, new_shape)
    def backward(self, grad_out):
        old_shape = self.ctx.meta["old_shape"]
        return grad_out.reshape(old_shape), None  # None for non-tensor arg

class Transpose(Function):
    @staticmethod
    def forward(ctx, x):
        # Simple 2D .T; if x is >2D, this mirrors NumPy default transpose reversing axes
        # For PyTorch-like .T (2D only), use x.swapaxes(-1, -2); here we mirror NumPy .T
        ctx.meta = {"axes": tuple(range(np.ndim(x)))[::-1]}
        return np.transpose(x)
    def backward(self, grad_out):
        axes = self.ctx.meta["axes"]
        # inverse permutation = same as axes for a simple reverse
        return np.transpose(grad_out, axes=axes)

class Flatten(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.meta = {"old_shape": np.shape(x)}
        return x.reshape(-1)
    def backward(self, grad_out):
        return grad_out.reshape(self.ctx.meta["old_shape"])

class Sum(Function):
    @staticmethod
    def forward(ctx, x, axis=None, keepdims=False):
        # Normalize axis: None | int | tuple[int]
        if axis is not None:
            if isinstance(axis, int):
                axis = (axis,)
            else:
                axis = tuple(axis)
        ctx.meta = {"axis": axis, "keepdims": bool(keepdims), "in_shape": np.shape(x)}
        return np.sum(x, axis=axis, keepdims=keepdims)

    def backward(self, grad_out):
        axis     = self.ctx.meta["axis"]
        keepdims = self.ctx.meta["keepdims"]
        in_shape = self.ctx.meta["in_shape"]

        g = grad_out
        if axis is None:
            # sum over all elements → each input position receives grad_out
            g = np.ones(in_shape, dtype=grad_out.dtype) * g
            return g, None, None

        # bring g to have reduced axes present if keepdims=False
        if not keepdims:
            # expand dims at the reduced axes
            for ax in sorted(axis):
                g = np.expand_dims(g, axis=ax)
        # tile/broadcast to input shape
        g = np.ones(in_shape, dtype=g.dtype) * g
        return g, None, None
