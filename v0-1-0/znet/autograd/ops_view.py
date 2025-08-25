# znet/autograd/ops_view.py
from __future__ import annotations
import torch as th
from .engine import Function

def _to_tuple_shape(new_shape):
    # Accept int | list | tuple | torch.Size | any iterable of ints
    if isinstance(new_shape, int):
        return (new_shape,)
    if isinstance(new_shape, (list, tuple)):
        return tuple(int(s) for s in new_shape)
    if isinstance(new_shape, th.Size):
        return tuple(new_shape)
    try:
        return tuple(int(s) for s in new_shape)
    except TypeError:
        return (int(new_shape),)

class Reshape(Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        new_shape = _to_tuple_shape(new_shape)
        ctx.meta = {"old_shape": tuple(x.shape)}
        return x.reshape(new_shape)
    def backward(self, grad_out):
        old_shape = self.ctx.meta["old_shape"]
        return grad_out.reshape(old_shape), None  # None for non-tensor arg

class Transpose(Function):
    @staticmethod
    def forward(ctx, x):
        # NumPy-like .T: reverse all axes
        nd = x.ndim
        perm = tuple(range(nd - 1, -1, -1))  # e.g., (2,1,0) for 3D
        ctx.meta = {"perm": perm}
        return x.permute(*perm)
    def backward(self, grad_out):
        perm = self.ctx.meta["perm"]
        # inverse of a full reverse is the same reverse
        return grad_out.permute(*perm),

class Flatten(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.meta = {"old_shape": tuple(x.shape)}
        return x.reshape(-1)
    def backward(self, grad_out):
        return grad_out.reshape(self.ctx.meta["old_shape"]),

class Sum(Function):
    @staticmethod
    def forward(ctx, x, axis=None, keepdims=False):
        # Normalize axis to a tuple of positive indices for backward
        in_shape = tuple(x.shape)
        if axis is None:
            axis_norm = None
        else:
            if isinstance(axis, int):
                axis = (axis,)
            n = len(in_shape)
            axis_norm = tuple(sorted(((ax + n) % n) for ax in axis))
        ctx.meta = {"axis": axis_norm, "keepdims": bool(keepdims), "in_shape": in_shape}
        return th.sum(x, dim=axis_norm, keepdim=keepdims)

    def backward(self, grad_out):
        axis     = self.ctx.meta["axis"]
        keepdims = self.ctx.meta["keepdims"]
        in_shape = self.ctx.meta["in_shape"]

        g = grad_out
        if axis is None:
            # sum over all -> each input element gets grad_out
            g = th.ones(in_shape, dtype=g.dtype, device=g.device) * g
            return g, None, None

        # If keepdims=False, we must unsqueeze along reduced axes first
        if not keepdims:
            for ax in axis:
                g = g.unsqueeze(ax)
        # Broadcast to input shape (expand is cheaper than materialize; multiply by 1 to materialize if needed)
        g = g.expand(in_shape)
        return g, None, None
