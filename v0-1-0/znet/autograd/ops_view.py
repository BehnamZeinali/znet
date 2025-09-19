# znet/autograd/ops_view.py
import numpy as np
from .engine import Function
from .utils import unbroadcast


class View(Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        # normalize shape
        if not isinstance(new_shape, tuple):
            new_shape = tuple(new_shape) if isinstance(new_shape, (list, np.ndarray)) else (new_shape,)
        # must be contiguous (PyTorch requirement)
        if not x.flags["C_CONTIGUOUS"]:
            raise RuntimeError("view(): input is not contiguous; call contiguous() or use reshape().")

        out = np.reshape(x, new_shape)  # may or may not share memory in NumPy
        # enforce true view (no copy)
        if not np.shares_memory(out, x):
            raise RuntimeError("view(): requested shape would require a copy; use reshape().")

        ctx.meta["old_shape"] = x.shape
        return out

    def backward(self, grad_out):
        old_shape = self.ctx.meta["old_shape"]
        return grad_out.reshape(old_shape), None

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
    def forward(ctx, x, dim0=-1, dim1=-2):
        nd = np.ndim(x)
        d0 = dim0 % nd; d1 = dim1 % nd
        axes = list(range(nd))
        axes[d0], axes[d1] = axes[d1], axes[d0]
        ctx.meta["axes"] = tuple(axes)
        return np.transpose(x, axes=axes)

    def backward(self, grad_out):
        axes = self.ctx.meta["axes"]
        # swapping two dims is its own inverse, so reuse the same axes
        return np.transpose(grad_out, axes=axes), None, None



import numpy as np
from .engine import Function

class Cat(Function):
    @staticmethod
    def forward(ctx, *args):
        *xs, dim = args                      # last arg is axis
        arrs = [ (x.data if hasattr(x, "data") else np.asarray(x)) for x in xs ]
        nd = arrs[0].ndim
        axis = int(dim) % nd

        # validate shapes (match on all dims except `axis`)
        ref = arrs[0].shape
        sizes = []
        for a in arrs:
            if a.ndim != nd:
                raise ValueError("cat: all tensors must have same ndim")
            if any(i != axis and a.shape[i] != ref[i] for i in range(nd)):
                raise ValueError("cat: shapes must match except along dim")
            sizes.append(a.shape[axis])

        ctx.meta["axis"] = axis
        ctx.meta["sizes"] = sizes
        return np.concatenate(arrs, axis=axis)

    def backward(self, grad_out):
        axis  = self.ctx.meta["axis"]
        sizes = self.ctx.meta["sizes"]
        cuts = np.cumsum(sizes)[:-1]
        parts = np.split(grad_out, cuts, axis=axis)
        return (*parts, None)   # one grad per input, None for dim

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
    
class SwapAxes(Function):
    @staticmethod
    def forward(ctx, x, axis1, axis2):
        # normalize negative axes
        ndim = x.ndim
        a1 = int(axis1) % ndim
        a2 = int(axis2) % ndim

        ctx.meta = {"axis1": a1, "axis2": a2}
        return np.swapaxes(x, a1, a2)

    def backward(self, grad_out):
        a1 = self.ctx.meta["axis1"]
        a2 = self.ctx.meta["axis2"]
        if a1 == a2:
            # no-op swap
            return grad_out, None, None
        # swap is its own inverse
        return np.swapaxes(grad_out, a1, a2), None, None
    



# ... your existing Reshape / Transpose / Flatten / Sum / SwapAxes ...

# znet/autograd/ops_view.py

class Index(Function):
    
    @staticmethod
    def forward(ctx, x, index):
        x_arr = np.asarray(x.data)        # <- ensures ndarray (not memoryview)
        ctx.meta["in_shape"] = x_arr.shape
        ctx.meta["index"] = index
        out = x_arr[index]
        return out
    @staticmethod
    def backward(ctx, grad_out):
        in_shape = ctx.meta["in_shape"]
        index = ctx.meta["index"]

        g_in = np.zeros(in_shape, dtype=grad_out.dtype)

        # Accumulate gradients into the sliced region
        # - basic slices: single assignment works
        # - boolean mask: += works
        # - integer arrays (possibly repeated): use add.at to accumulate
        try:
            # Heuristic: if any integer array indexing present, prefer add.at
            is_tuple = isinstance(index, tuple)
            ints_in_idx = (isinstance(index, np.ndarray) and np.issubdtype(index.dtype, np.integer)) or \
                          (is_tuple and any(isinstance(i, np.ndarray) and np.issubdtype(i.dtype, np.integer) for i in index))

            if ints_in_idx:
                np.add.at(g_in, index, grad_out)
            else:
                g_in[index] += grad_out
        except Exception:
            # Fallback to safe accumulation
            np.add.at(g_in, index, grad_out)

        return g_in, None


class MaskedFill(Function):
    @staticmethod
    def forward(ctx, x, mask, value):
        x_arr = np.asarray(x)

        # accept np.bool array or Tensor mask
        mask_arr = mask.data if hasattr(mask, "data") else mask
        m = np.asarray(mask_arr, dtype=bool)

        # broadcast mask to input shape
        try:
            m = np.broadcast_to(m, x_arr.shape)
        except ValueError:
            raise ValueError(f"mask shape {m.shape} not broadcastable to {x_arr.shape}")

        ctx.meta["mask"] = m
        val = np.array(value, dtype=x_arr.dtype)  # keep dtype (e.g., float32)

        # PyTorch semantics: fill where mask == True
        return np.where(m, val, x_arr)

    def backward(self, grad_out):
        m = self.ctx.meta["mask"]
        # no grad through filled positions
        gx = np.where(m, 0, grad_out)
        return gx, None, None  # grads for (x, mask, value)
