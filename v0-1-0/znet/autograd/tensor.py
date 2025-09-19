# znet/autograd/tensor.py
import numpy as np
from znet._grad_mode import is_grad_enabled
from .engine import engine_backward  # import the new engine

class Tensor:
    def __init__(self, data, requires_grad=False, dtype=np.float32):
        self.data = np.array(data, dtype=dtype)
        self.requires_grad = bool(requires_grad) and is_grad_enabled()
        self.grad = None
        self.grad_fn = None  # Function node that produced this Tensor (None => leaf)

        # TEMP: compatibility shim if old code references _grad_fn
        self._grad_fn = None

    # --- basic introspection ---
    @property
    def shape(self): return self.data.shape
    @property
    def dtype(self): return self.data.dtype
    def size(self): return self.data.shape
    def item(self):
        if self.data.size != 1:
            raise ValueError("Can only convert a Tensor with one element to a Python scalar")
        return self.data.item()

    # --- mutation helpers ---
    def zero_grad(self): self.grad = None
    def requires_grad_(self, mode: bool = True):
        self.requires_grad = bool(mode) and is_grad_enabled()
        return self
    def detach(self):
        return Tensor(self.data, requires_grad=False)

    # --- backward ---
    def backward(self, grad=None, retain_graph=False):
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be specified for non-scalar Tensor")
            grad = np.ones_like(self.data)
        engine_backward(self, grad, retain_graph=retain_graph)

    def numel(self) -> int:
        return int(self.data.size)
    def view(self, *shape):
        # accept view(B*T, C), view((B*T, C)), or view(-1, C)
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        from .ops_view import View
        return View.apply(self, shape)
    
    
    # ---- Views / reductions using Function.apply ----
    def reshape(self, new_shape):
        from .ops_view import Reshape
        return Reshape.apply(self, tuple(new_shape))
    
    def swapaxes(self, axis1, axis2):
        from .ops_view import SwapAxes
        return SwapAxes.apply(self, int(axis1), int(axis2))

    # inside class Tensor (optional convenience)
    @staticmethod
    def cat(tensors, dim=0):
        from .autograd.ops_view import Cat as _cat
        return _cat.apply(*tensors, int(dim))
    
    @property
    def T(self ):
        from .ops_view import Transpose
        return Transpose.apply(self)
    
    def transpose(self, dim0=-2, dim1=-1):
        from .ops_view import Transpose
        return Transpose.apply(self, dim0, dim1)

    def masked_fill(self, mask, value):
        from .ops_view import MaskedFill
        return MaskedFill.apply(self, mask, value)
    
    def flatten(self):
        from .ops_view import Flatten
        return Flatten.apply(self)

    def sum(self, axis=None, keepdims=False):
        from .ops_view import Sum
        return Sum.apply(self, axis, keepdims)
    def tolist(self):
        return self.data.tolist()
    def __str__(self):
        return str(self.data)
    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
    def __len__(self):
        if self.data.ndim == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.data.shape[0]
    
    def __getitem__(self, index):
    # normalize possible Tensor / numpy-scalar indices
        def _norm(i):
            if isinstance(i, Tensor):           # tensor index -> its data
                i = i.data
            if isinstance(i, np.generic):       # numpy scalar -> python int
                i = int(i)
            return i
        if isinstance(index, tuple):
            index = tuple(_norm(i) for i in index)
        else:
            index = _norm(index)

        from .ops_view import Index
        return Index.apply(self, index)
    

        # ----- elementwise -----
    def __add__(self, other):
        from .ops_math import add
        return add(self, other)
    def __radd__(self, other):
        from .ops_math import add
        return add(other, self)

    def __sub__(self, other):
        from .ops_math import sub
        return sub(self, other)
    def __rsub__(self, other):
        from .ops_math import sub
        return sub(other, self)

    def __mul__(self, other):
        from .ops_math import mul
        return mul(self, other)
    def __rmul__(self, other):
        from .ops_math import mul
        return mul(other, self)

    def __truediv__(self, other):
        from .ops_math import div
        return div(self, other)
    def __rtruediv__(self, other):
        from .ops_math import div
        return div(other, self)

    # ----- matmul (last two dims) -----
    def matmul(self, other):
        from .ops_matmul import matmul
        return matmul(self, other)

    # Python @ operator
    def __matmul__(self, other):
        return self.matmul(other)

