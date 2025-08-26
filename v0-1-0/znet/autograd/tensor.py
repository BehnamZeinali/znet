# znet/autograd/tensor.py
from znet._grad_mode import is_grad_enabled
from .engine import engine_backward
from znet.autograd.backend import xp, array, ones_like, float32  # MLX-backed

class Tensor:
    def __init__(self, data, requires_grad=False, dtype=float32):
        # data -> MLX array
        self.data = array(data, dtype=dtype)
        self.requires_grad = bool(requires_grad) and is_grad_enabled()
        self.grad = None
        self.grad_fn = None   # Function node that produced this Tensor (None => leaf)
        self._grad_fn = None  # TEMP shim if old code referenced this

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
        # share underlying storage (MLX arrays are immutable-by-value semantics like JAX/NumPy)
        return Tensor(self.data, requires_grad=False)

    # --- backward ---
    def backward(self, grad=None, retain_graph=False):
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be specified for non-scalar Tensor")
            grad = ones_like(self.data)
        engine_backward(self, grad, retain_graph=retain_graph)

    # ---- Views / reductions using Function.apply ----
    def reshape(self, new_shape):
        from .ops_view import Reshape
        return Reshape.apply(self, tuple(new_shape))

    @property
    def T(self):
        from .ops_view import Transpose
        return Transpose.apply(self)

    def flatten(self):
        from .ops_view import Flatten
        return Flatten.apply(self)

    def sum(self, axis=None, keepdims=False):
        from .ops_view import Sum
        return Sum.apply(self, axis, keepdims)

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

    def __matmul__(self, other):
        return self.matmul(other)
