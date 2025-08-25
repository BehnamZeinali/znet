# znet/autograd/tensor.py
from __future__ import annotations
import numpy as np
import torch as th
from znet._grad_mode import is_grad_enabled
from .engine import engine_backward  # your engine stays in charge of autograd

# ---------- device & dtype helpers ----------
def _auto_device() -> th.device:
    if th.cuda.is_available():
        return th.device("cuda")
    # MPS for Apple Silicon (macOS 12.3+ with a supported GPU)
    if th.backends.mps.is_available() and th.backends.mps.is_built():
        return th.device("mps")
    return th.device("cpu")

_NUMPY2TORCH = {
    np.float16: th.float16,
    np.float32: th.float32,
    np.float64: th.float64,
    np.int8:    th.int8,
    np.int16:   th.int16,
    np.int32:   th.int32,
    np.int64:   th.int64,
    np.uint8:   th.uint8,
    bool:       th.bool,
    np.bool_:   th.bool,
}
_STR2TORCH = {
    "float16": th.float16, "fp16": th.float16, "half": th.float16,
    "float32": th.float32, "fp32": th.float32, "float": th.float32,
    "float64": th.float64, "fp64": th.float64, "double": th.float64,
    "int8": th.int8, "int16": th.int16, "int32": th.int32, "int64": th.int64,
    "uint8": th.uint8, "bool": th.bool,
}

def _resolve_dtype(dtype) -> th.dtype:
    if dtype is None:
        return th.float32
    if isinstance(dtype, th.dtype):
        return dtype
    if isinstance(dtype, str):
        try: return _STR2TORCH[dtype.lower()]
        except KeyError:
            raise TypeError(f"Unsupported dtype string: {dtype}")
    # numpy dtype or python type
    try:
        return _NUMPY2TORCH[np.dtype(dtype).type]
    except Exception as e:
        raise TypeError(f"Unsupported dtype: {dtype}") from e


class Tensor:
    """
    A lightweight front-end tensor that stores data in torch.Tensor
    but delegates autograd to znet's custom engine.
    """
    def __init__(self, data, requires_grad: bool = False,
                 dtype=np.float32, device: str | th.device | None = None):
        tdtype = _resolve_dtype(dtype)
        dev = th.device(device) if device is not None else _auto_device()

        # Convert input to a torch tensor living on the chosen device/dtype.
        # Ensure it's detached so Torch's autograd never tracks it.
        if isinstance(data, th.Tensor):
            base = data.to(device=dev, dtype=tdtype)
            self.data = base.detach().clone() if base.requires_grad else base.detach()
        else:
            # as_tensor avoids unnecessary copies when possible
            self.data = th.as_tensor(data, dtype=tdtype, device=dev)

        # Keep znet's autograd semantics
        self.requires_grad = bool(requires_grad) and is_grad_enabled()
        self.grad = None               # expect torch.Tensor or None
        self.grad_fn = None            # your engine's function-node
        self._grad_fn = None           # legacy alias (kept for compatibility)

    # --- basic introspection ---
    @property
    def shape(self): return tuple(self.data.shape)
    @property
    def dtype(self): return self.data.dtype
    @property
    def device(self): return self.data.device
    def size(self): return tuple(self.data.shape)
    def item(self):
        if self.data.numel() != 1:
            raise ValueError("Can only convert a Tensor with one element to a Python scalar")
        return self.data.item()

    # --- mutation helpers ---
    def zero_grad(self): self.grad = None
    def requires_grad_(self, mode: bool = True):
        self.requires_grad = bool(mode) and is_grad_enabled()
        return self

    def detach(self):
        # same storage, no autograd
        return Tensor(self.data, requires_grad=False)

    # --- device/dtype transfer ---
    def to(self, device: str | th.device | None = None,
           dtype: th.dtype | str | None = None, copy: bool = False) -> "Tensor":
        dev = th.device(device) if device is not None else self.device
        td = _resolve_dtype(dtype) if dtype is not None else self.data.dtype
        out = self.data.to(device=dev, dtype=td)
        if copy:
            out = out.clone()
        return Tensor(out, device= device, requires_grad=self.requires_grad)

    def to_(self, device: str | th.device | None = None,
            dtype: th.dtype | str | None = None) -> "Tensor":
        """In-place variant."""
        dev = th.device(device) if device is not None else self.device
        td = _resolve_dtype(dtype) if dtype is not None else self.data.dtype
        self.data = self.data.to(device=dev, dtype=td)
        return self

    # handy shortcuts
    def cpu(self):  return self.to("cpu")
    def cuda(self): return self.to("cuda") if th.cuda.is_available() else self.to("cpu")
    def mps(self):  return self.to("mps")  if (th.backends.mps.is_available() and th.backends.mps.is_built()) else self.to("cpu")
    def numpy(self): return self.data.detach().cpu().numpy()

    # --- backward (znet engine) ---
    def backward(self, grad=None, retain_graph=False):
        if grad is None:
            if self.data.numel() != 1:
                raise RuntimeError("grad must be specified for non-scalar Tensor")
            grad = th.ones_like(self.data)
        elif isinstance(grad, np.ndarray):
            # allow legacy callers to pass numpy grads
            grad = th.as_tensor(grad, dtype=self.data.dtype, device=self.device)
        elif not isinstance(grad, th.Tensor):
            grad = th.as_tensor(grad, dtype=self.data.dtype, device=self.device)

        # Delegate to your engine (which should expect torch tensors now)
        engine_backward(self, grad, retain_graph=retain_graph)

    # ---- Views / reductions using Function.apply (unchanged public API) ----
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
