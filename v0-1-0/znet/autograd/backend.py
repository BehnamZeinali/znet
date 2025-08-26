# znet/backend.py
# Unified array backend for znet (MLX-first)

from typing import Any, Iterable, Optional

try:
    import mlx.core as mx
except Exception as e:
    raise RuntimeError(
        "MLX is required as the znet backend. "
        "Install with: pip install mlx"
    ) from e

# ---- Public "xp" surface (like numpy) ----
xp = mx

# Dtypes (aliases so your code doesn't import numpy dtypes)
float32 = mx.float32
float64 = mx.float64
int32   = mx.int32
int64   = mx.int64
bool_   = mx.bool_

def array(data: Any, dtype=None):
    return mx.array(data, dtype=dtype) if dtype is not None else mx.array(data)

def asarray(data: Any, dtype=None):
    return array(data, dtype=dtype)

def ones_like(a):
    return mx.ones_like(a)

def zeros_like(a):
    return mx.zeros_like(a)

def sum(a, axis=None, keepdims=False):
    return mx.sum(a, axis=axis, keepdims=keepdims)

def reshape(a, newshape):
    return mx.reshape(a, newshape)

def transpose(a, axes=None):
    return mx.transpose(a, axes=axes)

def flatten(a):
    # (-1,) is standard flatten; preserve row-major semantics
    return mx.reshape(a, (-1,))

def where(cond, x, y):
    return mx.where(cond, x, y)

def matmul(a, b):
    return a @ b  # MLX overloads @

def to_numpy(a):
    # Convert MLX array to NumPy for logging/debug if ever needed
    import numpy as _np
    return _np.asarray(a)
