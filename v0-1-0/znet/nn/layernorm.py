import numpy as np
from .module import Module
from ..autograd.tensor import Tensor
from ..autograd.ops_layernorm import layer_norm as layer_norm_op

class LayerNorm(Module):
    """
    LayerNorm over the last `normalized_shape` dims (PyTorch-compatible behavior).

    Args:
      normalized_shape: int or tuple[int, ...] – size of the last dims to normalize
      eps: float – numerical stability term added to variance
      elementwise_affine: if True, learn gamma (weight) and beta (bias)
    """
    def __init__(self, normalized_shape, eps: float = 1e-5,
                 elementwise_affine: bool = True, dtype=np.float32):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(int(s) for s in normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)

        if self.elementwise_affine:
            w = np.ones(self.normalized_shape, dtype=dtype)
            b = np.zeros(self.normalized_shape, dtype=dtype)
            self.add_parameter("weight", Tensor(w, requires_grad=True, dtype=dtype))
            self.add_parameter("bias",   Tensor(b, requires_grad=True, dtype=dtype))
        else:
            self.weight = None
            self.bias   = None

    def forward(self, x: Tensor) -> Tensor:
        # prepare gamma/beta tensors; if affine=False, use non-trainable ones/zeros
        if self.elementwise_affine:
            gamma_t = self.weight
            beta_t  = self.bias
        else:
            gamma_t = Tensor(np.ones(self.normalized_shape, dtype=x.data.dtype), requires_grad=False)
            beta_t  = Tensor(np.zeros(self.normalized_shape, dtype=x.data.dtype), requires_grad=False)

        # normalize over the last k dims, where k = len(normalized_shape)
        k = len(self.normalized_shape)
        axes = tuple(range(x.data.ndim - k, x.data.ndim))
        return layer_norm_op(x, gamma_t, beta_t, eps=self.eps, axes=axes)
