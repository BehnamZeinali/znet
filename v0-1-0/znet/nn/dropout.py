# znet/nn/dropout.py
import numpy as np
from .module import Module
from ..autograd.tensor import Tensor
from typing import Optional
class Dropout(Module):
    def __init__(self, p: float = 0.5, seed: Optional[int] = None):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError("dropout p must be in [0, 1).")
        self.p = float(p)
        # own RNG so module is reproducible if desired
        self._rng = np.random.default_rng(seed)

    def forward(self, x: Tensor) -> Tensor:
        if (not self.training) or self.p == 0.0:
            return x
        # exact-shape mask, inverted scaling
        keep_prob = 1.0 - self.p
        mask = (self._rng.random(x.shape) < keep_prob).astype(x.data.dtype) / keep_prob
        mask = Tensor(mask, requires_grad=False)
        return x * mask   # your Mul backward will pass grad * mask
