import numpy as np
from .module import Module
from .linear import Linear
from .softmax import Softmax
from ..autograd.tensor import Tensor
from ..autograd.ops_softmax import softmax as softmax_op  # for functional use if you prefer

class CausalSelfAttention(Module):
    """
    x: (B, T, D)
    returns: (B, T, D)
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # projections
        self.q = Linear(d_model, d_model)   # project to (B,T,D) then view as (B,T,H,Hd)
        self.k = Linear(d_model, d_model)
        self.v = Linear(d_model, d_model)
        self.out = Linear(d_model, d_model)

        self.softmax = Softmax(axis=-1)     # over key time dimension

    def forward(self, x: Tensor):
        B, T, D = x.shape

        # project
        q = self.q(x)   # (B,T,D)
        k = self.k(x)   # (B,T,D)
        v = self.v(x)   # (B,T,D)

        # split heads: (B,T,H,Hd) -> (B,H,T,Hd)
        H, Hd = self.n_heads, self.head_dim
        q = q.reshape((B, T, H, Hd)).swapaxes(1, 2)   # (B,H,T,Hd)
        k = k.reshape((B, T, H, Hd)).swapaxes(1, 2)   # (B,H,T,Hd)
        v = v.reshape((B, T, H, Hd)).swapaxes(1, 2)   # (B,H,T,Hd)

        # scores: (B,H,T,T) = (B,H,T,Hd) @ (B,H,Hd,T)
        k_t = k.swapaxes(-1, -2)                      # (B,H,Hd,T)
        scores = (q @ k_t) * (1.0 / np.sqrt(Hd))      # scale

        # causal mask: upper triangle (exclude future)
        # mask shape (T,T): True where masked
        causal = np.triu(np.ones((T, T), dtype=bool), k=1)
        # add -inf to masked positions; rely on broadcasting over (B,H)
        minus_inf = -1e30
        scores = scores + Tensor(causal[None, None, :, :], requires_grad=False) * minus_inf

        # softmax over last dim (keys)
        attn = self.softmax(scores)                   # (B,H,T,T)

        # weighted sum: (B,H,T,Hd) = (B,H,T,T) @ (B,H,T,Hd)
        out = attn @ v                                # (B,H,T,Hd)

        # merge heads: (B,T,H,Hd) -> (B,T,D)
        out = out.swapaxes(1, 2).reshape((B, T, D))

        # output projection
        out = self.out(out)                           # (B,T,D)
        return out
