import numpy as np
from .module import Module
from ..autograd.tensor import Tensor
from ..autograd.ops_embedding import embedding 

class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, dtype=np.float32):
        super().__init__()
        # GPT-2 uses small init ~N(0,0.02)
        w = (np.random.randn(num_embeddings, embedding_dim).astype(dtype) * 0.02)
        self.add_parameter("weight", Tensor(w, requires_grad=True, dtype=dtype))

    def forward(self, indices: Tensor) -> Tensor:
        # indices should be integer (int64) Tensor, requires_grad=False
        return embedding(self.weight, indices)
