import numpy as np
import znet.nn as nn
from znet.autograd import Tensor

# --- make an embedding table W (V=5, D=3) with fixed rows ---
W = np.array([
    [ 0.1,  0.0, -0.2],  # token 0
    [ 0.3,  0.5,  0.7],  # token 1
    [-0.4,  0.2,  0.0],  # token 2
    [ 1.0, -1.0,  0.5],  # token 3
    [ 0.6,  0.6,  0.6],  # token 4
], dtype=np.float32)

emb_tok = nn.Embedding(num_embeddings=5, embedding_dim=3)
emb_tok.weight.data[...] = W  # overwrite init for determinism
# emb_tok.weight.grad[...] = 0  # clear grads if any lingering

# indices (B=1, T=4)
idx = Tensor(np.array([[3, 1, 0, 4]], dtype=np.int64), requires_grad=False)

# forward
Y = emb_tok(idx)                 # shape (1,4,3)
print("Token Y:\n", Y.data)

# loss = sum(Y)  -> upstream grad is all ones

# better way (if your Tensor has .sum op):
loss = Y.sum()
loss.backward()

print("Token dW:\n", emb_tok.weight.grad)
