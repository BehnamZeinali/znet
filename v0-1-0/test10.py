import numpy as np
import znet.nn as nn
from znet.autograd import Tensor

# Positional table P (max_len=8, D=3); fill first 4 rows
P = np.zeros((8, 3), dtype=np.float32)
P[0] = [ 0.01,  0.02,  0.03]
P[1] = [ 0.10,  0.10,  0.10]
P[2] = [ 0.20,  0.00, -0.20]
P[3] = [-0.10,  0.20,  0.00]

pos_emb = nn.Embedding(num_embeddings=8, embedding_dim=3)
pos_emb.weight.data[...] = P

# Position indices (B=1, T=4): [0,1,2,3]
pos_ids = Tensor(np.array([[0, 1, 2, 3]], dtype=np.int64), requires_grad=False)

# Forward
Ypos = pos_emb(pos_ids)  # (1, 4, 3)
print("Pos Y:\n", Ypos.data)

# Loss
loss = Ypos.sum()
loss.backward()

print("Pos dP (first 5 rows):\n", pos_emb.weight.grad[:5])
