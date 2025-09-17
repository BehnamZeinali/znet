import numpy as np
from znet.autograd.tensor import Tensor
from znet.nn import Embedding

B,T, V, D = 2, 4, 100, 16
emb = Embedding(V, D)
idx = Tensor(np.random.randint(0, V, size=(B,T), dtype=np.int64), requires_grad=False)
x = emb(idx)           # (B,T,D)
loss = x.sum()
loss.backward()
assert emb.weight.grad.shape == (V, D)
print("test8 passed")
