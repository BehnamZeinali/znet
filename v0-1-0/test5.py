import numpy as np
from znet.autograd.tensor import Tensor
from znet.nn.linear import Linear

B, Fin, Fout = 4, 6, 3
layer = Linear(Fin, Fout, bias=True)

x = Tensor(np.random.randn(B, Fin).astype(np.float32), requires_grad=True)
y = layer(x)                 # (B, Fout)
print(y.data)
loss = y.sum()
loss.backward()

# grads exist and shapes correct
assert x.grad.shape == (B, Fin)
for p in layer.parameters():
    assert p.grad is not None
layer.zero_grad()
for p in layer.parameters():
    assert p.grad is None
