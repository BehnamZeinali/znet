import mlx.core as mx
from znet.autograd.tensor import Tensor
from znet.nn.linear import Linear

B, Fin, Fout = 4, 6, 3
layer = Linear(Fin, Fout, bias=True)

x = Tensor(mx.random.normal(shape=(B, Fin), dtype=mx.float32), requires_grad=True)
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

print("pass: linear_forward_backward_integration")
