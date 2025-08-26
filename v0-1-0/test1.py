import mlx.core as mx
from znet.autograd.tensor import Tensor

# Test A: scalar leaf backward writes ones
x = Tensor(3.14, requires_grad=True)
x.backward()
assert x.grad.shape == ()
assert bool(mx.allclose(x.grad, mx.array(1.0, dtype=x.grad.dtype)))
print("Test A passed.")

# Test B: vector leaf backward with explicit grad
v = Tensor([1.0, 2.0, 3.0], requires_grad=True)
v.backward(mx.array([10.0, 20.0, 30.0], dtype=v.dtype))
assert bool(mx.allclose(v.grad, mx.array([10.0, 20.0, 30.0], dtype=v.dtype)))
print("Test B passed.")
