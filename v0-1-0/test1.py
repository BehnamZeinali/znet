import numpy as np
from znet.autograd.tensor import Tensor

# Test A: scalar leaf backward writes ones
x = Tensor(3.14, requires_grad=True)
x.backward()
assert x.grad.shape == ()
assert np.allclose(x.grad, 1.0)
print("Test A passed.")

# Test B: vector leaf backward with explicit grad
v = Tensor([1.0, 2.0, 3.0], requires_grad=True)
v.backward(np.array([10.0, 20.0, 30.0], dtype=v.dtype))
assert np.allclose(v.grad, [10.0, 20.0, 30.0])
print("Test B passed.")
