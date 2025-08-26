import mlx.core as mx
from znet.autograd.tensor import Tensor

def test_reshape_backward():
    x = Tensor(mx.array([0, 1, 2, 3, 4, 5], dtype=mx.float32).reshape((2, 3)), requires_grad=True)
    y = x.reshape((3, 2))       # view
    s = y.sum()                 # scalar
    s.backward()                # ∂s/∂x = ones
    assert x.grad.shape == x.shape
    assert bool(mx.allclose(x.grad, mx.ones_like(x.data)))
    print("pass: reshape_backward")

def test_T_backward_2d():
    x = Tensor(mx.array([[1., 2.], [3., 4.]], dtype=mx.float32), requires_grad=True)
    y = x.T
    s = y.sum()
    s.backward()
    assert bool(mx.allclose(x.grad, mx.ones_like(x.data)))
    print("pass: T_backward_2d")

def test_flatten_backward():
    # random input
    x = Tensor(mx.random.normal(shape=(2, 3, 4), dtype=mx.float32), requires_grad=True)
    y = x.flatten()
    s = y.sum()
    s.backward()
    assert bool(mx.allclose(x.grad, mx.ones_like(x.data)))
    print("pass: flatten_backward")

def test_sum_axis_keepdims_false():
    x = Tensor(mx.ones((2, 3), dtype=mx.float32), requires_grad=True)
    y = x.sum(axis=1, keepdims=False)   # shape (2,)
    s = y.sum()                         # scalar
    s.backward()
    # every input contributed once to the final sum → grad ones
    assert bool(mx.allclose(x.grad, mx.ones_like(x.data)))
    print("pass: sum_axis_keepdims_false")

def test_sum_axis_tuple_keepdims_true():
    x = Tensor(mx.ones((2, 3, 4), dtype=mx.float32), requires_grad=True)
    y = x.sum(axis=(1, 2), keepdims=True)  # shape (2,1,1)
    # Put explicit upstream grad to verify broadcasting back
    g = mx.array([[[2.0]], [[3.0]]], dtype=mx.float32)  # shape (2,1,1)
    y.backward(g)
    # Each example i receives its scalar grad replicated over 3*4 positions
    assert bool(mx.allclose(x.grad[0], mx.full((3, 4), 2.0, dtype=x.data.dtype)))
    assert bool(mx.allclose(x.grad[1], mx.full((3, 4), 3.0, dtype=x.data.dtype)))
    print("pass: sum_axis_tuple_keepdims_true")

# Run tests
test_reshape_backward()
test_T_backward_2d()
test_flatten_backward()
test_sum_axis_keepdims_false()
test_sum_axis_tuple_keepdims_true()
