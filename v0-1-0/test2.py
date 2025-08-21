import numpy as np
from znet.autograd.tensor import Tensor

def test_reshape_backward():
    x = Tensor(np.arange(6, dtype=np.float32).reshape(2,3), requires_grad=True)
    y = x.reshape((3,2))       # view
    s = y.sum()                # scalar
    s.backward()               # ∂s/∂x = ones
    assert x.grad.shape == x.shape
    assert np.allclose(x.grad, np.ones_like(x.data))
    print("True")

def test_T_backward_2d():
    x = Tensor(np.array([[1.,2.],[3.,4.]], dtype=np.float32), requires_grad=True)
    y = x.T
    s = y.sum()
    s.backward()
    assert np.allclose(x.grad, np.ones_like(x.data))

def test_flatten_backward():
    x = Tensor(np.random.randn(2,3,4).astype(np.float32), requires_grad=True)
    y = x.flatten()
    s = y.sum()
    s.backward()
    assert np.allclose(x.grad, np.ones_like(x.data))

def test_sum_axis_keepdims_false():
    x = Tensor(np.ones((2,3), dtype=np.float32), requires_grad=True)
    y = x.sum(axis=1, keepdims=False)   # shape (2,)
    s = y.sum()                         # scalar
    s.backward()
    # every input contributed once to the final sum → grad ones
    assert np.allclose(x.grad, np.ones_like(x.data))

def test_sum_axis_tuple_keepdims_true():
    x = Tensor(np.ones((2,3,4), dtype=np.float32), requires_grad=True)
    y = x.sum(axis=(1,2), keepdims=True)  # shape (2,1,1)
    # Put explicit upstream grad to verify broadcasting back
    g = np.array([[[2.0]], [[3.0]]], dtype=np.float32)  # shape (2,1,1)
    y.backward(g)
    # Each example i receives its scalar grad replicated over 3*4 positions
    assert np.allclose(x.grad[0], np.full((3,4), 2.0))
    assert np.allclose(x.grad[1], np.full((3,4), 3.0))

test_reshape_backward()