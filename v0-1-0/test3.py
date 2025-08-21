import numpy as np
from znet.autograd.tensor import Tensor

def test_add_broadcast_backward():
    x = Tensor(np.random.randn(2,3,4).astype(np.float32), requires_grad=True)
    b = Tensor(np.random.randn(1,1,4).astype(np.float32), requires_grad=True)
    y = x + b
    s = y.sum()
    s.backward()
    # x grad: ones
    assert np.allclose(x.grad, np.ones_like(x.data))
    # b grad: summed over broadcasted axes (2,3)
    assert np.allclose(b.grad, np.ones_like(b.data) * (2*3))

def test_mul_backward():
    x = Tensor(np.random.randn(5,).astype(np.float32), requires_grad=True)
    y = Tensor(np.random.randn(5,).astype(np.float32), requires_grad=True)
    z = (x * y).sum()
    z.backward()
    assert np.allclose(x.grad, y.data)
    assert np.allclose(y.grad, x.data)

def test_div_backward():
    x = Tensor(np.random.randn(4,).astype(np.float32), requires_grad=True)
    y = Tensor((np.random.randn(4,).astype(np.float32) + 2.0), requires_grad=True)
    z = (x / y).sum()
    z.backward()
    assert np.allclose(x.grad, 1.0 / y.data)
    assert np.allclose(y.grad, -x.data / (y.data**2))

def test_matmul_2d():
    a = Tensor(np.random.randn(3,7).astype(np.float32), requires_grad=True)
    b = Tensor(np.random.randn(7,5).astype(np.float32), requires_grad=True)
    y = a @ b          # (3,5)
    s = y.sum()
    s.backward()
    # ∂sum/∂a = ones @ b^T = row-sum of b^T = sum over n-dim
    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape

def test_matmul_batched_nd():
    # a: (B, M, K), b: (1, K, N) -> broadcast over batch
    B,M,K,N = 4, 2, 3, 5
    a = Tensor(np.random.randn(B,M,K).astype(np.float32), requires_grad=True)
    b = Tensor(np.random.randn(1,K,N).astype(np.float32), requires_grad=True)
    y = a @ b   # (B,M,N)
    g = np.random.randn(B,M,N).astype(np.float32)
    y.backward(g)

    # Numerical sanity: shapes right and grads finite
    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape
    assert np.isfinite(a.grad).all() and np.isfinite(b.grad).all()

def test_matmul_mismatched_inner_raises():
    import pytest
    a = Tensor(np.zeros((2,3,4), dtype=np.float32), requires_grad=True)
    b = Tensor(np.zeros((2,5,6), dtype=np.float32), requires_grad=True)
    with pytest.raises(ValueError):
        # Let NumPy raise by attempting @ (inner dims mismatch)
        _ = a @ b
test_matmul_mismatched_inner_raises()
test_div_backward()