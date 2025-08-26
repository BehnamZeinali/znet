import mlx.core as mx
from znet.autograd.tensor import Tensor

def test_add_broadcast_backward():
    x = Tensor(mx.random.normal(shape=(2, 3, 4), dtype=mx.float32), requires_grad=True)
    b = Tensor(mx.random.normal(shape=(1, 1, 4), dtype=mx.float32), requires_grad=True)
    y = x + b
    s = y.sum()
    s.backward()
    # x grad: ones
    assert bool(mx.allclose(x.grad, mx.ones_like(x.data)))
    # b grad: summed over broadcasted axes (2,3)
    assert bool(mx.allclose(b.grad, mx.ones_like(b.data) * (2 * 3)))
    print("pass: add_broadcast_backward")

def test_mul_backward():
    x = Tensor(mx.random.normal(shape=(5,), dtype=mx.float32), requires_grad=True)
    y = Tensor(mx.random.normal(shape=(5,), dtype=mx.float32), requires_grad=True)
    z = (x * y).sum()
    z.backward()
    assert bool(mx.allclose(x.grad, y.data))
    assert bool(mx.allclose(y.grad, x.data))
    print("pass: mul_backward")

def test_div_backward():
    x = Tensor(mx.random.normal(shape=(4,), dtype=mx.float32), requires_grad=True)
    y = Tensor(mx.random.normal(shape=(4,), dtype=mx.float32) + 2.0, requires_grad=True)
    z = (x / y).sum()
    z.backward()
    assert bool(mx.allclose(x.grad, 1.0 / y.data))
    assert bool(mx.allclose(y.grad, -x.data / (y.data ** 2)))
    print("pass: div_backward")

def test_matmul_2d():
    a = Tensor(mx.random.normal(shape=(3, 7), dtype=mx.float32), requires_grad=True)
    b = Tensor(mx.random.normal(shape=(7, 5), dtype=mx.float32), requires_grad=True)
    y = a @ b          # (3,5)
    s = y.sum()
    s.backward()
    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape
    print("pass: matmul_2d")

def test_matmul_batched_nd():
    # a: (B, M, K), b: (1, K, N) -> broadcast over batch
    B, M, K, N = 4, 2, 3, 5
    a = Tensor(mx.random.normal(shape=(B, M, K), dtype=mx.float32), requires_grad=True)
    b = Tensor(mx.random.normal(shape=(1, K, N), dtype=mx.float32), requires_grad=True)
    y = a @ b   # (B,M,N)
    g = mx.random.normal(shape=(B, M, N), dtype=mx.float32)
    y.backward(g)

    # Numerical sanity: shapes right and grads finite
    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape
    assert bool(mx.all(mx.isfinite(a.grad)))
    assert bool(mx.all(mx.isfinite(b.grad)))
    print("pass: matmul_batched_nd")

def test_matmul_mismatched_inner_raises():
    a = Tensor(mx.zeros((2, 3, 4), dtype=mx.float32), requires_grad=True)
    b = Tensor(mx.zeros((2, 5, 6), dtype=mx.float32), requires_grad=True)
    try:
        _ = a @ b
        raise AssertionError("Expected ValueError due to inner-dim mismatch, but none was raised")
    except ValueError:
        pass
    print("pass: matmul_mismatched_inner_raises")

# Run tests
test_add_broadcast_backward()
test_mul_backward()
test_div_backward()
test_matmul_2d()
test_matmul_batched_nd()
test_matmul_mismatched_inner_raises()
