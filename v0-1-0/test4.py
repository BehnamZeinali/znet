import mlx.core as mx
from znet.autograd.tensor import Tensor
from znet.nn.linear import Linear

def test_matmul_1d_1d_scalar():
    D = 6
    a = Tensor(mx.random.normal(shape=(D,), dtype=mx.float32), requires_grad=True)
    b = Tensor(mx.random.normal(shape=(D,), dtype=mx.float32), requires_grad=True)
    y = a @ b          # scalar
    y.backward()       # ∂/∂a = b, ∂/∂b = a
    assert y.shape == ()
    assert bool(mx.allclose(a.grad, b.data, atol=1e-5))
    assert bool(mx.allclose(b.grad, a.data, atol=1e-5))
    print("pass: matmul_1d_1d_scalar")

def test_matmul_1d_2d():
    D, N = 5, 3
    a = Tensor(mx.random.normal(shape=(D,), dtype=mx.float32), requires_grad=True)
    Bm = Tensor(mx.random.normal(shape=(D, N), dtype=mx.float32), requires_grad=True)
    y = a @ Bm              # (N,)
    s = y.sum()
    s.backward()
    # ds/da_i = sum_j B[i,j]
    assert bool(mx.allclose(a.grad, mx.sum(Bm.data, axis=1), atol=1e-5))
    # ds/dB[i,j] = a_i
    assert bool(mx.allclose(Bm.grad, mx.broadcast_to(a.data[:, None], (D, N)), atol=1e-5))
    print("pass: matmul_1d_2d")

def test_matmul_2d_1d():
    M, D = 4, 7
    A = Tensor(mx.random.normal(shape=(M, D), dtype=mx.float32), requires_grad=True)
    b = Tensor(mx.random.normal(shape=(D,), dtype=mx.float32), requires_grad=True)
    y = A @ b              # (M,)
    s = y.sum()
    s.backward()
    # ds/dA[i,k] = b_k
    assert bool(mx.allclose(A.grad, mx.broadcast_to(b.data[None, :], (M, D)), atol=1e-5))
    # ds/db_k = sum_i A[i,k]
    assert bool(mx.allclose(b.grad, mx.sum(A.data, axis=0), atol=1e-5))
    print("pass: matmul_2d_1d")

def test_matmul_broadcast_nd():
    B, M, K, N = 3, 2, 4, 5
    A = Tensor(mx.random.normal(shape=(B, M, K), dtype=mx.float32), requires_grad=True)
    Bm = Tensor(mx.random.normal(shape=(1, K, N), dtype=mx.float32), requires_grad=True)
    G = mx.random.normal(shape=(B, M, N), dtype=mx.float32)
    Y = A @ Bm
    Y.backward(G)
    assert A.grad.shape == (B, M, K)
    assert Bm.grad.shape == (1, K, N)
    print("pass: matmul_broadcast_nd")

def test_linear_forward_backward_shapes():
    B, Fin, Fout = 8, 6, 4
    layer = Linear(Fin, Fout, bias=True)
    x = Tensor(mx.random.normal(shape=(B, Fin), dtype=mx.float32), requires_grad=True)
    y = layer(x)                 # (B, Fout)
    g = mx.random.normal(shape=(B, Fout), dtype=mx.float32)
    y.backward(g)
    # Grad shapes
    assert x.grad.shape == (B, Fin)
    # layer.weight has shape (Fout, Fin); grad matches parameter shape
    assert layer.weight.grad.shape == (Fout, Fin)
    assert layer.bias.grad.shape == (Fout,)
    print("pass: linear_forward_backward_shapes")

# Run tests
test_matmul_1d_1d_scalar()
test_matmul_1d_2d()
test_matmul_2d_1d()
test_matmul_broadcast_nd()
test_linear_forward_backward_shapes()
