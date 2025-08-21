import numpy as np
from znet.autograd.tensor import Tensor
from znet.nn.linear import Linear

def test_matmul_1d_1d_scalar():
    D = 6
    a = Tensor(np.random.randn(D).astype(np.float32), requires_grad=True)
    b = Tensor(np.random.randn(D).astype(np.float32), requires_grad=True)
    y = a @ b          # scalar
    y.backward()       # ∂/∂a = b, ∂/∂b = a
    assert y.shape == ()
    assert np.allclose(a.grad, b.data, atol=1e-5)
    assert np.allclose(b.grad, a.data, atol=1e-5)

def test_matmul_1d_2d():
    D, N = 5, 3
    a = Tensor(np.random.randn(D).astype(np.float32), requires_grad=True)
    B = Tensor(np.random.randn(D, N).astype(np.float32), requires_grad=True)
    y = a @ B              # (N,)
    s = y.sum()
    s.backward()
    # ds/da_i = sum_j B[i,j]
    assert np.allclose(a.grad, B.data.sum(axis=1), atol=1e-5)
    # ds/dB[i,j] = a_i
    assert np.allclose(B.grad, np.repeat(a.data[:,None], N, axis=1), atol=1e-5)

def test_matmul_2d_1d():
    M, D = 4, 7
    A = Tensor(np.random.randn(M, D).astype(np.float32), requires_grad=True)
    b = Tensor(np.random.randn(D).astype(np.float32), requires_grad=True)
    y = A @ b              # (M,)
    s = y.sum()
    s.backward()
    # ds/dA[i,k] = b_k
    assert np.allclose(A.grad, np.repeat(b.data[None,:], M, axis=0), atol=1e-5)
    # ds/db_k = sum_i A[i,k]
    assert np.allclose(b.grad, A.data.sum(axis=0), atol=1e-5)

def test_matmul_broadcast_nd():
    B,M,K,N = 3, 2, 4, 5
    A = Tensor(np.random.randn(B,M,K).astype(np.float32), requires_grad=True)
    Bm = Tensor(np.random.randn(1,K,N).astype(np.float32), requires_grad=True)
    G = np.random.randn(B,M,N).astype(np.float32)
    Y = A @ Bm
    Y.backward(G)
    assert A.grad.shape == (B,M,K)
    assert Bm.grad.shape == (1,K,N)
    print("OK")

def test_linear_forward_backward_shapes():
    B, Fin, Fout = 8, 6, 4
    layer = Linear(Fin, Fout, bias=True)
    x = Tensor(np.random.randn(B, Fin).astype(np.float32), requires_grad=True)
    y = layer(x)                 # (B, Fout)
    g = np.random.randn(B, Fout).astype(np.float32)
    y.backward(g)
    # Grad shapes
    assert x.grad.shape == (B, Fin)
    assert layer.weight.grad.shape == (Fin, Fout)
    assert layer.bias.grad.shape == (Fout,)
test_matmul_broadcast_nd()