# quick_matmul_check.py
import torch as th
from znet.autograd.tensor import Tensor
from znet.autograd.ops_matmul import matmul

# Case A: (D,) @ (D,) -> scalar
# L = x·y = sum_i x_i y_i
# => dL/dx = y
# => dL/dy = x
x = Tensor(th.arange(5.0), requires_grad=True)      # [0,1,2,3,4]
y = Tensor(th.ones(5), requires_grad=True)          # [1,1,1,1,1]
s = matmul(x, y)                                    # scalar
s.backward()
print("A) x.grad:", x.grad)
print("A) y.grad:", y.grad)
assert th.allclose(x.grad, y.data)
assert th.allclose(y.grad, x.data)

# Case B: (D,) @ (D,N) -> (N,)
# z_j = sum_i x_i B_{i,j}
# Let L = sum_j z_j = sum_{i,j} x_i B_{i,j}
# => dL/dx_i = sum_j B_{i,j}   (row-sum of B)
# => dL/dB_{i,j} = x_i
x = Tensor(th.arange(3.0), requires_grad=True)      # shape (3,)
B = Tensor(th.arange(6.0).reshape(3,2), requires_grad=True)  # shape (3,2)
z = matmul(x, B)                                    # shape (2,)
L = z.sum()
L.backward()
print("B) x.grad:", x.grad)     # expected: B.sum(dim=1)
print("B) B.grad:\n", B.grad)   # expected: outer(x, ones(N))
assert th.allclose(x.grad, B.data.sum(dim=1))
assert th.allclose(B.grad, x.data.unsqueeze(1).expand_as(B.data))

# Case C: (M,D) @ (D,) -> (M,)
# u_m = sum_i A_{m,i} y_i
# Let L = sum_m u_m = sum_{m,i} A_{m,i} y_i
# => dL/dA_{m,i} = y_i          (row i copied across M rows)
# => dL/dy_i = sum_m A_{m,i}    (column-sum of A)
A = Tensor(th.arange(6.0).reshape(2,3), requires_grad=True)  # shape (2,3)
y = Tensor(th.tensor([1.,2.,3.]), requires_grad=True)        # shape (3,)
u = matmul(A, y)                                             # shape (2,)
u.sum().backward()
print("C) A.grad:\n", A.grad)    # expected: rows all equal to y
print("C) y.grad:", y.grad)      # expected: A.sum(dim=0)
assert th.allclose(A.grad, y.data.unsqueeze(0).expand_as(A.data))
assert th.allclose(y.grad, A.data.sum(dim=0))

# Case D: (..., M, D) @ (..., D, N) with broadcasting -> (..., M, N)
# For L = sum(c), grads follow:
# dL/da = ones_like(c) @ b^T  (then reduced over broadcasted batch dims)
# dL/db = a^T @ ones_like(c)  (then reduced over broadcasted batch dims)
a = Tensor(th.randn(4,1,3,7), requires_grad=True)  # batch (4,1,3), M=3, D=7
b = Tensor(th.randn(   5,7,2), requires_grad=True)  # batch (5), D=7, N=2
c = matmul(a, b)                                    # -> (4,5,3,2)
c.sum().backward()
print("D) shapes:", a.grad.shape, b.grad.shape, c.data.shape)
assert a.grad.shape == a.data.shape
assert b.grad.shape == b.data.shape
print("matmul sanity passed ✅")
