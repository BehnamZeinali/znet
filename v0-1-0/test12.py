# test_residual_gradcheck.py
import numpy as np

from znet.autograd import Tensor
from znet.nn.residual import ResidualBlock2d   # the block we wrote: Conv3x3->ReLU->Conv3x3 + skip

np.random.seed(0)

# Small, no-projection block: in_ch=out_ch, stride=1
B, C, H, W = 1, 3, 5, 5
block = ResidualBlock2d(in_ch=C, out_ch=C, stride=1)

# Deterministic tiny input
x = Tensor(np.random.randn(B, C, H, W).astype(np.float32), requires_grad=True)

# ---- AUTOGRAD GRADS (analytic) ----
y = block(x)               # (1, 3, 5, 5)
loss = y.sum()             # upstream grad = 1 everywhere
loss.backward()

# Pick one element to check
w1_idx = (0, 0, 1, 1)      # conv1.weight[o, i, r, s]
w2_idx = (0, 0, 1, 1)      # conv2.weight[o, i, r, s]
x_idx  = (0, 0, 0, 0)      # input element

g_w1_autograd = block.conv1.weight.grad[w1_idx]
g_w2_autograd = block.conv2.weight.grad[w2_idx]
g_x_autograd  = x.grad[x_idx]

# ---- FINITE DIFFERENCE (numeric) ----
eps = 1e-4

def f_value():
    # forward only, return scalar loss value
    y_ = block(x)
    return y_.data.sum()   # detach to Python float for numeric diff

# Helper to finite-diff a single tensor element
def finite_diff_param(param_tensor, index, eps=1e-4):
    orig = param_tensor.data[index]
    param_tensor.data[index] = orig + eps
    f_pos = f_value()
    param_tensor.data[index] = orig - eps
    f_neg = f_value()
    param_tensor.data[index] = orig  # restore
    return (f_pos - f_neg) / (2 * eps)

# For input x we need a separate helper (since x is also a Tensor)
def finite_diff_input(x_tensor, index, eps=1e-4):
    orig = x_tensor.data[index]
    x_tensor.data[index] = orig + eps
    f_pos = f_value()
    x_tensor.data[index] = orig - eps
    f_neg = f_value()
    x_tensor.data[index] = orig
    return (f_pos - f_neg) / (2 * eps)

g_w1_num = finite_diff_param(block.conv1.weight, w1_idx, eps)
g_w2_num = finite_diff_param(block.conv2.weight, w2_idx, eps)
g_x_num  = finite_diff_input(x, x_idx, eps)

# ---- REPORT ----
def rel_err(a, b, tol=1e-2):
    denom = max(1.0, abs(a) + abs(b))
    return abs(a - b) / denom

print("Check conv1.weight", w1_idx)
print(" autograd:", float(g_w1_autograd), " numeric:", float(g_w1_num),
      " rel.err:", rel_err(float(g_w1_autograd), float(g_w1_num)))

print("Check conv2.weight", w2_idx)
print(" autograd:", float(g_w2_autograd), " numeric:", float(g_w2_num),
      " rel.err:", rel_err(float(g_w2_autograd), float(g_w2_num)))

print("Check input x", x_idx)
print(" autograd:", float(g_x_autograd), " numeric:", float(g_x_num),
      " rel.err:", rel_err(float(g_x_autograd), float(g_x_num)))
