# quick_conv_check_mlx_compare.py
import mlx.core as mx
from znet.autograd.tensor import Tensor
from znet.autograd.ops_conv_2 import conv2d

# Compare our from-scratch Conv2d (autograd) vs MLX's built-in conv2d + conv2d_backward

N, C_in, C_out = 2, 3, 4
H, W = 7, 6
kH, kW = 3, 2
stride = (2, 1)
padding = (1, 0)
dilation = (1, 1)
groups = 1

# Initialize MLX arrays (NCHW + OIHW for our implementation)
x_mx = mx.random.normal(shape=(N, C_in, H, W), dtype=mx.float32)
w_mx = mx.random.normal(shape=(C_out, C_in // groups, kH, kW), dtype=mx.float32)  # OIHW
b_mx = mx.random.normal(shape=(C_out,), dtype=mx.float32)

# --- Our implementation (Tensor + from-scratch conv) ---
x = Tensor(x_mx, requires_grad=True)
w = Tensor(w_mx, requires_grad=True)
b = Tensor(b_mx, requires_grad=True)

y = conv2d(x, w, b, stride=stride, padding=padding, dilation=dilation, groups=groups)
y.sum().backward()  # populates x.grad, w.grad, b.grad
print("-----------------------------------")

# --- MLX reference (built-in conv + analytic backward) ---
# MLX conv2d expects NHWC input but **still** uses OIHW weights.
x_nhwc = mx.transpose(x_mx, (0, 2, 3, 1))  # (N, H, W, C)

# Forward (use OIHW weight directly)
y_ref_nhwc = mx.conv2d(x_nhwc, w_mx, stride=stride, padding=padding, dilation=dilation, groups=groups)
y_ref_nhwc = y_ref_nhwc + b_mx.reshape(1, 1, 1, -1)  # bias add in channel-last

# Upstream grad for sum is all ones (same shape as y_ref_nhwc)
gout = mx.ones_like(y_ref_nhwc)

# Backward in MLX layout; returns dx in NHWC and dw in OIHW
dx_ref_nhwc, dw_ref = mx.conv2d_backward(x_nhwc, w_mx, gout, stride, padding, dilation, groups)
db_ref = mx.sum(gout, axis=(0, 1, 2))  # sum over N,H,W -> (C_out,)

# Convert MLX reference outputs to our layouts for comparison
y_ref = mx.transpose(y_ref_nhwc, (0, 3, 1, 2))   # NCHW
dx_ref = mx.transpose(dx_ref_nhwc, (0, 3, 1, 2)) # NCHW

# --- Comparisons ---
print("Forward close (y):", bool(mx.allclose(y.data, y_ref, atol=1e-5, rtol=1e-5)))
print("dx close:", bool(mx.allclose(x.grad, dx_ref, atol=1e-5, rtol=1e-5)))
print("dw close:", bool(mx.allclose(w.grad, dw_ref, atol=1e-4, rtol=1e-3)))
print("db close:", bool(mx.allclose(b.grad, db_ref, atol=1e-5, rtol=1e-5)))

# Diagnostics
dx_diff = mx.abs(x.grad - dx_ref)
dw_diff = mx.abs(w.grad - dw_ref)
db_diff = mx.abs(b.grad - db_ref)

print("dx max abs diff:", float(mx.max(dx_diff)))
print("dw max abs diff:", float(mx.max(dw_diff)))
print("db max abs diff:", float(mx.max(db_diff)))

print("dx mean abs diff:", float(mx.mean(dx_diff)))
print("dw mean abs diff:", float(mx.mean(dw_diff)))
print("db mean abs diff:", float(mx.mean(db_diff)))

# Shape sanity
print("y shape (ours):", y.shape, "y shape (mlx):", y_ref.shape)
print("dx shape:", x.grad.shape, "dw shape:", w.grad.shape, "db shape:", b.grad.shape)
