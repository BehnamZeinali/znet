# quick_conv_check.py
import torch as th
import torch.nn.functional as F
from znet.autograd.tensor import Tensor
from znet.autograd.ops_conv import conv2d

# Expected gradients (comments):
# Let y = conv2d(x, w, b). For scalar L = sum(y):
# dL/db = sum_{n,h,w} 1   -> grad_out.sum over (N,H,W)
# dL/dw = sum_{n,h,w} x_patches[n,h,w] (a cross-correlation with grad_out)
# dL/dx = conv_transpose2d(grad_out, w, stride, padding, output_padding, groups, dilation)

dev = th.device("cuda" if th.cuda.is_available() else "cpu")
N, C_in, C_out = 2, 3, 4
H, W = 7, 6
kH, kW = 3, 2
stride = (2, 1)
padding = (1, 0)
dilation = (1, 1)
groups = 1

x = Tensor(th.randn(N, C_in, H, W, device=dev), requires_grad=True)
w = Tensor(th.randn(C_out, C_in//groups, kH, kW, device=dev), requires_grad=True)
b = Tensor(th.randn(C_out, device=dev), requires_grad=True)

y = conv2d(x, w, b, stride=stride, padding=padding, dilation=dilation, groups=groups)
y.sum().backward()

# Numerical check vs PyTorch autograd on separate tensors
x_t = x.data.clone().requires_grad_(True)
w_t = w.data.clone().requires_grad_(True)
b_t = b.data.clone().requires_grad_(True)
y_t = F.conv2d(x_t, w_t, b_t, stride=stride, padding=padding, dilation=dilation, groups=groups)
y_t.sum().backward()

print("dx close:", th.allclose(x.grad, x_t.grad, atol=1e-5, rtol=1e-5))
print("dw close:", th.allclose(w.grad, w_t.grad, atol=1e-4, rtol=1e-3))
print("db close:", th.allclose(b.grad, b_t.grad, atol=1e-5, rtol=1e-5))
print("max abs diff:", (w.grad - w_t.grad).abs().max().item())
print("mean abs diff:", (w.grad - w_t.grad).abs().mean().item())
# print(w_t.grad)
# print('-----------------')
# print(w.grad)