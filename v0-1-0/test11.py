import numpy as np
from znet.autograd import Tensor
from znet.nn.residual import ResidualBlock2d

np.random.seed(0)
B, Cin, H, W = 2, 8, 28, 28
x = Tensor(np.random.randn(B, Cin, H, W).astype(np.float32), requires_grad=True)

block = ResidualBlock2d(in_ch=8, out_ch=16, stride=2)  # downsample & expand channels
y = block(x)             # (B, 16, 14, 14)
loss = y.sum()
loss.backward()

print("y shape:", y.shape)
# Check a few grads exist
print("dx shape:", x.grad.shape)
print("conv1.w grad:", block.conv1.weight.grad.shape)
print("conv2.w grad:", block.conv2.weight.grad.shape)
if block.proj is not None:
    print("proj.w grad:", block.proj.weight.grad.shape)
