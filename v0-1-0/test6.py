import numpy as np
from znet.autograd.tensor import Tensor
from znet.nn.conv2d import Conv2d

N,C,H,W = 2,3,32,32
x = Tensor(np.random.randn(N,C,H,W).astype(np.float32), requires_grad=False)

conv = Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=2, padding=2)
y = conv(x)
print("y shape:", y.shape)  # expected (2, 8, 16, 16)

conv_dilated = Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=2, dilation=2)
y2 = conv_dilated(x)
print("y2 shape:", y2.shape)  # expected (2, 4, 32, 32)
