import numpy as np
from znet.autograd.tensor import Tensor
from znet.nn.conv2d import Conv2d

np.random.seed(0)
N,Cin,H,W = 2,3,11,13
Cout = 4
x = Tensor(np.random.randn(N,Cin,H,W).astype(np.float32), requires_grad=True)
conv = Conv2d(in_channels=Cin, out_channels=Cout, kernel_size=3, stride=2, padding=1, dilation=1, groups=1)
y = conv(x)
loss = y.sum()
try:
    loss.backward()
    print("Backward ran. dx shape:", x.grad.shape, "dw shape:", conv.weight.grad.shape,
          ("db shape:"+str(conv.bias.grad.shape) if conv.bias is not None else "no bias"))
except NotImplementedError as e:
    print("Backward not implemented?", e)
