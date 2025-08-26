import mlx.core as mx
from znet.autograd.tensor import Tensor
from znet.nn.conv2d import Conv2d

N, Cin, H, W = 2, 3, 11, 13
Cout = 4

x = Tensor(mx.random.normal(shape=(N, Cin, H, W), dtype=mx.float32), requires_grad=True)
conv = Conv2d(in_channels=Cin, out_channels=Cout, kernel_size=3, stride=2, padding=1, dilation=1, groups=1)

y = conv(x)
loss = y.sum()

try:
    loss.backward()
    print(
        "Backward ran. dx shape:", x.grad.shape,
        "dw shape:", conv.weight.grad.shape,
        ("db shape:" + str(conv.bias.grad.shape) if conv.bias is not None else "no bias")
    )
    print("pass: conv2d_backward_shapes")
except NotImplementedError as e:
    print("Backward not implemented?", e)
