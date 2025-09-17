# znet/nn/residual.py
import numpy as np
from .module import Module
from .conv2d import Conv2d
from .activation import ReLU

class ResidualBlock2d(Module):
    """
    Basic ResNet block:
      y = Conv3x3(in->out, stride) → ReLU → Conv3x3(out->out, stride=1)
      s = x            if (in==out and stride==1)
        = Conv1x1(in->out, stride)  otherwise
      out = ReLU(y + s)
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = Conv2d(in_channels=in_ch, out_channels=out_ch,
                            kernel_size=3, stride=stride, padding=1)
        self.relu1 = ReLU()
        self.conv2 = Conv2d(in_channels=out_ch, out_channels=out_ch,
                            kernel_size=3, stride=1, padding=1)

        # projection for shape/channel change
        self.proj = None
        if (in_ch != out_ch) or (stride != 1):
            self.proj = Conv2d(in_channels=in_ch, out_channels=out_ch,
                               kernel_size=1, stride=stride, padding=0)

        # Optional: a final ReLU after the skip-add
        self.relu_out = ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.conv2(y)

        s = x if self.proj is None else self.proj(x)  # skip path

        out = y + s    # relies on your Tensor __add__ autograd
        out = self.relu_out(out)
        return out
