# Znet

Znet is a minimalist deep learning framework implemented in pure Python + NumPy, inspired by PyTorch.

It aims to reproduce PyTorch's API and design philosophy, but with a fully transparent and educational codebase — ideal for learning how deep learning frameworks work under the hood.

## Features

- `zenet.nn.Module` — base class for neural network components
- `zenet.nn.Linear`, `ReLU` — common layers
- `zenet.optim.SGD` — simple optimizer
- `zenet.autograd.Tensor` — gradient-tracking wrapper
- PyTorch-style API: `loss.backward()`, `optimizer.step()`, `model.parameters()`

## Example Usage

```python
from zenet.nn import Linear, ReLU, Module
from zenet.optim import SGD
from zenet.autograd import Tensor
import numpy as np

class MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 128)
        self.relu = ReLU()
        self.fc2 = Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

x = Tensor(np.random.randn(32, 784), requires_grad=True)
target = np.random.randint(0, 10, size=(32,))

model = MLP()
output = model(x)
loss_fn = CrossEntropyLoss()
loss = loss_fn(output, target)
loss.backward()

optimizer = SGD(model.parameters(), lr=0.01)
optimizer.step()
optimizer.zero_grad()
