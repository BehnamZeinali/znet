import numpy as np
from znet.autograd import Tensor
from znet.autograd.ops_softmax import softmax

np.random.seed(0)
x = Tensor(np.random.randn(2,3).astype(np.float32), requires_grad=True)
y = softmax(x, axis=-1)
loss = y.sum()
loss.backward()
print("x.grad shape:", x.grad.shape)  # (2,3)

x = Tensor(np.array([[2.,1.,0.]], dtype=np.float32), requires_grad=True)
y = softmax(x, axis=-1)
print(y.data)  # [[0.6652, 0.2447, 0.0900]] approx
(y.sum()).backward()
print(x.grad.shape)  # (1,3)
