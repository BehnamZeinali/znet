import numpy as np
from znet.autograd import Tensor
from znet.nn.attention import CausalSelfAttention

np.random.seed(0)
B, T, D, H = 2, 4, 16, 4   # head_dim = 4
x = Tensor(np.random.randn(B, T, D).astype(np.float32), requires_grad=True)

attn = CausalSelfAttention(d_model=D, n_heads=H)
y = attn(x)             # (B,T,D)
loss = y.sum()
loss.backward()

print("y shape:", y.shape)                     # (2,4,16)
print("dx shape:", x.grad.shape)               # (2,4,16)
print("q.w grad:", attn.q.weight.grad.shape)   # (D, D)
print("k.w grad:", attn.k.weight.grad.shape)
print("v.w grad:", attn.v.weight.grad.shape)
print("o.w grad:", attn.out.weight.grad.shape)

# import numpy as np
# from znet.autograd import Tensor
# from znet.nn.attention import CausalSelfAttention

# np.random.seed(0)
# B,T,D,H = 2,4,16,4
# x = Tensor(np.random.randn(B,T,D).astype(np.float32), requires_grad=True)
# attn = CausalSelfAttention(d_model=D, n_heads=H)  # uses nn.Softmax(axis=-1)
# y = attn(x)
# loss = y.sum()
# loss.backward()
# print("y shape:", y.shape)           # (2,4,16)
# print("x.grad shape:", x.grad.shape) # (2,4,16)  ✅


import numpy as np
from znet.autograd import Tensor
from znet.nn.linear import Linear

np.random.seed(0)
B,T,D = 2,4,16

# Variant A: simple Linear -> sum
x = Tensor(np.random.randn(B,T,D).astype(np.float32), requires_grad=True)
lin = Linear(D, D)
y = lin(x)
(y.sum()).backward()
print("A:", x.grad is not None)   # must be True

# Variant B: q@k^T without scale/mask/softmax
x = Tensor(np.random.randn(B,T,D).astype(np.float32), requires_grad=True)
q = lin(x); k = lin(x)
scores = q @ k.swapaxes(-1, -2)   # (B,T,T)
(scores.sum()).backward()
print("B:", x.grad is not None)   # must be True (tests matmul + swapaxes)

# Variant C: add scale only
x = Tensor(np.random.randn(B,T,D).astype(np.float32), requires_grad=True)
q = lin(x); k = lin(x)
scores = (q @ k.swapaxes(-1, -2)) * (1.0 / np.sqrt(D))
(scores.sum()).backward()
print("C:", x.grad is not None)   # if False -> Mul backward is buggy

# Variant D: add mask but NO softmax
x = Tensor(np.random.randn(B,T,D).astype(np.float32), requires_grad=True)
q = lin(x); k = lin(x)
scores = q @ k.swapaxes(-1, -2)   # (B,T,T)
Tlen = scores.shape[-1]
mask_np = np.triu(np.ones((Tlen,Tlen), dtype=np.float32), k=1) * (-1e30)
mask_np = np.broadcast_to(mask_np, (B, 1, Tlen, Tlen))   # no head dim
mask = Tensor(mask_np, requires_grad=False)
scores = scores.reshape((B,1,Tlen,Tlen)) + mask          # force same shape
(scores.sum()).backward()
print("D:", x.grad is not None)   # if False -> Add backward (broadcast) is buggy


