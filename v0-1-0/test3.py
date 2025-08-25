# quick_ce_check.py
import torch as th
from znet.autograd.tensor import Tensor
from znet.autograd.ops_loss import cross_entropy

def softmax(x): return th.softmax(x, dim=1)

# ------------------------
# Case 1: reduction="mean"
# ------------------------
# L = (1/N) * sum_n -log softmax(logits)[n, t[n]]
# => dL/dlogits = (softmax - one_hot(targets)) / N
logits = Tensor(th.tensor([[2.0, 0.5, -1.0],
                           [1.0, 3.0, 0.1]], dtype=th.float32), requires_grad=True)
targets = [0, 2]
loss = cross_entropy(logits, targets, reduction="mean")
loss.backward()

with th.no_grad():
    N = 2
    S = softmax(logits.data)
    exp_grad = S.clone()
    exp_grad[th.arange(N), th.tensor(targets)] -= 1
    exp_grad = exp_grad / N
    print("Case1 logits.grad:\n", logits.grad)
    print("Case1 expected:\n", exp_grad)
    assert th.allclose(logits.grad, exp_grad, atol=1e-6)

# ----------------------
# Case 2: reduction="sum"
# ----------------------
# L = sum_n -log p[y_n]
# => dL/dlogits = softmax - one_hot
logits = Tensor(th.tensor([[0.2, -0.1, 0.0],
                           [0.5,  1.0, 2.0]], dtype=th.float32), requires_grad=True)
targets = th.tensor([1, 2])
loss = cross_entropy(logits, targets, reduction="sum")
loss.backward()

with th.no_grad():
    S = softmax(logits.data)
    exp_grad = S.clone()
    exp_grad[th.arange(2), targets] -= 1
    print("Case2 logits.grad:\n", logits.grad)
    print("Case2 expected:\n", exp_grad)
    assert th.allclose(logits.grad, exp_grad, atol=1e-6)

# ------------------------
# Case 3: reduction="none"
# ------------------------
# L_n = -log p[y_n]; upstream gradient g of shape (N,)
# => dL/dlogits[n,:] = (softmax - one_hot)[n,:] * g[n]
logits = Tensor(th.tensor([[1.0, 2.0, 3.0],
                           [0.0, 0.0, 0.0],
                           [5.0, 1.0, 0.0]], dtype=th.float32, device="cuda"),  # or default device
                requires_grad=True)
targets = [2, 0, 1]
loss_vec = cross_entropy(logits, targets, reduction="none")  # shape (3,)

g = th.tensor([1.0, 2.0, 0.5], dtype=th.float32).to(logits.data.device)  # <-- move to same device
loss_vec.backward(g)

with th.no_grad():
    S = th.softmax(logits.data, dim=1)
    exp_grad = S.clone()
    exp_grad[th.arange(3, device=logits.data.device), th.tensor(targets, device=logits.data.device)] -= 1
    exp_grad = exp_grad * g.reshape(-1, 1)  # g now on the same device ✅

    print("Case3 logits.grad:\n", logits.grad)
    print("Case3 expected:\n", exp_grad)
    assert th.allclose(logits.grad, exp_grad, atol=1e-6)

print("CrossEntropy sanity passed ✅")
