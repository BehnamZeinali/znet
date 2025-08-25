# znet/optim/sgd.py
from __future__ import annotations
import torch as th

class SGD:
    """
    Tensor-native SGD optimizer (Torch backend).

    params: iterable of Tensor (e.g., model.parameters())
    lr: learning rate
    momentum: 0.0 = no momentum
    weight_decay: L2 penalty (adds wd * p.data to the gradient)
    nesterov: use Nesterov momentum (requires momentum > 0)
    """
    def __init__(self, params, lr=1e-2, momentum=0.0, weight_decay=0.0, nesterov=False):
        self.params = list(params)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.nesterov = bool(nesterov)
        if self.nesterov and self.momentum <= 0.0:
            raise ValueError("nesterov=True requires momentum > 0")

        # per-parameter state (e.g., momentum buffers)
        # id(param) -> {"buf": torch.Tensor}
        self._state = {}

    def step(self):
        # Make sure we do not create autograd history in Torch (we don't use it).
        with th.no_grad():
            for p in self.params:
                if p is None or getattr(p, "grad", None) is None:
                    continue

                # grads as torch tensor on the same device/dtype as param
                g = p.grad
                if not isinstance(g, th.Tensor):
                    g = th.as_tensor(g, device=p.data.device, dtype=p.data.dtype)
                else:
                    g = g.to(device=p.data.device, dtype=p.data.dtype)

                # weight decay (L2)
                if self.weight_decay != 0.0:
                    g = g.add(p.data, alpha=self.weight_decay)

                # momentum
                if self.momentum != 0.0:
                    st = self._state.setdefault(id(p), {})
                    buf = st.get("buf", None)

                    if (buf is None or
                        not isinstance(buf, th.Tensor) or
                        buf.device != p.data.device or
                        buf.dtype  != p.data.dtype or
                        buf.shape  != p.data.shape):
                        # initialize buffer as a copy of the gradient
                        buf = g.clone()
                    else:
                        # buf = momentum * buf + g
                        buf.mul_(self.momentum).add_(g)

                    st["buf"] = buf

                    # Nesterov or classic momentum
                    if self.nesterov:
                        # update = g + momentum * buf
                        update = g.add(buf, alpha=self.momentum)
                    else:
                        # classic: use buffer directly
                        update = buf
                else:
                    update = g

                # in-place param update: p = p - lr * update
                p.data.add_(update, alpha=-self.lr)

    def zero_grad(self):
        # Delegate to Tensor API; keeps you backend-agnostic
        for p in self.params:
            if p is not None:
                p.zero_grad()
