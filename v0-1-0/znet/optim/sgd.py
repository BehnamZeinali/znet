# znet/optim/sgd.py
import numpy as np

class SGD:
    """
    Tensor-native SGD optimizer.

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
        self._state = {}  # id(param) -> {"buf": ndarray}

    def step(self):
        for p in self.params:
            if p is None or p.grad is None:
                continue

            g = p.grad
            # ensure dtype matches param
            if g.dtype != p.data.dtype:
                g = g.astype(p.data.dtype, copy=False)

            # weight decay (L2)
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data

            if self.momentum != 0.0:
                st = self._state.setdefault(id(p), {})
                buf = st.get("buf")
                if buf is None:
                    # init momentum buffer as gradient
                    buf = np.array(g, dtype=p.data.dtype, copy=True)
                else:
                    buf *= self.momentum
                    buf += g
                st["buf"] = buf

                if self.nesterov:
                    # g + momentum*buf
                    update = g + self.momentum * buf
                else:
                    # classic momentum: use buffer directly
                    update = buf
            else:
                update = g

            # in-place param update
            p.data -= self.lr * update

    def zero_grad(self):
        # Delegate to Tensor API; keeps you backend-agnostic
        for p in self.params:
            if p is not None:
                p.zero_grad()
