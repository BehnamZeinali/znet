# znet/autograd/ops_cross_entropy.py
from __future__ import annotations
import torch as th
from .engine import Function
from .tensor import Tensor

class CrossEntropy(Function):
    @staticmethod
    def forward(ctx, logits: th.Tensor, targets, reduction: str = "mean"):
        """
        logits:   (N, C) float torch.Tensor
        targets:  (N,)   int class indices (any array-like) or torch tensor
        reduction: "mean" | "sum" | "none"
        """
        if logits.ndim != 2:
            raise ValueError(f"CrossEntropy expects logits of shape (N, C), got {tuple(logits.shape)}")

        # normalize/validate targets
        t = targets if isinstance(targets, th.Tensor) else th.as_tensor(targets, device=logits.device)
        t = t.to(dtype=th.long).reshape(-1)
        N, C = logits.shape
        if t.shape[0] != N:
            raise ValueError(f"targets length {t.shape[0]} != batch {N}")

        # Stable log_softmax
        # log_probs[n, c] = logits[n, c] - logsumexp(logits[n, :])
        log_probs = th.log_softmax(logits, dim=1)
        # probs needed for backward
        probs = log_probs.exp()  # softmax

        # per-example loss: -log p[y_n]
        losses = -log_probs[th.arange(N, device=logits.device), t]

        # Save for backward
        ctx.save_for_backward(probs, t)
        ctx.meta["reduction"] = reduction
        ctx.meta["N"] = N

        if reduction == "mean":
            return losses.mean().to(dtype=logits.dtype)
        elif reduction == "sum":
            return losses.sum().to(dtype=logits.dtype)
        elif reduction == "none":
            return losses.to(dtype=logits.dtype)
        else:
            raise ValueError(f"Invalid reduction: {reduction!r}")

    def backward(self, grad_out: th.Tensor):
        """
        Let S = softmax(logits). For each row n:
          dL/dlogits[n, :] = S[n, :] - one_hot(t[n])
        With reduction="mean": divide by N.
        With "sum": unchanged.
        With "none": scale each row by grad_out[n].
        """
        probs, t = self.ctx.saved_tensors
        reduction = self.ctx.meta["reduction"]
        N = self.ctx.meta["N"]

        # probs clone so we don't mutate saved tensor
        grad_logits = probs.clone()
        grad_logits[th.arange(N, device=probs.device), t] -= 1

        if reduction == "mean":
            grad_logits = grad_logits / N
            grad = grad_logits * grad_out           # grad_out is scalar ()
        elif reduction == "sum":
            grad = grad_logits * grad_out           # scalar multiplier
        else:  # "none" -> grad_out shape (N,)
            grad = grad_logits * grad_out.reshape(-1, 1)

        # parents are (logits, targets[, reduction_str-not-a-parent])
        return grad, None, None

def cross_entropy(logits, targets, reduction: str = "mean"):
    if not isinstance(logits, Tensor): logits = Tensor(logits)
    # targets can be raw list/ndarray/torch tensor or a znet.Tensor; either is fine
    return CrossEntropy.apply(logits, targets, reduction)
