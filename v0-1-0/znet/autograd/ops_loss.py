import numpy as np
from .engine import Function

class CrossEntropy(Function):
    @staticmethod
    def forward(ctx, logits, targets, reduction="mean"):
        """
        logits: (N, C) float
        targets: (N,) int (class indices)
        reduction: "mean" | "sum" | "none"
        """
        # normalize targets and compute stable softmax
        t = np.asarray(targets).astype(np.int64).reshape(-1)
        if logits.ndim != 2:
            raise ValueError(f"CrossEntropy expects logits of shape (N, C), got {logits.shape}")
        if t.shape[0] != logits.shape[0]:
            raise ValueError(f"targets length {t.shape[0]} != batch {logits.shape[0]}")

        z = logits - logits.max(axis=1, keepdims=True)    # stability
        exp = np.exp(z)
        probs = exp / exp.sum(axis=1, keepdims=True)

        N = logits.shape[0]
        losses = -np.log(probs[np.arange(N), t])

        # Save for backward
        ctx.save_for_backward(probs, t)
        ctx.meta["reduction"] = reduction
        ctx.meta["N"] = N

        if reduction == "mean":
            return np.array(losses.mean(), dtype=logits.dtype)
        elif reduction == "sum":
            return np.array(losses.sum(), dtype=logits.dtype)
        elif reduction == "none":
            return losses.astype(logits.dtype)
        else:
            raise ValueError(f"Invalid reduction: {reduction!r}")

    def backward(self, grad_out):
        probs, t = self.ctx.saved_tensors
        reduction = self.ctx.meta["reduction"]
        N = self.ctx.meta["N"]

        # dL/dlogits = probs; subtract 1 at the target indices
        grad_logits = probs.copy()
        grad_logits[np.arange(N), t] -= 1

        if reduction == "mean":
            grad_logits = grad_logits / N
            grad = grad_logits * grad_out  # grad_out is scalar ()
        elif reduction == "sum":
            grad = grad_logits * grad_out  # scalar multiplier
        else:  # "none" -> grad_out shape (N,)
            grad = grad_logits * grad_out.reshape(-1, 1)

        # parents are (logits, targets) — no grad for targets
        return grad, None

def cross_entropy(logits, targets, reduction="mean"):
    return CrossEntropy.apply(logits, targets, reduction)
