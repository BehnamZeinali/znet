import mlx.core as mx
from .engine import Function

class CrossEntropy(Function):
    @staticmethod
    def forward(ctx, logits, targets, reduction="mean"):
        """
        logits: (N, C) float
        targets: (N,) int (class indices)
        reduction: "mean" | "sum" | "none"
        """
        t = mx.array(targets, dtype=mx.int64).reshape(-1)
        if logits.ndim != 2:
            raise ValueError(f"CrossEntropy expects logits of shape (N, C), got {logits.shape}")
        if t.shape[0] != logits.shape[0]:
            raise ValueError(f"targets length {t.shape[0]} != batch {logits.shape[0]}")

        # Stable softmax
        z = logits - mx.max(logits, axis=1, keepdims=True)
        exp = mx.exp(z)
        probs = exp / mx.sum(exp, axis=1, keepdims=True)

        N, C = logits.shape
        # one-hot without advanced indexing
        oh = (mx.arange(C)[None, :] == t[:, None]).astype(logits.dtype)
        p_true = mx.sum(probs * oh, axis=1)
        losses = -mx.log(p_true)

        # Save for backward
        ctx.save_for_backward(probs, t)
        ctx.meta["reduction"] = reduction
        ctx.meta["N"] = N

        if reduction == "mean":
            return mx.mean(losses)
        elif reduction == "sum":
            return mx.sum(losses)
        elif reduction == "none":
            return losses
        else:
            raise ValueError(f"Invalid reduction: {reduction!r}")

    def backward(self, grad_out):
        probs, t = self.ctx.saved_tensors
        reduction = self.ctx.meta["reduction"]
        N = self.ctx.meta["N"]
        C = probs.shape[1]

        # dL/dlogits = probs - one_hot(target)
        oh = (mx.arange(C)[None, :] == t[:, None]).astype(probs.dtype)
        grad_logits = probs - oh

        if reduction == "mean":
            grad_logits = grad_logits / N
            grad = grad_logits * grad_out  # scalar
        elif reduction == "sum":
            grad = grad_logits * grad_out  # scalar
        else:  # "none" -> grad_out shape (N,)
            grad = grad_logits * mx.reshape(grad_out, (-1, 1))

        return grad, None

def cross_entropy(logits, targets, reduction="mean"):
    return CrossEntropy.apply(logits, targets, reduction)
