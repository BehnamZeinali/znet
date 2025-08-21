from .module import Module
from ..autograd.tensor import Tensor
from ..autograd.ops_loss import cross_entropy as ce_op

class CrossEntropyLoss(Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        # targets should be integer class indices; we won’t require_grad on them
        return ce_op(logits, targets, self.reduction)
