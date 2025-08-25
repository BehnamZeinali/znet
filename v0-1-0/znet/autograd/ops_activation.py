
from .engine import Function

class ReLU(Function):
    @staticmethod
    def forward(ctx, x):
        # Save a compact boolean mask; cheaper than saving x
        mask = x > 0
        ctx.save_for_backward(mask)
        # x * mask keeps dtype of x, mask is {True,False}
        return x * mask

    def backward(self, grad_out):
        (mask,) = self.ctx.saved_tensors  # bool array
        # d/dx relu(x) = 1{x>0}; at x=0 we return 0 by convention
        return grad_out * mask

def relu(x):
    return ReLU.apply(x)
