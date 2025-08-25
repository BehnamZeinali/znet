
# znet/autograd/utils.py
# znet/autograd/_utils.py
import torch as th

def unbroadcast_like(grad: th.Tensor, target_shape: tuple[int, ...]) -> th.Tensor:
    # Drop leading dims
    while grad.ndim > len(target_shape):
        grad = grad.sum(dim=0)
    # Sum over broadcasted axes
    for i, (g, t) in enumerate(zip(grad.shape, target_shape)):
        if t == 1 and g != 1:
            grad = grad.sum(dim=i, keepdim=True)
    return grad

