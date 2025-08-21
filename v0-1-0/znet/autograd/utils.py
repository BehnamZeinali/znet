# znet/autograd/utils.py
import numpy as np

def unbroadcast(grad, target_shape):
    """
    Sum grad over broadcasted dimensions so that it matches target_shape.
    Works for scalars, vectors, and higher ranks; target_shape may have 1s
    where grad had larger sizes.
    """
    g = grad
    # Remove leading axes
    while g.ndim > len(target_shape):
        g = g.sum(axis=0)
    # Sum along axes where target has 1
    for axis, (gs, ts) in enumerate(zip(g.shape, target_shape)):
        if ts == 1 and gs != 1:
            g = g.sum(axis=axis, keepdims=True)
    return g.reshape(target_shape)
