# znet/autograd/ops_matmul.py
from __future__ import annotations
import torch as th
from .engine import Function
from .tensor import Tensor
from .utils import unbroadcast_like


def _promote_for_matmul(a: th.Tensor, b: th.Tensor):
    a1d = (a.ndim == 1)
    b1d = (b.ndim == 1)
    a_exp = a.unsqueeze(0) if a1d else a        # (1, D) or (..., M, D)
    b_exp = b.unsqueeze(-1) if b1d else b       # (D, 1) or (..., D, N)

    if a_exp.shape[-1] != b_exp.shape[-2]:
        raise ValueError(
            f"matmul: inner dimensions must match, got {tuple(a.shape)} @ {tuple(b.shape)} "
            f"(treated as {tuple(a_exp.shape)} @ {tuple(b_exp.shape)})"
        )

    if a1d and b1d:      squeeze_axes = (-2, -1)  # (1,1) -> ()
    elif a1d:            squeeze_axes = (-2,)     # (1,N) -> (N,)
    elif b1d:            squeeze_axes = (-1,)     # (M,1) -> (M,)
    else:                squeeze_axes = ()
    return a_exp, b_exp, a1d, b1d, squeeze_axes

class Matmul(Function):
    @staticmethod
    def forward(ctx, a: th.Tensor, b: th.Tensor):
        if a.ndim < 1 or b.ndim < 1:
            raise RuntimeError("matmul expects rank >= 1 on both inputs")

        a_exp, b_exp, a1d, b1d, squeeze_axes = _promote_for_matmul(a, b)
        y_exp = a_exp @ b_exp

        y = y_exp
        if squeeze_axes:
            # IMPORTANT: squeeze from highest axis to lowest so indices don't shift
            axes = sorted([(ax if ax >= 0 else y.ndim + ax) for ax in squeeze_axes], reverse=True)
            for ax in axes:
                y = y.squeeze(dim=ax)

        ctx.save_for_backward(a, b)
        ctx.meta["a1d"] = a1d
        ctx.meta["b1d"] = b1d
        ctx.meta["y_exp_shape"] = tuple(y_exp.shape)
        return y

    def backward(self, grad_out: th.Tensor):
        a, b = self.ctx.saved_tensors
        a1d = self.ctx.meta["a1d"]
        b1d = self.ctx.meta["b1d"]
        y_exp_shape = self.ctx.meta["y_exp_shape"]

        a_exp = a.unsqueeze(0) if a1d else a
        b_exp = b.unsqueeze(-1) if b1d else b

        g = grad_out.reshape(y_exp_shape)

        ga = g @ b_exp.transpose(-1, -2)
        gb = a_exp.transpose(-1, -2) @ g

        if a1d and ga.ndim > a.ndim:
            ga = ga.squeeze(dim=-2)  # (1, D) -> (D,)
        if b1d and gb.ndim > b.ndim:
            gb = gb.squeeze(dim=-1)  # (D, 1) -> (D,)

        ga = unbroadcast_like(ga, tuple(a.shape))
        gb = unbroadcast_like(gb, tuple(b.shape))
        return ga, gb

def matmul(a, b):
    if not isinstance(a, Tensor): a = Tensor(a)
    if not isinstance(b, Tensor): b = Tensor(b)
    return Matmul.apply(a, b)
