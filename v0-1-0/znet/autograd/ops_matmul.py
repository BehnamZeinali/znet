import mlx.core as mx
from .engine import Function
from .utils import unbroadcast

"""
Emulate PyTorch/NumPy semantics:
  - (D,) @ (D,)      -> ()        [scalar]
  - (D,) @ (D,N)     -> (N,)
  - (M,D) @ (D,)     -> (M,)
  - (.., M, D) @ (.., D, N)  -> (.., M, N)  with broadcasting on leading dims
We promote 1D inputs to 2D with size-1 dims, track what to squeeze, then run @.
"""

def _promote_for_matmul(a, b):
    a_was_1d = (a.ndim == 1)
    b_was_1d = (b.ndim == 1)

    a_exp = mx.expand_dims(a, axis=0) if a_was_1d else a
    b_exp = mx.expand_dims(b, axis=-1) if b_was_1d else b

    if a_exp.shape[-1] != b_exp.shape[-2]:
        raise ValueError(
            f"matmul: inner dimensions must match, got {a.shape} @ {b.shape} "
            f"(treated as {a_exp.shape} @ {b_exp.shape})"
        )

    # Which axes to squeeze from the output
    squeeze_axes = ()
    if a_was_1d and b_was_1d:
        squeeze_axes = (-2, -1)  # (1,1) -> ()
    elif a_was_1d:
        squeeze_axes = (-2,)     # (1,N) -> (N,)
    elif b_was_1d:
        squeeze_axes = (-1,)     # (M,1) -> (M,)

    return a_exp, b_exp, a_was_1d, b_was_1d, squeeze_axes

class Matmul(Function):
    @staticmethod
    def forward(ctx, a, b):
        if a.ndim < 1 or b.ndim < 1:
            raise RuntimeError("matmul expects rank >= 1 on both inputs")

        a_exp, b_exp, a1d, b1d, squeeze_axes = _promote_for_matmul(a, b)
        y_exp = a_exp @ b_exp  # MLX handles leading-dim broadcasting

        y = y_exp
        if squeeze_axes:
            # squeeze in ascending axis order to keep indices stable
            axes = sorted([ax if ax >= 0 else y.ndim + ax for ax in squeeze_axes])
            for ax in axes:
                y = mx.squeeze(y, axis=ax)

        # Save inputs + flags for backward
        ctx.save_for_backward(a, b)
        ctx.meta["a1d"] = a1d
        ctx.meta["b1d"] = b1d
        ctx.meta["y_exp_shape"] = y_exp.shape  # for reshaping grad_out back

        return y

    def backward(self, grad_out):
        a, b = self.ctx.saved_tensors
        a1d = self.ctx.meta["a1d"]
        b1d = self.ctx.meta["b1d"]
        y_exp_shape = self.ctx.meta["y_exp_shape"]

        # Rebuild expanded inputs
        a_exp = mx.expand_dims(a, axis=0) if a1d else a
        b_exp = mx.expand_dims(b, axis=-1) if b1d else b

        # Unsqueeze grad to pre-squeeze shape
        g = mx.reshape(grad_out, y_exp_shape)

        # Raw grads in broadcasted shapes
        ga = g @ mx.swapaxes(b_exp, -1, -2)
        gb = mx.swapaxes(a_exp, -1, -2) @ g

        # Remove singleton axes introduced by 1D promotion
        if a1d and ga.ndim > a.ndim:
            ga = mx.squeeze(ga, axis=-2)  # (1,D) -> (D,)
        if b1d and gb.ndim > b.ndim:
            gb = mx.squeeze(gb, axis=-1)  # (D,1) -> (D,)

        # Reduce over broadcasted batch dims
        ga = unbroadcast(ga, a.shape)
        gb = unbroadcast(gb, b.shape)
        return ga, gb

def matmul(a, b):
    return Matmul.apply(a, b)
