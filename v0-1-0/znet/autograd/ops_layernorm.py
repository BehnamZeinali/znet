import numpy as np
from .engine import Function

class LayerNormFn(Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, eps, axes):
        """
        x:    np.ndarray, arbitrary shape
        gamma: np.ndarray, shape == normalized_shape
        beta:  np.ndarray, shape == normalized_shape
        eps:  float (stability)
        axes: tuple of ints — the normalized axes (typically the last k axes)
        """
        # reshape gamma/beta for broadcasting over leading (batch) dims
        k = gamma.ndim
        lead = x.ndim - k
        bshape = (1,) * lead + tuple(gamma.shape)

        mu  = np.mean(x, axis=axes, keepdims=True)
        var = np.mean((x - mu) ** 2, axis=axes, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + eps)

        xhat = (x - mu) * inv_std
        y = xhat * gamma.reshape(bshape) + beta.reshape(bshape)

        # save for backward
        ctx.save_for_backward(xhat, inv_std, gamma.reshape(bshape))
        ctx.meta.update({
            "axes": tuple(axes),
            "gamma_shape": tuple(gamma.shape),
            "beta_shape":  tuple(beta.shape),
            "x_shape":     tuple(x.shape),
        })
        return y

    def backward(self, g_out):
        xhat, inv_std, gamma_b = self.ctx.saved_tensors
        axes        = self.ctx.meta["axes"]
        gamma_shape = self.ctx.meta["gamma_shape"]
        x_shape     = self.ctx.meta["x_shape"]

        # upstream to xhat
        dxhat = g_out * gamma_b

        # sums over normalized axes
        m = 1
        for ax in axes:
            m *= x_shape[ax]
        m = float(m)

        sum_dxhat       = np.sum(dxhat, axis=axes, keepdims=True)
        sum_dxhat_xhat  = np.sum(dxhat * xhat, axis=axes, keepdims=True)

        # compact LN gradient:
        # dx = (1/m) * inv_std * [ m*dxhat - sum(dxhat) - xhat*sum(dxhat*xhat) ]
        dx = (inv_std / m) * (m * dxhat - sum_dxhat - xhat * sum_dxhat_xhat)

        # dgamma, dbeta — reduce over leading (batch) dims only
        lead_axes = tuple(range(0, len(x_shape) - len(gamma_shape)))
        dgamma = np.sum(g_out * xhat, axis=lead_axes, keepdims=False)
        dbeta  = np.sum(g_out,        axis=lead_axes, keepdims=False)

        return dx, dgamma, dbeta, None, None  # grads only for Tensor parents

def layer_norm(x, gamma, beta, eps=1e-5, axes=(-1,)):
    """
    Functional LayerNorm — normalize x over 'axes' using gamma/beta.
    All args except x/gamma/beta are non-tensors, so no grads expected for them.
    """
    return LayerNormFn.apply(x, gamma, beta, eps, tuple(axes))
