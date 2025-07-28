import numpy as np
from ..autograd import Tensor

class CrossEntropyLoss:
    def __call__(self, logits: Tensor, targets: np.ndarray):
        # Softmax + negative log likelihood
        shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=1, keepdims=True)

        self.batch_size = logits.data.shape[0]
        self.probs = probs
        self.targets = targets

        correct_log_probs = -np.log(probs[np.arange(self.batch_size), targets])
        loss_value = np.mean(correct_log_probs)

        loss = Tensor(np.array(loss_value), requires_grad=True)

        def _grad_fn(_):
            grad = probs
            grad[np.arange(self.batch_size), targets] -= 1
            grad /= self.batch_size
            return grad

        loss.set_backward(_grad_fn, [logits])
        return loss


# class CrossEntropyLoss:
#     def __call__(self, logits: Tensor, targets: np.ndarray):
#         shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
#         exp_logits = np.exp(shifted)
#         probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

#         batch_size = logits.data.shape[0]
#         correct_log_probs = -np.log(probs[np.arange(batch_size), targets])
#         loss_value = np.mean(correct_log_probs)

#         loss = Tensor(np.array(loss_value), requires_grad=True)

#         def _grad_fn(_):
#             grad = probs
#             grad[np.arange(batch_size), targets] -= 1
#             grad /= batch_size
#             return grad  # This goes to logits

#         loss.set_backward(_grad_fn, [logits])
#         return loss