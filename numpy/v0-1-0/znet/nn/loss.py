import numpy as np
from ..autograd import Tensor

class CrossEntropyLoss:
    def __call__(self, logits: Tensor, targets: Tensor):
        # shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
        # exp = np.exp(shifted)
        # probs = exp / np.sum(exp, axis=1, keepdims=True)
        probs = self.softmax(logits.data)
        
        batch_size = logits.data.shape[0]
        correct_log_probs = -np.log(probs[np.arange(batch_size), targets.data])
        loss_data = np.mean(correct_log_probs)

        loss = Tensor(loss_data, requires_grad=logits.requires_grad)

        def _grad_fn(grad_output):
            # grad_output is scalar (∂L/∂loss), usually 1.0
            grad = probs.copy()
            grad[np.arange(batch_size), targets.data] -= 1
            grad /= batch_size
            return grad_output * grad  # Chain rule

        loss._backward = lambda: None
        loss._prev = [logits]
        loss._grad_fn = _grad_fn
        return loss


    def softmax(self,x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
# class CrossEntropyLoss:
#     def __call__(self, logits: Tensor, targets: np.ndarray):
#         # Softmax + negative log likelihood
#         shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
#         exp = np.exp(shifted)
#         probs = exp / np.sum(exp, axis=1, keepdims=True)

#         self.batch_size = logits.data.shape[0]
#         self.probs = probs
#         self.targets = targets
#         self.logits = logits  # Keep logits Tensor for backprop
#         correct_log_probs = -np.log(probs[np.arange(self.batch_size), targets])
#         loss_value = np.mean(correct_log_probs)

#         loss = Tensor(np.array(loss_value), requires_grad=True)

#         def _grad_fn(_):
#             grad = self.probs.copy()
#             grad[np.arange(self.batch_size), targets] -= 1
#             grad /= self.batch_size
#             return grad

#         loss.set_backward(_grad_fn, [self.logits])
#         return loss


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