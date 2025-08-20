import numpy as np
from znet._grad_mode import is_grad_enabled
class Tensor:
    def __init__(self, data, requires_grad=False ,  dtype=np.float32):
        self.data = np.array(data, dtype=dtype)
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self.requires_grad = requires_grad and is_grad_enabled()

        self._backward = lambda: None
        self._prev = []
        self._grad_fn = None

    def backward(self, grad=None):
        # Base case: scalar loss
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be specified for non-scalar Tensor")
            grad = np.ones_like(self.data)

        self.grad = grad

        # Run local gradient logic
        self._backward()

        # Now recursively propagate to parents
        if self._grad_fn is not None:
            grads_to_parents = self._grad_fn(self.grad)
            if not isinstance(grads_to_parents, (list, tuple)):
                grads_to_parents = [grads_to_parents]
            for parent, g in zip(self._prev, grads_to_parents):
                if parent.requires_grad:
                    parent.backward(g)

    def requires_grad_(self, mode: bool = True):
        self.requires_grad = mode
        return self
    
    def detach(self):
        # Return a new tensor with same data, no grad tracking
        detached = Tensor(self.data, requires_grad=False)
        return detached
    def item(self):
        if self.data.size != 1:
            raise ValueError("Can only convert a Tensor with one element to a Python scalar")
        return self.data.item()
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def size(self):
        return self.data.shape
    
    def reshape(self, new_shape):
        reshaped_data = self.data.reshape(new_shape)
        out = Tensor(reshaped_data, requires_grad=self.requires_grad)

        def _grad_fn(grad_output):
            # Reshape gradient back to original shape
            return grad_output.reshape(self.data.shape)

        out._grad_fn = _grad_fn
        out._prev = [self]
        return out
    
    @property
    def T(self):
        transposed_data = self.data.T
        out = Tensor(transposed_data, requires_grad=self.requires_grad)

        def _grad_fn(grad_output):
            return grad_output.T  # just transpose back

        out._grad_fn = _grad_fn
        out._prev = [self]
        return out
    
    def flatten(self):
        original_shape = self.data.shape
        flattened_data = self.data.flatten()
        out = Tensor(flattened_data, requires_grad=self.requires_grad)

        def _grad_fn(grad_output):
            return grad_output.reshape(original_shape)

        out._grad_fn = _grad_fn
        out._prev = [self]
        return out
    
    def sum(self, axis=None, keepdims=False):
        original_shape = self.data.shape
        out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad)

        def _grad_fn(grad_output):
            grad = grad_output
            if not keepdims and axis is not None:
                grad = np.expand_dims(grad_output, axis)

            return np.ones_like(self.data) * grad  # broadcast over original shape

        out._grad_fn = _grad_fn
        out._prev = [self]
        return out
