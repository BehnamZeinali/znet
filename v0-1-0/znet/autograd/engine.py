# znet/autograd/engine.py
from __future__ import annotations
import torch as th

def _accumulate(grads, t, g: th.Tensor):
    if t not in grads:
        grads[t] = g
    else:
        grads[t] = grads[t] + g

class Context:
    __slots__ = ("saved_tensors", "meta")
    def __init__(self):
        self.saved_tensors = ()
        self.meta = {}

    def save_for_backward(self, *objs):
        """
        Store tensors (as detached torch.Tensors) and keep non-tensors as-is
        (e.g., shapes/ints/tuples). If a znet Tensor is passed, store its .data.
        """
        saved = []
        from .tensor import Tensor as ZTensor
        for a in objs:
            if isinstance(a, ZTensor):
                saved.append(a.data)                                 # torch.Tensor
            elif isinstance(a, th.Tensor):
                saved.append(a.detach())                              # torch.Tensor (leaf, no grad_fn)
            else:
                saved.append(a)                                       # shape tuples, ints, etc.
        self.saved_tensors = tuple(saved)

class Function:
    """
    Subclass pattern:
      @staticmethod
      def forward(ctx, *raw_inputs) -> raw_output(s)   # raw_inputs are torch.Tensors or Python scalars
      def backward(self, *grad_outputs) -> grad_inputs # torch.Tensors aligned with tensor parents
    """
    parents = ()
    ctx = None

    @classmethod
    def apply(cls, *inputs):
        ctx = Context()

        # Collect raw inputs and tensor parents
        tensor_parents = []
        requires_grad = False
        raw_inputs = []
        for x in inputs:
            if hasattr(x, "data"):           # znet Tensor
                raw_inputs.append(x.data)    # torch.Tensor
                tensor_parents.append(x)
                requires_grad = requires_grad or bool(getattr(x, "requires_grad", False))
            else:
                raw_inputs.append(x)

        # Forward (should return torch tensor(s) or Python scalars/arrays)
        outputs = cls.forward(ctx, *raw_inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        # Default device/dtype for wrapping non-tensor outputs
        parent_dev  = tensor_parents[0].device if tensor_parents else None
        parent_dtype = tensor_parents[0].dtype if tensor_parents else None

        from .tensor import Tensor
        out_tensors = []
        for out in outputs:
            if not isinstance(out, th.Tensor):
                # Map to torch with best-effort device/dtype from first parent
                if parent_dev is not None and parent_dtype is not None:
                    out = th.as_tensor(out, dtype=parent_dtype, device=parent_dev)
                elif parent_dev is not None:
                    out = th.as_tensor(out, device=parent_dev)
                elif parent_dtype is not None:
                    out = th.as_tensor(out, dtype=parent_dtype)
                else:
                    out = th.as_tensor(out)

            t = Tensor(out, requires_grad=requires_grad)
            if requires_grad:
                node = cls.__new__(cls)
                node.ctx = ctx
                node.parents = tuple(tensor_parents)
                t.grad_fn = node
                t._grad_fn = node  # legacy alias
            out_tensors.append(t)

        return out_tensors[0] if len(out_tensors) == 1 else tuple(out_tensors)

def _collect_topo(output):
    visited, order = set(), []
    def visit(t):
        if id(t) in visited: return
        visited.add(id(t))
        node = t.grad_fn
        if node is not None:
            for p in node.parents:
                if getattr(p, "requires_grad", False):
                    visit(p)
        order.append(t)
    visit(output)
    return order

def _ensure_like(x, like: th.Tensor) -> th.Tensor:
    if isinstance(x, th.Tensor):
        return x.to(device=like.device, dtype=like.dtype)
    return th.as_tensor(x, dtype=like.dtype, device=like.device)

def engine_backward(output_tensor, grad, *, retain_graph=False):
    topo = _collect_topo(output_tensor)
    grads = {output_tensor: _ensure_like(grad, output_tensor.data)}

    for t in reversed(topo):
        g_out = grads.get(t)
        if g_out is None:
            continue

        if t.grad_fn is None:
            # leaf
            t.grad = g_out if t.grad is None else (t.grad + g_out)
            continue

        node = t.grad_fn
        in_grads = node.backward(g_out)
        if not isinstance(in_grads, (tuple, list)):
            in_grads = (in_grads,)

        for parent, gin in zip(node.parents, in_grads):
            if getattr(parent, "requires_grad", False) and gin is not None:
                _accumulate(grads, parent, _ensure_like(gin, parent.data))

        if not retain_graph:
            node.ctx.saved_tensors = ()
            node.parents = ()
