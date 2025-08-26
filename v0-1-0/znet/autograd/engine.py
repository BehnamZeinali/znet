# znet/autograd/engine.py
import mlx.core as mx

def _accumulate(grads, t, g):
    if t not in grads:
        grads[t] = g
    else:
        grads[t] = grads[t] + g

class Context:
    __slots__ = ("saved_tensors", "meta")
    def __init__(self):
        self.saved_tensors = ()
        self.meta = {}

    def save_for_backward(self, *arrs):
        self.saved_tensors = tuple(mx.array(a) for a in arrs)

class Function:
    """
    Subclass pattern:
      @staticmethod
      def forward(ctx, *raw_inputs) -> raw_output(s)
      def backward(self, *grad_outputs) -> grad_inputs (aligned with TENSOR parents)
    """
    parents = ()
    ctx = None

    @classmethod
    def apply(cls, *inputs):
        # Build context
        ctx = Context()

        # Determine requires_grad and collect TENSOR parents (only tensors!)
        tensor_parents = []
        requires_grad = False
        raw_inputs = []
        for x in inputs:
            if hasattr(x, "data"):  # a Tensor
                raw_inputs.append(x.data)
                tensor_parents.append(x)
                requires_grad = requires_grad or bool(getattr(x, "requires_grad", False))
            else:
                raw_inputs.append(x)

        # Forward to raw MLX outputs
        outputs = cls.forward(ctx, *raw_inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        # Wrap outputs as Tensors and attach grad_fn node if needed
        from .tensor import Tensor  # late import to avoid cycles
        out_tensors = []
        for out in outputs:
            t = Tensor(out, requires_grad=requires_grad)
            if requires_grad:
                node = cls.__new__(cls)
                node.ctx = ctx
                node.parents = tuple(tensor_parents)
                t.grad_fn = node
                # backwards-compat: some legacy code may look at _grad_fn
                t._grad_fn = node
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

def engine_backward(output_tensor, grad, *, retain_graph=False):
    topo = _collect_topo(output_tensor)
    grads = {output_tensor: mx.array(grad, dtype=output_tensor.data.dtype)}

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
                _accumulate(grads, parent, mx.array(gin, dtype=parent.data.dtype))

        if not retain_graph:
            node.ctx.saved_tensors = ()
            node.parents = ()
