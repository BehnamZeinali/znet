# znet/nn/sequential.py
from .module import Module

class Sequential(Module):
    """
    A container that applies submodules in order, like PyTorch's nn.Sequential.

    Usage:
        nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    Notes:
      - Registers children so .parameters(), .zero_grad(), .train()/.eval() recurse.
      - Forward will pass the output of module i as input to module i+1.
      - If a module returns a tuple, it is passed positionally to the next module.
    """
    def __init__(self, *modules):
        super().__init__()
        self._list = []

        # support Sequential([m1, m2, ...]) or Sequential(m1, m2, ...)
        if len(modules) == 1 and isinstance(modules[0], (list, tuple)):
            self.extend(modules[0])
        else:
            self.extend(modules)

    # ---------- list-like API ----------
    def __len__(self):
        return len(self._list)

    def __iter__(self):
        return iter(self._list)

    def __getitem__(self, idx):
        return self._list[idx]

    def __setitem__(self, idx, module):
        if not isinstance(module, Module):
            raise TypeError("Sequential only accepts Module instances.")
        self._list[idx] = module
        self._reindex()

    def append(self, module):
        if not isinstance(module, Module):
            raise TypeError("Sequential only accepts Module instances.")
        self._list.append(module)
        self._reindex()

    def extend(self, modules):
        for m in modules:
            self.append(m)

    # ---------- forward ----------
    def forward(self, *args, **kwargs):
        curr_args = args
        curr_kwargs = kwargs
        for m in self._list:
            out = m(*curr_args, **curr_kwargs)
            # Next module gets only positional outputs by default
            if isinstance(out, tuple):
                curr_args, curr_kwargs = out, {}
            else:
                curr_args, curr_kwargs = (out,), {}
        # unwrap single return
        return curr_args[0] if len(curr_args) == 1 else curr_args

    # ---------- helpers ----------
    def _reindex(self):
        # keep children registered so .parameters() and .train() recurse
        self._submodules.clear()
        for i, m in enumerate(self._list):
            self._submodules[str(i)] = m

    def __repr__(self):
        items = ",\n  ".join(repr(m) for m in self._list)
        return f"Sequential([\n  {items}\n])"
