# znet/nn/module.py
from ..autograd.tensor import Tensor

class Module:
    def __init__(self):
        self._parameters = []      # list[Tensor]
        self._submodules = {}      # name -> Module

    def __setattr__(self, name, value):
        # preserve your submodule auto-registration
        if isinstance(value, Module):
            # replace any previous submodule of same name
            try:
                self._submodules[name] = value
            except AttributeError:
                # during __init__, _submodules may not exist yet
                object.__setattr__(self, "_submodules", {name: value})
        object.__setattr__(self, name, value)

    # lightweight helper to register a parameter explicitly
    def add_parameter(self, name: str, param: Tensor | None):
        if param is None:
            setattr(self, name, None)
            return
        if not isinstance(param, Tensor):
            raise TypeError("add_parameter expects a Tensor or None")
        setattr(self, name, param)
        # store the Tensor itself (not dicts); avoid duplicates
        if all(id(param) != id(p) for p in self._parameters):
            self._parameters.append(param)

    def parameters(self):
        # return unique Tensors from this module + all submodules
        seen = set()
        out = []
        for p in self._parameters:
            if isinstance(p, Tensor) and id(p) not in seen:
                out.append(p); seen.add(id(p))
        for sm in self._submodules.values():
            for p in sm.parameters():
                if id(p) not in seen:
                    out.append(p); seen.add(id(p))
        return out

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()   # Tensor API; backend-agnostic

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)