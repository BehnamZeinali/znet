# znet/nn/modulelist.py
from .module import Module

class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        self._list = []
        if modules is not None:
            self.extend(modules)

    # --- list protocol ---
    def __len__(self):
        return len(self._list)

    def __iter__(self):
        return iter(self._list)

    def __getitem__(self, idx):
        return self._list[idx]

    def __setitem__(self, idx, module):
        if not isinstance(module, Module):
            raise TypeError("ModuleList only accepts Module instances.")
        # remove old registration (if replacing)
        old = self._list[idx]
        # find its key and delete (keys are str(index))
        if str(idx) in self._submodules:
            del self._submodules[str(idx)]
        # set new
        self._list[idx] = module
        self._submodules[str(idx)] = module  # register child

    # --- mutators ---
    def append(self, module):
        if not isinstance(module, Module):
            raise TypeError("ModuleList only accepts Module instances.")
        idx = len(self._list)
        self._list.append(module)
        self._submodules[str(idx)] = module  # register child

    def extend(self, modules):
        for m in modules:
            self.append(m)

    # optional: nice repr
    def __repr__(self):
        items = ",\n  ".join(repr(m) for m in self._list)
        return f"ModuleList([\n  {items}\n])"
