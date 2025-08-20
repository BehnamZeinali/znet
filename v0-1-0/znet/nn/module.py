class Module:
    def __init__(self):
        self._parameters = []
        self._submodules = {}

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._submodules[name] = value
        object.__setattr__(self, name, value)

    def parameters(self):
        params = list(self._parameters)
        for sm in self._submodules.values():
            params.extend(sm.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p['grad'].fill(0)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
