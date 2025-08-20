_grad_enabled = True

def is_grad_enabled():
    return _grad_enabled

def set_grad_enabled(mode: bool):
    global _grad_enabled
    _grad_enabled = bool(mode)

class no_grad:
    def __enter__(self):
        global _grad_enabled
        self.prev = _grad_enabled
        _grad_enabled = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _grad_enabled
        _grad_enabled = self.prev