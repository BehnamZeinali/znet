# e.g., in znet/autograd/ops_view.py (or a small znet/functional.py)

def cat(tensors, dim=0):
    from .autograd.ops_view import Cat
    return Cat.apply(*tensors, int(dim))


