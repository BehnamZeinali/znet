class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for p in self.parameters:
            p['value'] -= self.lr * p['grad']

    def zero_grad(self):
        for p in self.parameters:
            p['grad'].fill(0)
