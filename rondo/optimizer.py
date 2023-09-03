class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self

    def update(self):
        # Get parameters that have gradients
        params = [p for p in self.target.params() if p.grad is not None]

        # Preprocessing
        for f in self.hooks:
            f(params)

        # Update parameters
        for param in params:
            self.update_param(param)

    def update_param(self, param):
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)
