import rondo

class SGD(rondo.Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_param(self, param):
        param.data -= self.lr * param.grad.data
