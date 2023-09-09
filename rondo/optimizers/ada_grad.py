import numpy as np

import rondo

class AdaGrad(rondo.Optimizer):
    def __init__(self, lr=0.01, eps=1e-8):
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.hs = {}

    def update_param(self, param):
        grad = param.grad.data
        if grad is None:
            return

        h_key = id(param)
        if h_key not in self.hs:
            self.hs[h_key] = np.zeros_like(param.data)

        h = self.hs[h_key]
        h += grad * grad

        param.data -= self.lr * grad / (np.sqrt(h) + self.eps)
