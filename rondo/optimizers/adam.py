import math

import numpy as np

import rondo

class Adam(rondo.Optimizer):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.t = 0
        self.alpha = alpha
        self.eps = eps

        # Exposure decay rates for the moment estimates
        self.beta1 = beta1
        self.beta2 = beta2

        self.ms = {} # First moment estimate
        self.vs = {} # Second moment estimate

    def update(self):
        # Increment the timestep
        self.t += 1
        super().update()

    @property
    def lr(self):
        fix1 = 1. - math.pow(self.beta1, self.t)
        fix2 = 1. - math.pow(self.beta2, self.t)
        return self.alpha * math.sqrt(fix2) / fix1

    def update_param(self, param):
        if param.grad is None:
            return
        grad = param.grad.data

        key = id(param)
        if key not in self.ms:
            self.ms[key] = np.zeros_like(param.data)
            self.vs[key] = np.zeros_like(param.data)

        m, v = self.ms[key], self.vs[key]
        m += (1 - self.beta1) * (grad - m)
        v += (1 - self.beta2) * (grad * grad - v)
        param.data -= self.lr * m / (np.sqrt(v) + self.eps)
