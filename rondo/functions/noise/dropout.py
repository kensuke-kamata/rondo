import numpy as np

import rondo

class Dropout(rondo.Function):
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x):
        if rondo.Config.train:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            scale = np.array(1. - self.dropout_ratio).astype(x.dtype)
            y = x * self.mask / scale
            return y
        else:
            return x

    def backward(self, gy):
        if rondo.Config.train:
            gx = gy * self.mask
            return gx
        else:
            return gy

def dropout(x, dropout_ratio=0.5):
    return Dropout(dropout_ratio)(x)
