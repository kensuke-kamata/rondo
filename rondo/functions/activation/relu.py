import numpy as np

import rondo
from rondo.function import Function

class ReLU(Function):
    def forward(self, x):
        y = np.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x, = self.input
        mask = x.data > 0
        gx = gy * mask
        return gx

def relu(x):
    return ReLU()(x)