import numpy as np

import rondo
from rondo.function import Function

class Sigmoid(Function):
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx

def sigmoid(x):
    return Sigmoid()(x)
