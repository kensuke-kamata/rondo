import numpy

import rondo
from rondo.function import Function

class Transpose(Function):
    def forward(self, x):
        y = numpy.transpose(x)
        return y

    def backward(self, gy):
        gx = transpose(gy)
        return gx

def transpose(x):
    return Transpose()(x)
