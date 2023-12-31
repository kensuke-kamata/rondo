import numpy

import rondo
from rondo.function import Function

class Exp(Function):
    def forward(self, x):
        return numpy.exp(x)

    def backward(self, gy):
        x = self.inputs[0].data
        gx = numpy.exp(x) * gy
        return gx

def exp(x):
    return Exp()(x)
