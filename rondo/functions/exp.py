import numpy

import rondo
from rondo.function import Function

class Exp(Function):
    def forward(self, x):
        return numpy.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = numpy.exp(x) * gy
        return gx
