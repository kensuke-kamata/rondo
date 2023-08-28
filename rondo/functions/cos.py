import numpy
import math

import rondo
from rondo.function import Function
from rondo.functions import sin

class Cos(Function):
    def forward(self, x):
        y = numpy.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

def cos(x):
    return Cos()(x)
