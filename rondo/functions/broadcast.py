import numpy

import rondo
from rondo.function import Function

class Broadcast(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = numpy.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = rondo.functions.sum_to(gy, self.x_shape)
        return gx

def broadcast_to(x, shape):
    if x.shape == shape:
        return rondo.as_variable(x)
    return Broadcast(shape)(x)
