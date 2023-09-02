import numpy

import rondo
from rondo.function import Function

class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)
        axes_len = len(self.axes)
        # Compute the inverse permutation:
        #   Use numpy.argsort to find the indices that would sort the adjusted axes.
        #   This gives the inverse permutation required to
        #   reverse the transpose operation performed in the forward pass.
        inv_axes = tuple(numpy.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)

def transpose(x, axes=None):
    return Transpose(axes)(x)
