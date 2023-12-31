import rondo
from rondo.function import Function

class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x, = self.inputs
        gx = 2 * x * gy
        return gx

def square(x):
    return Square()(x)
