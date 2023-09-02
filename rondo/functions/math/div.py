import rondo
from rondo.function import Function

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1

def div(x0, x1):
    x1 = rondo.as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = rondo.as_array(x1)
    return Div()(x1, x0)
