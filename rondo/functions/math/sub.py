import rondo
from rondo.function import Function

class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy, -gy

def sub(x0, x1):
    x1 = rondo.as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = rondo.as_array(x1)
    return Sub()(x1, x0)
