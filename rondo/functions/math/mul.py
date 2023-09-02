import rondo
from rondo.function import Function

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        return gy * x1, gy * x0

def mul(x0, x1):
    x1 = rondo.as_array(x1)
    return Mul()(x0, x1)
