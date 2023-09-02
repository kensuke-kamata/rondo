import rondo
from rondo.function import Function

class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = rondo.functions.sum_to(gx0, self.x0_shape)
            gx1 = rondo.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

def add(x0, x1):
    x1 = rondo.as_array(x1)
    return Add()(x0, x1)
