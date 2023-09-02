import rondo
from rondo.function import Function

class Linear(Function):
    def forward(self, x, W, b=None):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else rondo.functions.sum_to(gy, b.shape)
        gx = rondo.functions.matmul(gy, W.T)
        gW = rondo.functions.matmul(x.T, gy)
        return gx, gW, gb

def linear(x, W, b=None):
    return Linear()(x, W, b)
