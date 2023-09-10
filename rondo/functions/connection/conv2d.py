import numpy as np
import rondo

class Conv2d(rondo.Function):
    def __init__(self, stride=1, pad=0):
        super().__init__()
        self.stride = rondo.utils.pair(stride)
        self.pad = rondo.utils.pair(pad)

    def forward(self, x, W, b):
        KH, KW = W.shape[2:]
        col = rondo.utils.im2col(x, (KH, KW), self.stride, self.pad, to_matrix=False)

        # (N, OH, OW, OC)
        y = np.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))
        if b is not None:
            y += b

        # (N, OC, OH, OW)
        y = np.rollaxis(y, 3, 1)
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        # gx
        # gW
        # gb
        pass

def conv2d(x, W, b=None, stride=1, pad=0):
    return Conv2d(stride, pad)(x, W, b)
