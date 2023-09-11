import numpy as np
import rondo
import rondo.utils as U

class PoolingMax2d(rondo.Function):
    def __init__(self, kernel, stride=1, pad=0):
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.indexes = None

    def forward(self, x):
        col = U.im2col(x, self.kernel, self.stride, self.pad, to_matrix=False)
        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        self.indexes = col.argmax(axis=2)
        y = col.max(axis=2)
        return y

    def backward(self, gy):
        # NOTE: Should be separated into another class with forward and backwardI
        x = self.inputs[0]
        gy = gy.data

        N, C, OH, OW = gy.shape
        N, C, H, W = x.shape
        KH, KW = U.pair(self.kernel)

        gcol = np.zeros((N * C * OH * OW * KH * KW), dtype=x.dtype)
        indexes = (self.indexes.ravel() + np.arange(0, self.indexes.size * KH * KW, KH * KW))
        gcol[indexes] = gy.ravel()
        gcol = gcol.reshape(N, C, OH, OW, KH, KW)
        gcol = np.swapaxes(gcol, 2, 4)
        gcol = np.swapaxes(gcol, 3, 5)
        gx = U.col2im(gcol, (N, C, H, W), self.kernel, self.stride, self.pad, to_matrix=False)
        return rondo.as_variable(gx)

def pooling_max_2d(x, kernel, stride=1, pad=0):
    return PoolingMax2d(kernel, stride, pad)(x)
