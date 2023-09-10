import numpy as np
import rondo
import rondo.functions as F
import rondo.utils as U

class Conv2d(rondo.Function):
    def __init__(self, stride=1, pad=0):
        super().__init__()
        self.stride = U.pair(stride)
        self.pad = U.pair(pad)

    def forward(self, x, W, b):
        KH, KW = W.shape[2:]
        col = U.im2col(x, (KH, KW), self.stride, self.pad, to_matrix=False)

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
        gx = F.deconv2d(gy, W, b=None, stride=self.stride, pad=self.pad, outsize=(x.shape[2], x.shape[3]))

        # gW
        gW = Conv2dGradW(self)(x, gy)

        # gb
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))

        return  gx, gW, gb

def conv2d(x, W, b=None, stride=1, pad=0):
    return Conv2d(stride, pad)(x, W, b)

class Conv2dGradW(rondo.Function):
    def __init__(self, conv2d):
        super().__init__()
        W = conv2d.inputs[1]
        kh, kw = W.shape[2:]
        self.kernel = (kh, kw)
        self.stride = conv2d.stride
        self.pad = conv2d.pad

    def forward(self, x, gy):
        col = U.im2col(x, self.kernel, self.stride, self.pad, to_matrix=False)
        gW = np.tensordot(gy, col, ((0, 2, 3), (0, 4, 5)))
        return gW

    def backward(self, gys):
        x, gy = self.inputs
        gW, = self.outputs

        xh, xw = x.shape[2:]
        gx = F.deconv2d(gy, gW, stride=self.stride, pad=self.pad, outsize=(xh, xw))
        ggy = F.conv2d(x, gW, stride=self.stride, pad=self.pad)
        return gx, ggy
