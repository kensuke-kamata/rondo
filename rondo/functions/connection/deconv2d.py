import numpy as np
import rondo
import rondo.functions as F
import rondo.utils as U

class Deconv2d(rondo.Function):
    def __init__(self, stride=1, pad=0, outsize=None):
        super().__init__()
        self.stride = U.pair(stride)
        self.pad = U.pair(pad)
        self.outsize = outsize

    def forward(self, x, W, b):
        N, C, IH, IW = x.shape
        C, OC, KH, KW = W.shape
        SH, SW = self.stride
        PH, PW = self.pad

        if self.outsize is None:
            out_h = U.get_deconv_outsize(IH, KH, SH, PH)
            out_w = U.get_deconv_outsize(IW, KW, SW, PW)
        else:
            out_h, out_w = U.pair(self.outsize)
        img_shape = (N, OC, out_h, out_w)

        gcol = np.tensordot(W, x, (0, 1))
        gcol = np.rollaxis(gcol, 3)
        y = U.col2im(gcol, img_shape, (KH, KW), self.stride, self.pad, to_matrix=False)

        if b is not None:
            y += b.reshape((1, b.size, 1, 1))

        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gx = F.conv2d(gy, W, b=None, stride=self.stride, pad=self.pad)
        gW = F.conv2d_gradW(self, gy, x) # NOTE: Check if the order of parameters is correct
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb

def deconv2d(x, W, b=None, stride=1, pad=0, outsize=None):
    return Deconv2d(stride, pad, outsize)(x, W, b)
