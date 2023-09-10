import numpy as np
import rondo

class Conv2d(rondo.Layer):
    def __init__(self, channels_out, kernel, stride=1, pad=0, nobias=False, dtype=np.float32, channels_in=None):
        super().__init__()
        self.channels_in  = channels_in
        self.channels_out = channels_out
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = rondo.Parameter(None, name='W')
        if channels_in is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = rondo.Parameter(np.zeros(channels_out, dtype=dtype), name='b')

    def _init_W(self):
        C, OC = self.channels_in, self.channels_out
        KH, KW = rondo.utils.pair(self.kernel)
        scale = np.sqrt(1 / (C * KH * KW))
        data = np.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = data

    def forward(self, x):
        if self.W.data is None:
            self.channels_in = x.shape[1]
            self._init_W()

        y = rondo.functions.conv2d(x, self.W, self.b, self.stride, self.pad)
        return y
