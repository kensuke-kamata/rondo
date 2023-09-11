import numpy as np

import rondo
import rondo.functions as F

class Linear(rondo.Layer):
    def __init__(self, size_out, size_in=None, nobias=False, dtype=np.float32):
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out
        self.dtype = dtype

        self.W = rondo.Parameter(None, name='W')
        if self.size_in is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = rondo.Parameter(np.zeros(size_out, dtype=dtype), name='b')

    def _init_W(self):
        I, O = self.size_in, self.size_out
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.size_in = x.shape[1]
            self._init_W()

        y = F.linear(x, self.W, self.b)
        return y
