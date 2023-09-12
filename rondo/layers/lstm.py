import rondo
import rondo.functions as F

class LSTM(rondo.Layer):
    def __init__(self, size_hidden, size_in=None):
        super().__init__()

        H, I = size_hidden, size_in
        self.x2f = rondo.layers.Linear(H, size_in=I)
        self.x2i = rondo.layers.Linear(H, size_in=I)
        self.x2o = rondo.layers.Linear(H, size_in=I)
        self.x2u = rondo.layers.Linear(H, size_in=I)
        self.h2f = rondo.layers.Linear(H, size_in=H, nobias=True)
        self.h2i = rondo.layers.Linear(H, size_in=H, nobias=True)
        self.h2o = rondo.layers.Linear(H, size_in=H, nobias=True)
        self.h2u = rondo.layers.Linear(H, size_in=H, nobias=True)
        self.reset()

    def reset(self):
        self.h = None
        self.c = None

    def forward(self, x):
        if self.h is None:
            f = F.sigmoid(self.x2f(x))
            i = F.sigmoid(self.x2i(x))
            o = F.sigmoid(self.x2o(x))
            u = F.tanh(self.x2u(x))
        else:
            f = F.sigmoid(self.x2f(x) + self.h2f(self.h))
            i = F.sigmoid(self.x2i(x) + self.h2i(self.h))
            o = F.sigmoid(self.x2o(x) + self.h2o(self.h))
            u = F.tanh(self.x2u(x) + self.h2u(self.h))

        if self.c is None:
            c_new = (i * u)
        else:
            c_new = (f * self.c) + (i * u)

        h_new = o * F.tanh(c_new)
        self.h = h_new
        self.c = c_new

        return h_new
