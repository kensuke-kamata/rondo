import rondo

class RNN(rondo.Layer):
    def __init__(self, size_hidden, size_in=None):
        super().__init__()
        self.x2h = rondo.layers.Linear(size_hidden, size_in=size_in)
        self.h2h = rondo.layers.Linear(size_hidden, size_in=size_in, nobias=True)
        self.h = None

    def reset(self):
        self.h = None

    def forward(self, x):
        if self.h is None:
            h_new = rondo.functions.tanh(self.x2h(x))
        else:
            h_new = rondo.functions.tanh(self.x2h(x) + self.h2h(self.h))

        self.h = h_new
        return h_new
