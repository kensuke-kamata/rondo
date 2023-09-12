import rondo
import rondo.layers as L

class RNN(rondo.Model):
    def __init__(self, size_hidden, size_out):
        super().__init__()
        self.rnn = L.RNN(size_hidden)
        self.fc = L.Linear(size_out)

    def reset(self):
        self.rnn.reset()

    def forward(self, x):
        h = self.rnn(x)
        y = self.fc(h)
        return y
