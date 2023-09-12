import rondo
import rondo.layers as L

class LSTM(rondo.Model):
    def __init__(self, size_hidden, size_out):
        super().__init__()
        self.lstm = L.LSTM(size_hidden)
        self.fc = L.Linear(size_out)

    def reset(self):
        self.lstm.reset()

    def forward(self, x):
        h = self.lstm(x)
        y = self.fc(h)
        return y
