import rondo

class Accuracy(rondo.Function):
    def forward(self, y, t):
        pred = y.argmax(axis=1).reshape(t.shape)
        result = (pred == t)
        acc = result.mean()
        return acc

def accuracy(y, t):
    return Accuracy()(y, t)
