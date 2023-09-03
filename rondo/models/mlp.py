import rondo
import rondo.functions as F
import rondo.layers as L

class MLP(rondo.Model):
    """Multi-Layer Perceptron"""
    def __init__(self, sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, size in enumerate(sizes):
            layer = L.Linear(size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)
