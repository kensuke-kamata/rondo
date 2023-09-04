import numpy as np

import rondo

class Softmax(rondo.Function):
    def __init__(self, axis=1):
        # Set axis=1 as the default allows computing the softmax over batches of data e.g. [batch_size, num_features]
        self.axis = axis

    def forward(self, x):
        # Subtract the maximum value from each element to avoid overflow.
        y = x - np.max(x, axis=self.axis, keepdims=True)
        y = np.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        # TODO: Reconfirm this implementation.
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx

def softmax(x, axis=1):
    return Softmax(axis)(x)
