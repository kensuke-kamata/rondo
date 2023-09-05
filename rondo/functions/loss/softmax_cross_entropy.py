import numpy as np
import rondo

class SoftmaxCrossEntropy(rondo.Function):
    def forward(self, x, t):
        # Number of samples
        N = x.shape[0]

        # Subtract max score to avoid overflow
        m = x.max(axis=1, keepdims=True)
        z = x - m

        # Exponentiate adjusted logits
        np.exp(z, out=z)

        # Compute sum of exponentials for normalization and its logarithm
        s = z.sum(axis=1, keepdims=True)
        np.log(s, out=s)
        m += s

        # Compute adjusted logits for the softmax values
        p = x - m
        p = p[np.arange(N), t.ravel()]

        # Compute average cross-enteropy loss
        y = -p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        # The gradient of the loss with respect to the input x is:
        # ∂L/∂x = y - t
        # where y is the output of the softmax function and
        # t is the one-hot encoded target vector.
        y = rondo.functions.softmax(x)
        t_onehot = np.eye(CLS_NUM, dtype=t.dtype)[t.data]
        gy *= 1/N
        gx = (y - t_onehot) * gy
        return gx

def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)
