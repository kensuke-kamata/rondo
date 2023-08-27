import rondo
from rondo.variable import Variable

def numerical_diff(f, *inputs, eps=1e-4):
    grads = []
    inputs = list(inputs)
    for i, input in enumerate(inputs):
        tmp0 = inputs.copy()
        tmp1 = inputs.copy()
        x0 = Variable(rondo.as_array(input.data - eps))
        x1 = Variable(rondo.as_array(input.data + eps))
        tmp0[i] = x0
        tmp1[i] = x1
        y0 = f(*tmp0)
        y1 = f(*tmp1)
        grad = (y1.data - y0.data) / (2 * eps)
        grads.append(grad)
    return grads if len(grads) > 1 else grads[0]
