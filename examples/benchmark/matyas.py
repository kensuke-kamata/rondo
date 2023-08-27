import numpy
import rondo

def matyas(x0, x1):
    y = 0.26 * (x0 ** 2 + x1 ** 2) - 0.48 * x0 * x1
    return y

a = rondo.Variable(numpy.array(1.0))
b = rondo.Variable(numpy.array(1.0))
y = matyas(a, b)
y.backward()
print(a.grad, b.grad)
