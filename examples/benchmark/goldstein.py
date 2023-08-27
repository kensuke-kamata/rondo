import numpy
import rondo

def goldstein(x0, x1):
    y = (1 + (x0 + x1 + 1)**2 * (19 - 14*x0 + 3*x0**2 - 14*x1 + 6*x0*x1 + 3*x1**2)) * \
        (30 + (2*x0 - 3*x1)**2 * (18 - 32*x0 + 12*x0**2 + 48*x1 - 36*x0*x1 + 27*x1**2))
    return y

a = rondo.Variable(numpy.array(1.0))
b = rondo.Variable(numpy.array(1.0))
y = goldstein(a, b)
y.backward()
print(a.grad, b.grad)
