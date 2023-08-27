import numpy
import rondo

def sphere(*xs):
    y = sum(x ** 2 for x in xs)
    return y

a = rondo.Variable(numpy.array(1.0))
b = rondo.Variable(numpy.array(1.0))
c = rondo.Variable(numpy.array(1.0))
y = sphere(a, b, c)
y.backward()
print(a.grad, b.grad, c.grad)
