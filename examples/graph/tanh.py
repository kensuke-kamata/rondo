import numpy as np
import matplotlib.pyplot as plt

import rondo
import rondo.functions as F
from rondo.utils import plot_dot_graph

x = rondo.Variable(np.array(1.0))
y = F.tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(create_graph=True)

iters = 6
for i in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

# Draw a graph
gx = x.grad
gx.name = 'gx' + str(iters + 1)
plot_dot_graph(gx, verbose=False, to_file='tanh.png')
