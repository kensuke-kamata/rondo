import numpy
import gc
from memory_profiler import profile

import rondo
import rondo.functions as F
from rondo.variable import Variable

@profile
def profile_memory_intensive_computation():
    for _ in range(10):
        x = Variable(numpy.random.randn(10000))
        y = F.square(F.square(F.square(x)))

if __name__ == '__main__':
    profile_memory_intensive_computation()
    print(f'Number of unreachable objects: {gc.collect()}')
