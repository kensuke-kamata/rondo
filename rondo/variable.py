import heapq
import numpy

import rondo

class Variable:
    def __init__(self, data) -> None:
        if data is not None:
            if not isinstance(data, numpy.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self):
        if self.grad is None:
            self.grad = numpy.ones_like(self.data)

        # Initialize the priority queue with the creator of the current Variable.
        # We use negative generation values because heapq pops the smallest value first,
        # so by storing negative values, we can retrieve the function with the largest generation first.
        funcs = [(-self.creator.generation, self.creator)]
        seen_set = set()
        seen_set.add(self.creator)

        while funcs:
            _, f = heapq.heappop(funcs)

            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for input, gx in zip(f.inputs, gxs):
                if input.grad is None:
                    input.grad = gx
                else:
                    input.grad = input.grad + gx
                if input.creator is not None and input.creator not in seen_set:
                    heapq.heappush(funcs, (-input.creator.generation, input.creator))
                    seen_set.add(input.creator)

    def cleargrad(self):
        self.grad = None
