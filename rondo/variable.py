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

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = numpy.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for input, gx in zip(f.inputs, gxs):
                if input.grad is None:
                    input.grad = gx
                else:
                    input.grad = input.grad + gx
                if input.creator is not None:
                    funcs.append(input.creator)

    def cleargrad(self):
        self.grad = None
