import heapq
import numpy

import rondo

class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, numpy.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'Variable(None)'
        p = str(self.data).replace('\n', '\n' +  ' ' * 9)
        return 'Variable(' + p + ')'

    def __add__(self, other):
        from rondo.functions import add
        return add(self, other)

    def __radd__(self, other):
        from rondo.functions import add
        return add(self, other)

    def __mul__(self, other):
        from rondo.functions import mul
        return mul(self, other)

    def __rmul__(self, other):
        from rondo.functions import mul
        return mul(self, other)

    def __neg__(self):
        from rondo.functions import neg
        return neg(self)

    def __sub__(self, other):
        from rondo.functions import sub
        return sub(self, other)

    def __rsub__(self, other):
        from rondo.functions import rsub
        return rsub(self, other)

    def __truediv__(self, other):
        from rondo.functions import div
        return div(self, other)

    def __rtruediv__(self, other):
        from rondo.functions import rdiv
        return rdiv(self, other)

    def __pow__(self, other):
        from rondo.functions import pow
        return pow(self, other)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        return rondo.functions.transpose(self)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return rondo.functions.reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return rondo.functions.transpose(self, axes)

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            self.grad = Variable(numpy.ones_like(self.data))

        # Initialize the priority queue with the creator of the current Variable.
        # We use negative generation values because heapq pops the smallest value first,
        # so by storing negative values, we can retrieve the function with the largest generation first.
        funcs = [(-self.creator.generation, self.creator)]
        seen_set = set()
        seen_set.add(self.creator)

        while funcs:
            _, f = heapq.heappop(funcs)

            # Retrieve the gradients from the function's outputs.
            # Since the references to outputs are from the Function are weak,
            # we need to dereference them (using output()) to access the actual Variable objects and their gradients.
            gys = [output().grad for output in f.outputs]
            with rondo.using('enable_backprop', create_graph):
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

            # Clear the gradient data from the Function's outputs to optimize memory usage.
            # This reduces the  memory footprint when not required to retain intermediate gradient data.
            # However, if the user specifies `retain_grad=True`, the gradients will be retained.
            if not retain_grad:
                for output in f.outputs:
                    output().grad = None

    def cleargrad(self):
        self.grad = None

def as_array(x):
    if numpy.isscalar(x):
        return numpy.array(x)
    return x

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)
