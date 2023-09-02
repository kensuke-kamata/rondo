import rondo
from rondo.function import Function

class Sum(Function):
    def __init__(self, axis, keepdims) -> None:
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        # If keepdims is False and axis is provided, reshape gy
        if not (self.keepdims or self.axis is None):
            ndim = len(self.x_shape)
            # Convert negative axis value to positive
            # e.g. for ndim=3 `axis=(-2, -1)` -> `actual_axis=[1, 2]`
            if isinstance(self.axis, int):
                actual_axis = [self.axis] if self.axis >= 0 else [self.axis + ndim]
            else:
                actual_axis = [axis if axis >= 0 else axis + ndim for axis in self.axis]

            # Insert singleton dimensions at the specified axes
            # e.g. `gy.shape = (3,)` -> `(3, 1, 1)` for axes (1, 2)
            shape = list(gy.shape)
            for axis in sorted(actual_axis):
                shape.insert(axis, 1)
            gy = rondo.functions.reshape(gy, shape)

        gx = rondo.functions.broadcast_to(gy, self.x_shape)
        return gx

def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = rondo.utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = rondo.functions.broadcast_to(gy, self.x_shape)
        return gx

def sum_to(x, shape):
    if x.shape == shape:
        return rondo.as_variable(x)
    return SumTo(shape)(x)
