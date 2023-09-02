import numpy

import rondo

def as_vec(x):
    if not isinstance(x, numpy.ndarray):
        raise TypeError('{} is not supported'.format(type(x)))
    if x.ndim == 1:
        return x
    return x.ravel()

def as_mat(x):
    if not isinstance(x, numpy.ndarray):
        raise TypeError('{} is not supported'.format(type(x)))
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)

def sum_to(x, shape):
    if x.shape == shape:
        return x
    if isinstance(x, rondo.Variable):
        raise TypeError(
            'rondo.utils.sum_to does not support Variable input.'
            'Use rondo.functions.sum_to instead.')

    # Calculate the difference in dimensions between x and target shape
    ndim = len(shape)
    lead = x.ndim - ndim

    # Create a tuple of the leading axes that are extra in 'x' and need to be reduced.
    # which will be the axes in the range [0, lead)
    lead_axis = tuple(range(lead))

    # Identify the axes where the target shape has dimension size 1.
    # Dimensions with size 1 in the target shape are the ones we need to reduce in the tensor x
    # by summing over them.
    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)

    # If there were any leading dimensions in x not present in target shape, remove them.
    if lead > 0:
        y = y.squeeze(lead_axis)

    return y
