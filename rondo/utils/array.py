import numpy

def as_array(x):
    if numpy.isscalar(x):
        return numpy.array(x)
    return x

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
