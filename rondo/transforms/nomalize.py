import numpy as np

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, array):
        mean, std = self.mean, self.std

        if not np.isscalar(mean):
            mshape = [1] * array.ndim
            mshape[0] = len(array) if len(mean) == 1 else len(mean)
            mean = np.array(mean, dtype=array.dtype).reshape(*mshape)
        if not np.isscalar(std):
            rshape = [1] * array.ndim
            rshape[0] = len(array) if len(std) == 1 else len(std)
            std = np.array(std, dtype=array.dtype).reshape(*rshape)
        return (array - mean) / std
