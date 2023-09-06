import numpy as np

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        mean, std = self.mean, self.std

        if not np.isscalar(mean):
            mshape = [1] * sample.ndim
            mshape[0] = len(sample) if len(mean) == 1 else len(mean)
            mean = np.array(mean, dtype=sample.dtype).reshape(*mshape)
        if not np.isscalar(std):
            rshape = [1] * sample.ndim
            rshape[0] = len(sample) if len(std) == 1 else len(std)
            std = np.array(std, dtype=sample.dtype).reshape(*rshape)
        return (sample - mean) / std
