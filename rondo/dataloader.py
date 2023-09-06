import math
import numpy as np

class DataLoader:
    def __init__(self, dataset, batchsize, shuffule=True):
        self.dataset = dataset
        self.batchsize = batchsize
        self.shuffle = shuffule
        self.datasize = len(dataset)
        self.maxiter = math.ceil(self.datasize / batchsize)

        self.reset()

    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(self.datasize)
        else:
            self.index = np.arange(self.datasize)

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.maxiter:
            self.reset()
            raise StopIteration

        i, batchsize = self.iteration, self.batchsize
        batchindex = self.index[i * batchsize:(i + 1) * batchsize]
        batch = [self.dataset[i] for i in batchindex]

        x = np.array([example[0] for example in batch])
        t = np.array([example[1] for example in batch])

        self.iteration += 1
        return x, t

    def next(self):
        return self.__next__()
