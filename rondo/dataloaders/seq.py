import numpy as np
import rondo

class SeqDataLoader(rondo.DataLoader):
    def __init__(self, dataset, batchsize):
        super().__init__(dataset, batchsize, shuffle=False)

    def __next__(self):
        if self.iteration >= self.maxiter:
            self.reset()
            raise StopIteration

        jump = self.datasize // self.batchsize
        batchindex = [(i * jump + self.iteration) % self.datasize for i in range(self.batchsize)]
        batch = [self.dataset[i] for i in batchindex]

        x = np.array([example[0] for example in batch])
        t = np.array([example[1] for example in batch])

        self.iteration += 1
        return x, t
