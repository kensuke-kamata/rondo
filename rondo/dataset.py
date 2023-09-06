import numpy as np

class Dataset:
    def __init__(self, train=True, transform_x=None, transform_t=None) -> None:
        self.train = train
        self.transform_x = transform_x
        self.transform_t = transform_t
        if self.transform_x is None:
            self.transform_x = lambda x: x
        if self.transform_t is None:
            self.transform_t = lambda x: x

        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, index):
        assert np.isscalar(index)
        if self.label is None:
            return self.transform_x(self.data[index]), None
        else:
            return self.transform_x(self.data[index]), self.transform_t(self.label[index])

    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass
