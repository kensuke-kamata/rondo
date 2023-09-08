import gzip
import numpy as np
import matplotlib.pyplot as plt

import rondo
import rondo.transforms as T
import rondo.utils as U

class MNIST(rondo.Dataset):
    def __init__(self, train=True,
                 transform_x=T.Compose([T.Flatten(), T.ToFloat(), T.Normalize(0., 255.)]),
                 transform_t=None):
        self.url = 'http://yann.lecun.com/exdb/mnist/'
        self.train_files = ['train-images-idx3-ubyte.gz',
                            'train-labels-idx1-ubyte.gz']
        self.test_files = ['t10k-images-idx3-ubyte.gz',
                           't10k-labels-idx1-ubyte.gz']
        super().__init__(train, transform_x, transform_t)

    def prepare(self):
        files = self.train_files if self.train else self.test_files
        dpath = U.download(self.url + files[0])
        lpath = U.download(self.url + files[1])

        self.data  = self.load(dpath)
        self.label = self.load(lpath, label=True)

    def load(self, path, label=False):
        offset = 8 if label else 16
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=offset)
        return data if label else data.reshape(-1, 1, 28, 28)

    def show(self, row=10, col=10):
        H, W = 28, 28
        img = np.zeros((H * row, W * col))
        for r in range(row):
            for c in range(col):
                img[r * H:(r + 1) * H, c * W:(c + 1) * W] = self.data[np.random.randint(0, len(self.data) - 1)].reshape(H, W)
        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.show()
