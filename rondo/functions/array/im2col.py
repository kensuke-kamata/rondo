import rondo

class Im2Col(rondo.Function):
    def __init__(self, kernel, stride, pad, to_matrix):
        super().__init__()
        self.shape = None
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, img):
        self.shape = img.shape
        y = rondo.utils.im2col(img, self.kernel, self.stride, self.pad, self.to_matrix)
        return y

    def backward(self, gy):
        gx = rondo.utils.col2im(gy.data, self.shape, self.kernel, self.stride, self.pad, self.to_matrix)
        return gx

def im2col(img, kernel, stride=1, pad=0, to_matrix=True):
    return Im2Col(kernel, stride, pad, to_matrix)(img)

class Col2Im(rondo.Function):
    def __init__(self, shape, kernel, stride, pad, to_matrix):
        super().__init__()
        self.shape = shape
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, col):
        y = rondo.utils.col2im(col, self.shape, self.kernel, self.stride, self.pad, self.to_matrix)
        return y

    def backward(self, gy):
        gx = rondo.utils.im2col(gy.data, self.kernel, self.stride, self.pad, self.to_matrix)
        return gx

def col2im(col, shape, kernel, stride=1, pad=0, to_matrix=True):
    return Col2Im(shape, kernel, stride, pad, to_matrix)(col)
