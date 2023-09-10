import unittest
import numpy as np

import rondo
import rondo.functions as F

class TestCol2Im(unittest.TestCase):
    def test_basic(self):
        img = np.random.rand(1, 1, 4, 4)
        x = rondo.Variable(img)
        kernel = 2
        stride = 2
        pad = 0

        y = F.im2col(x, kernel, stride, pad)
        res = F.col2im(y, img.shape, kernel, stride, pad)

        np.testing.assert_almost_equal(res.data, img, decimal=5)

    def test_to_matrix_false(self):
        img = np.random.rand(1, 1, 4, 4)
        x = rondo.Variable(img)
        kernel = 2
        stride = 2
        pad = 0

        y = F.im2col(x, kernel, stride, pad, to_matrix=False)
        res = F.col2im(y, img.shape, kernel, stride, pad, to_matrix=False)

        np.testing.assert_almost_equal(res.data, img, decimal=5)

    def test_padding(self):
        img = np.random.rand(1, 1, 4, 4)
        x = rondo.Variable(img)
        kernel = 2
        stride = 2
        pad = 1

        y = F.im2col(x, kernel, stride, pad)
        res = F.col2im(y, img.shape, kernel, stride, pad)

        np.testing.assert_almost_equal(res.data, img, decimal=5)
