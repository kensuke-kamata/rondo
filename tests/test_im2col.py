import unittest
import numpy as np

import rondo
import rondo.functions as F

class TestIm2Col(unittest.TestCase):
    def test_basic(self):
        img = np.random.rand(1, 1, 4, 4)
        kernel = 2
        stride = 2
        pad = 0

        result = F.im2col(img, kernel, stride, pad)
        # Expected shape: (N * OH * OW, C * KH * KW)
        self.assertEqual(result.data.shape, (4, 4))

    def test_to_matrix_false(self):
        img = np.random.rand(1, 1, 4, 4)
        kernel = 2
        stride = 2
        pad = 0

        result = F.im2col(img, kernel, stride, pad, to_matrix=False)
        # Expected shape: (N, C, KH, KW, OH, OW)
        self.assertEqual(result.data.shape, (1, 1, 2, 2, 2, 2))

    def test_paddng(self):
        img = np.random.rand(1, 1, 4, 4)
        kernel = 2
        stride = 2
        pad = 1

        result = F.im2col(img, kernel, stride, pad)
        self.assertEqual(result.data.shape, (9, 4))

    def test_backward(self):
        img = np.random.rand(1, 1, 4, 4)
        x = rondo.Variable(img)
        kernel = 2
        stride = 2
        pad = 0

        y = F.im2col(x, kernel, stride, pad)
        y.backward()
        result = x.grad.data.shape
        expect = img.shape
        np.testing.assert_almost_equal(result, expect, decimal=5)
