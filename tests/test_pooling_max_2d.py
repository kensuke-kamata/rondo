import unittest
import numpy as np
import rondo
import rondo.functions as F
import dezero
import dezero.functions as DF

class TestPoolingMax2d(unittest.TestCase):

    def setUp(self):
        self.x = np.array([
            [[
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]
            ]]
        ]).astype('f')
        self.kernel = 2
        self.stride = 2
        self.pad = 0

    def test_forward(self):
        y = F.pooling_max_2d(self.x, self.kernel, self.stride, self.pad)
        result = y.data
        expect = np.array([
            [[
                [6, 8],
                [14, 16]
            ]]
        ])
        np.testing.assert_array_equal(result, expect)

    def test_backward(self):
        x = rondo.as_variable(self.x)
        y = F.pooling_max_2d(x, self.kernel, self.stride, self.pad)
        y.backward()

        result = x.grad.data
        expect = np.array([
            [[
                [0, 0, 0, 0],
                [0, 1, 0, 1],
                [0, 0, 0, 0],
                [0, 1, 0, 1]
            ]]
        ])
        np.testing.assert_array_equal(result, expect)
