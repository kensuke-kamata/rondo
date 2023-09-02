import numpy as np
import unittest

import rondo
import rondo.functions as F

class TestLinear(unittest.TestCase):
    def setUp(self):
        self.x = rondo.Variable(np.array([[1, 2], [3, 4]]))
        self.W = rondo.Variable(np.array([[0.1, 0.2], [0.3, 0.4]]))
        self.b = rondo.Variable(np.array([0.5, 0.6]))

    def test_forward(self):
        y = F.linear(self.x, self.W, self.b)
        result = y.data
        expect = self.x.data.dot(self.W.data) + self.b.data
        np.testing.assert_array_almost_equal(result, expect)

    def test_forward_without_bias(self):
        y = F.linear(self.x, self.W)
        result = y.data
        expect = self.x.data.dot(self.W.data)
        np.testing.assert_array_almost_equal(result, expect)

    def test_backward(self):
        self.x.cleargrad()
        self.W.cleargrad()
        self.b.cleargrad()
        y = F.linear(self.x, self.W, self.b)
        y.backward()

        np.testing.assert_array_almost_equal(self.x.grad.data, np.array([[0.3, 0.7], [0.3, 0.7]]))
        np.testing.assert_array_almost_equal(self.W.grad.data, np.array([[4, 4], [6, 6]]))
        np.testing.assert_array_almost_equal(self.b.grad.data, np.array([2, 2]))
