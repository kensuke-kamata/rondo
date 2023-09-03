import numpy as np
import unittest

import rondo
import rondo.functions as F

class TestGetItem(unittest.TestCase):
    def setUp(self):
        self.x = rondo.Variable(np.array([[1, 2, 3], [4, 5, 6]]))

    def test_forward_simple(self):
        y = F.get_item(self.x, 1)
        result = y.data
        expect = np.array([4, 5, 6])
        self.assertTrue(np.allclose(result, expect))

    def test_forward_slice(self):
        indices = np.array([0, 0, 1])
        y = F.get_item(self.x, indices)
        result = y.data
        expect = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]])
        self.assertTrue(np.allclose(result, expect))

    def test_forward_slice_magic_method(self):
        indices = np.array([0, 0, 1])
        y = self.x[indices]
        result = y.data
        expect = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]])
        self.assertTrue(np.allclose(result, expect))

    def test_backward(self):
        self.x.cleargrad()
        y = F.get_item(self.x, 1)
        y.backward()
        result = self.x.grad.data
        expect = np.array([[0, 0, 0], [1, 1, 1]])
        self.assertTrue(np.allclose(result, expect))
