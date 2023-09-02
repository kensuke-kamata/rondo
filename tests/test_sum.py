import numpy
import unittest

import rondo

class SumTest(unittest.TestCase):
    def setUp(self):
        self.x0 = rondo.Variable(numpy.array([[1, 2, 3, 4, 5, 6]]))
        self.x1 = rondo.Variable(numpy.array([[1, 2, 3], [4, 5, 6]]))
        self.x2 = rondo.Variable(numpy.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9,10,11,12]
        ]))

    def test_forward_simple(self):
        y = self.x0.sum()
        result = y.data
        expect = 21
        self.assertEqual(result, expect)

    def test_forward_axis(self):
        y = self.x1.sum(axis=0)
        result = y.data
        expect = numpy.array([5, 7, 9])
        flg = numpy.allclose(result, expect)
        self.assertTrue(flg)

    def test_forward_keepdims(self):
        y = self.x1.sum(keepdims=True)
        result = y.data
        expect = numpy.array([21])
        flg = numpy.allclose(result, expect)
        self.assertTrue(flg)

    def test_backward_simple(self):
        self.x0.cleargrad()
        y = self.x0.sum()
        y.backward()
        result = self.x0.grad.data
        expect = numpy.array([1, 1, 1, 1, 1, 1])
        flg = numpy.allclose(result, expect)
        self.assertTrue(flg)

    def test_backward_keepdims(self):
        self.x1.cleargrad()
        y = self.x1.sum(keepdims=True)
        y.backward()
        result = self.x1.grad.data
        expect = numpy.array([[1, 1, 1], [1, 1, 1]])
        flg = numpy.allclose(result, expect)
        self.assertTrue(flg)

    def test_forward(self):
        y = self.x2.sum(axis=1, keepdims=False)
        result = y.data
        expect = numpy.array([10, 26, 42])
        numpy.testing.assert_array_almost_equal(result, expect)

    def test_backward(self):
        # Ensure the gradient is correctly reshaped and broadcasted back to the
        # shape of the input when summing over a specified axis with keepdims=False.
        self.x2.cleargrad()
        y = self.x2.sum(axis=1, keepdims=False)
        y.backward()
        result = self.x2.grad.data
        expect = numpy.ones_like(self.x2.data)
        numpy.testing.assert_array_equal(result, expect)
