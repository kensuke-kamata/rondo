import numpy
import unittest

import rondo

class TransposeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.arr = numpy.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    def test_transpose01(self):
        a = rondo.Variable(self.arr)
        b = numpy.array(self.arr)
        result = a.T.data
        expect = b.T
        flg = numpy.allclose(result, expect)
        self.assertTrue(flg)

    def test_transpose02(self):
        a = rondo.Variable(self.arr)
        b = numpy.array(self.arr)
        result = a.transpose(1, 2, 0).data
        expect = b.transpose(1, 2, 0)
        flg = numpy.allclose(result, expect)
        self.assertTrue(flg)

    def test_transpose03(self):
        x = rondo.Variable(self.arr)
        y = x.transpose()
        y.backward()
        result = x.grad.data
        expect = numpy.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
        flg = numpy.allclose(result, expect)
        self.assertTrue(flg)
