import numpy
import unittest

import rondo

class BroadcastTest(unittest.TestCase):
    def setUp(self) -> None:
        self.arr = numpy.array([1, 2, 3])
        self.scl = numpy.array([10])
        self.x0 = rondo.Variable(self.arr)
        self.x1 = rondo.Variable(self.scl)

    def test_backward(self):
        y = self.x0 + self.x1
        y.backward()
        result = self.x0.grad.data
        expect = numpy.array([3])
        flg = numpy.allclose(result, expect)
        self.assertTrue(flg)
