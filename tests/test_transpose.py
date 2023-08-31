import numpy
import unittest

import rondo

class TransposeTest(unittest.TestCase):
    def test_transpose(self):
        x = rondo.Variable(numpy.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
        a = numpy.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = x.T.data
        expect = a.T
        flg = numpy.allclose(result, expect)
        self.assertTrue(flg)
