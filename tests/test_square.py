import numpy
import unittest

import rondo
from rondo.variable import Variable
from rondo.functions import square

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(numpy.array(2.0))
        y = square(x)
        expected = numpy.array(4.0)
        self.assertEqual(y.data, expected)
