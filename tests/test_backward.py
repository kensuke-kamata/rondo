import numpy
import unittest

import rondo
from rondo.variable import Variable
from rondo.functions import add
from rondo.functions import square
from rondo.utils import numerical_diff

class BackwardTest(unittest.TestCase):
    def test_backward(self):
        x = Variable(numpy.array(2.0))
        a = square(x)
        y = add(square(a), square(a))
        y.backward()
        result = x.grad
        expect = numpy.array(64.0)
        self.assertEqual(result, expect)
