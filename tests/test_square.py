import numpy
import unittest

import rondo
from rondo.variable import Variable
from rondo.functions import square
from rondo.utils import numerical_diff

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(numpy.array(2.0))
        y = square(x)
        expected = numpy.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(numpy.array(3.0))
        y = square(x)
        y.backward()
        expected = numpy.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient(self):
        x = Variable(numpy.random.rand(1))
        y = square(x)
        y.backward()
        numgrad = numerical_diff(square, x)
        isclose = numpy.allclose(x.grad, numgrad)
        self.assertTrue(isclose)
