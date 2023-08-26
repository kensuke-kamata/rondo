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
        result = y.data
        expect = numpy.array(4.0)
        self.assertEqual(result, expect)

    def test_backward(self):
        x = Variable(numpy.array(3.0))
        y = square(x)
        y.backward()
        result = x.grad
        expect = numpy.array(6.0)
        self.assertEqual(result, expect)

    def test_gradient(self):
        x = Variable(numpy.random.rand(1))
        y = square(x)
        y.backward()
        grad_back = x.grad
        grad_num = numerical_diff(square, x)
        isclose = numpy.allclose(grad_back, grad_num)
        self.assertTrue(isclose)
