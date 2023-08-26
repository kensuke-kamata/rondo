import numpy
import unittest

from rondo.variable import Variable
from rondo.functions import add
from rondo.functions import exp
from rondo.functions import square
from rondo.utils import numerical_diff

class NumericalDiffTest(unittest.TestCase):
    def test_numerical_diff_add0(self):
        x0 = Variable(numpy.array(2.0))
        x1 = Variable(numpy.array(2.0))
        grads = numerical_diff(add, x0, x1)
        result0, result1 = grads[0], grads[1]
        expect = numpy.array(1.0)
        isclose0 = numpy.allclose(result0, expect)
        isclose1 = numpy.allclose(result1, expect)
        self.assertTrue(isclose0 and isclose1)

    def test_numerical_diff_add1(self):
        x = Variable(numpy.array(2.0))
        grads = numerical_diff(add, x, x)
        result = grads[0] + grads[1]
        expect = numpy.array(2.0)
        isclose = numpy.allclose(result, expect)
        self.assertTrue(isclose)

    def test_numerical_diff_exp(self):
        x = Variable(numpy.array(2.0))
        grad = numerical_diff(exp, x)
        result = grad
        expect = numpy.array(7.38905609893)
        isclose = numpy.allclose(result, expect)
        self.assertTrue(isclose)

    def test_numerical_diff_square(self):
        x = Variable(numpy.array(2.0))
        grad = numerical_diff(square, x)
        result = grad
        expect = numpy.array(4.0)
        isclose = numpy.allclose(result, expect)
        self.assertTrue(isclose)
