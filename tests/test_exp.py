import numpy
import unittest

import rondo
from rondo.variable import Variable
from rondo.functions import exp
from rondo.utils import numerical_diff

class ExpTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(numpy.array(2.0))
        y = exp(x)
        result = y.data
        expect = numpy.array(7.38905609893)
        isclose = numpy.allclose(result, expect)
        self.assertTrue(isclose)

    def test_backward(self):
        x = Variable(numpy.array(2.0))
        y = exp(x)
        y.backward()
        result = x.grad
        expect = numpy.array(7.38905609893)
        isclose = numpy.allclose(result, expect)
        self.assertTrue(isclose)

    def test_gradient(self):
        x = Variable(numpy.array(2.0))
        y = exp(x)
        y.backward()
        grad_back = x.grad
        grad_num = numerical_diff(exp, x)
        isclose = numpy.allclose(grad_back, grad_num)
        self.assertTrue(isclose)
