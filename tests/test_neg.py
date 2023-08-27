import numpy
import unittest

import rondo
from rondo.variable import Variable
from rondo.functions import neg
from rondo.utils import numerical_diff

class NegTest(unittest.TestCase):
    def setUp(self):
        self.x = Variable(numpy.array(2.0))
        self.y = neg(self.x)
        self.y.backward()

    def test_forward(self):
        result = self.y.data
        expect = numpy.array(-2.0)
        self.assertEqual(result, expect)

    def test_backward(self):
        result = self.x.grad
        expect = numpy.array(-1.0)
        self.assertEqual(result, expect)

    def test_gradient(self):
        grad_back = self.x.grad
        grad_num = numerical_diff(neg, self.x)
        isclose = numpy.allclose(grad_back, grad_num)
        self.assertTrue(isclose)

    def test_overload(self):
        y = -self.x
        result = y.data
        expect = numpy.array(-2.0)
        self.assertEqual(result, expect)
