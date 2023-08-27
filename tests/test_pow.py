import numpy
import unittest

import rondo
from rondo.variable import Variable
from rondo.utils import numerical_diff

class PowTest(unittest.TestCase):
    def setUp(self):
        self.x = Variable(numpy.array(2.0))
        self.y = self.x ** 3
        self.y.backward()

    def test_forward(self):
        result = self.y.data
        expect = numpy.array(8.0)
        self.assertEqual(result, expect)

    def test_backward(self):
        result = self.x.grad
        expect = numpy.array(12.0)
        self.assertEqual(result, expect)

    def test_gradient(self):
        grad_back = self.x.grad
        c = 3.0
        grad_num = numerical_diff(pow, self.x, c)
        isclose = numpy.allclose(grad_back, grad_num)
        self.assertTrue(isclose)
