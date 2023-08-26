import numpy
import unittest

import rondo
from rondo.variable import Variable
from rondo.functions import add
from rondo.utils import numerical_diff

class AddTest(unittest.TestCase):
    def test_forward(self):
        x0 = Variable(numpy.array(2.0))
        x1 = Variable(numpy.array(3.0))
        y = add(x0, x1)
        result = y.data
        expect = numpy.array(5.0)
        self.assertEqual(result, expect)

    def test_backward(self):
        x0 = Variable(numpy.array(2.0))
        x1 = Variable(numpy.array(3.0))
        y = add(x0, x1)
        y.backward()
        result0 = x0.grad
        result1 = x1.grad
        expect = 1
        self.assertEqual(result0, expect)
        self.assertEqual(result1, expect)
