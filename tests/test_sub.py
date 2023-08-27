import numpy
import unittest

import rondo
from rondo.variable import Variable
from rondo.functions import sub
from rondo.utils import numerical_diff

class SubTest(unittest.TestCase):
    def setUp(self):
        self.x0 = Variable(numpy.array(2.0))
        self.x1 = Variable(numpy.array(3.0))
        self.y = sub(self.x0, self.x1)
        self.y.backward()

    def test_forward(self):
        result = self.y.data
        expect = numpy.array(-1.0)
        self.assertEqual(result, expect)

    def test_backward(self):
        result0 = self.x0.grad
        result1 = self.x1.grad
        expect0 = numpy.array(1.0)
        expect1 = numpy.array(-1.0)
        self.assertEqual(result0, expect0)
        self.assertEqual(result1, expect1)

    def test_gradient(self):
        grad_back0 = self.x0.grad
        grad_back1 = self.x1.grad
        grads_num = numerical_diff(sub, self.x0, self.x1)
        isclose0 = numpy.allclose(grad_back0, grads_num[0])
        isclose1 = numpy.allclose(grad_back1, grads_num[1])
        self.assertTrue(isclose0)
        self.assertTrue(isclose1)

    def test_overload(self):
        a = self.x0 - 3.0
        result = a.data
        expect = numpy.array(-1.0)
        self.assertEqual(result, expect)

        a = self.x0 - numpy.array(3.0)
        result = a.data
        expect = numpy.array(-1.0)
        self.assertEqual(result, expect)

        a = 3.0 - self.x0
        result = a.data
        expect = numpy.array(1.0)
        self.assertEqual(result, expect)

        a = numpy.array(3.0) - self.x0
        result = a.data
        expect = numpy.array(1.0)
        self.assertEqual(result, expect)

