import numpy
import unittest

import rondo
from rondo.variable import Variable
from rondo.functions import add
from rondo.utils import numerical_diff

class AddTest(unittest.TestCase):
    def setUp(self) -> None:
        self.arr = numpy.array([1, 2, 3])
        self.scl = numpy.array([10])
        self.x0 = rondo.Variable(self.arr)
        self.x1 = rondo.Variable(self.scl)

    def test_forward(self):
        x0 = Variable(numpy.array(2.0))
        x1 = Variable(numpy.array(3.0))
        y = add(x0, x1)
        result = y.data
        expect = numpy.array(5.0)
        self.assertEqual(result, expect)

    def test_gradient(self):
        x0 = Variable(numpy.random.rand(1))
        x1 = Variable(numpy.random.rand(1))
        y = add(x0, x1)
        y.backward()
        grad_back0 = x0.grad.data
        grad_back1 = x1.grad.data
        grads_num = numerical_diff(add, x0, x1)
        isclose0 = numpy.allclose(grad_back0, grads_num[0])
        isclose1 = numpy.allclose(grad_back1, grads_num[1])
        self.assertTrue(isclose0 and isclose1)

    def test_backward00(self):
        """A basic backward testing"""
        x0 = Variable(numpy.array(2.0))
        x1 = Variable(numpy.array(3.0))
        y = add(x0, x1)
        y.backward()
        result0 = x0.grad.data
        result1 = x1.grad.data
        expect = 1
        self.assertEqual(result0, expect)
        self.assertEqual(result1, expect)

    def test_backward01(self):
        """Use the same variable"""
        x = Variable(numpy.array(3.0))
        y = add(x, x)
        y.backward()
        result = x.grad.data
        expect = numpy.array(2.0)
        self.assertEqual(result, expect)

    def test_backward02(self):
        # broadcast
        y = self.x0 + self.x1
        y.backward()
        result0 = self.x0.grad.data
        result1 = self.x1.grad.data
        expect0 = numpy.array([1, 1, 1,])
        expect1 = numpy.array([3])
        flg0 = numpy.allclose(result0, expect0)
        self.assertTrue(flg0)
        flg1 = numpy.allclose(result1, expect1)
        self.assertTrue(flg1)
