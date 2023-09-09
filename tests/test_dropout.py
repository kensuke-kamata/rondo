import unittest

import numpy as np

import rondo
from rondo.functions import dropout

class TestDropout(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def test_forward_train(self):
        # rondo.Config.train = True # Default True
        x = np.ones(5)
        y = dropout(x, dropout_ratio=0.5).data
        self.assertTrue(x.shape == y.shape)

    def test_forward_eval(self):
        with rondo.test_mode():
            x = np.ones(5)
            y = dropout(x, dropout_ratio=0.5).data
            np.testing.assert_array_equal(y, x)

    def test_backward_train(self):
        # rondo.Config.train = True # Default True
        x = rondo.Variable(np.ones(5))
        y = dropout(x, dropout_ratio=0.5)
        forward = y.data != 0
        y.backward(rondo.Variable(np.ones_like(x.data)))
        backward = x.grad.data != 0
        np.testing.assert_array_equal(forward, backward)

    def test_backward_eval(self):
        with rondo.test_mode():
            x = rondo.Variable(np.ones(5))
            y = dropout(x, dropout_ratio=0.5)
            y.backward(rondo.Variable(np.ones_like(x.data)))
            np.testing.assert_array_equal(y.data, x.grad.data)
