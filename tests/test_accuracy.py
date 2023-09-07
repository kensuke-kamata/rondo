import numpy as np
import unittest

import rondo.functions as F

class TestAccuracy(unittest.TestCase):
    def setUp(self):
        self.y = np.array([[0.2, 0.8, 0], [0.1, 0.9, 0], [0.8, 0.1, 0.1]])
        self.t = np.array([1, 2, 0])

    def test_forward(self):
        acc = F.accuracy(self.y, self.t)
        result = acc.data
        expect = np.array(0.66666)
        np.testing.assert_array_almost_equal(result, expect, 5)
