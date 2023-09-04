import numpy as np
import unittest

import rondo
import rondo.functions as F
import rondo.models as M

class TestSoftmax(unittest.TestCase):
    def setUp(self):
        self.model = M.MLP((10, 3), activation=F.softmax)
        self.x = np.array([[0.2, -0.4]])
        self.y = self.model(self.x)

    def test_forward(self):
        print(self.y)
        p = F.softmax(self.y)

        self.assertTrue((p.data >= 0).all() and (p.data <= 1).all())
        self.assertTrue(np.allclose(p.sum(axis=1).data, 1.0))
