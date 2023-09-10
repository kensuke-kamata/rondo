import unittest
import numpy as np
import rondo

class TestConv2d(unittest.TestCase):
    def test_forward(self):
        # The number of channels has to be the same between x and W
        # (N, C, IH, IW)
        x = np.random.rand(1, 3, 4, 4) # Batch size of 1, 3 channels, 4x4 image
        # (OC, C, KH, KW)
        W = np.random.rand(2, 3, 3, 3) # 2 filters, 3 channels, 3x3 filter
        # (OC, 1, 1)
        b = np.random.rand(2)
        stride = 1
        pad = 1

        # Expect
        IH, IW = x.shape[2:]
        KH, KW = W.shape[2:]
        OH = rondo.utils.get_conv_outsize(IH, KH, stride, pad)
        OW = rondo.utils.get_conv_outsize(IW, KW, stride, pad)
        # (N, OC, OH, OW)
        expected_shape = (x.shape[0], W.shape[0], OH, OW)

        # Call the forward method
        y = rondo.functions.conv2d(x, W, b, stride, pad)
        self.assertEqual(y.data.shape, expected_shape)
