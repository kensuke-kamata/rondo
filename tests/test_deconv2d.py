import unittest
import numpy as np
import rondo

class TestDeconv2d(unittest.TestCase):
    def test_forward(self):
        # The number of channels has to be the same between x and W
        # (N, C, IH, IW)
        x = np.random.rand(1, 2, 3, 3) # Batch size of 1, 2 channels, 3x3 filter
        # (C, OC, KH, KW)
        W = np.random.rand(2, 3, 4, 4) # 2 input channels, 3 output channels (RGB), 4x4 image
        b = np.random.rand(3)
        stride = 1
        pad = 1

        # Expect
        IH, IW = x.shape[2:]
        OC, KH, KW = W.shape[1:]
        out_h = rondo.utils.get_deconv_outsize(IH, KH, stride, pad)
        out_w = rondo.utils.get_deconv_outsize(IW, KW, stride, pad)
        # (N, OC, OH, OW)
        expected_shape = (x.shape[0], OC, out_h, out_w)

        # Call the forward method
        y = rondo.functions.deconv2d(x, W, b, stride, pad)
        self.assertEqual(y.data.shape, expected_shape)

        # Console output
        print(x)
        print(y.data)
