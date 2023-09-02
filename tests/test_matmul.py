import numpy as np
import unittest

import rondo
import rondo.functions as F

class TestMatmul(unittest.TestCase):
    def test_forward_basic(self):
        arr_x = np.array([[1, 2], [3, 4]])
        arr_W = np.array([[2, 3], [4, 5]])
        x = rondo.Variable(arr_x)
        W = rondo.Variable(arr_W)
        result = F.matmul(x, W).data
        expect = np.dot(arr_x, arr_W)
        np.testing.assert_array_equal(result, expect)

    def test_backward_basic(self):
        arr_x = np.array([[1, 2], [3, 4]])
        arr_W = np.array([[2, 3], [4, 5]])
        x = rondo.Variable(arr_x)
        W = rondo.Variable(arr_W)
        gy = np.array([[1, 2], [3, 4]], dtype=float)

        y = F.matmul(x, W)
        y.grad = rondo.Variable(gy)
        y.backward()
        gx = x.grad.data
        gW = W.grad.data

        expect_gx = np.dot(gy, arr_W.T)
        expect_gW = np.dot(arr_x.T, gy)

        np.testing.assert_array_almost_equal(gx, expect_gx)
        np.testing.assert_array_almost_equal(gW, expect_gW)

    def test_forward_non_square_matrices(self):
        arr_x = np.array([[1, 2, 3], [4, 5, 6]])
        arr_W = np.array([[1, 2], [3, 4], [5, 6]])
        x = rondo.Variable(arr_x)
        W = rondo.Variable(arr_W)
        result = F.matmul(x, W).data
        expect = np.dot(arr_x, arr_W)
        np.testing.assert_array_equal(result, expect)

    def test_backward_non_square_matrices(self):
        arr_x = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        arr_W = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        x = rondo.Variable(arr_x)
        W = rondo.Variable(arr_W)
        gy = np.array([[1, 2], [3, 4]], dtype=float)

        y = F.matmul(x, W)
        y.grad = rondo.Variable(gy)
        y.backward()
        gx = x.grad.data
        gW = W.grad.data

        expect_gx = np.dot(gy, arr_W.T)
        expect_gW = np.dot(arr_x.T, gy)

        np.testing.assert_array_almost_equal(gx, expect_gx)
        np.testing.assert_array_almost_equal(gW, expect_gW)

    def test_forward_edge_cases(self):
        # Edge case: Matrix containing zeros
        arr_x = np.array([[0, 0], [0, 0]])
        arr_W = np.array([[1, 2], [3, 4]])
        x = rondo.Variable(arr_x)
        W = rondo.Variable(arr_W)
        result = F.matmul(x, W).data
        expect = np.dot(arr_x, arr_W)
        np.testing.assert_array_equal(result, expect)

    def test_error_scenario_incompatible_shapes(self):
        # Error scenario: Mismatched matrix dimensions
        arr_x = np.array([[1, 2, 3], [4, 5, 6]])
        arr_W = np.array([[1, 2, 3], [4, 5, 6]])
        x = rondo.Variable(arr_x)
        W = rondo.Variable(arr_W)
        with self.assertRaises(ValueError):
            F.matmul(x, W)
