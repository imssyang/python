import unittest
import time
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    sigmoid,
    sigmoid_derivative,
    softmax,
    L1,
    L2,
    printer,
)


class NumpyTest(unittest.TestCase):
    def test_vector(self):
        x = np.array([1, 2, 3])
        np.testing.assert_array_equal(
            x + 3,
            np.array([4, 5, 6]),
        )
        self.assertEqual(
            (x + 3).tolist(),
            [4, 5, 6],
        )

    def test_reshape(self):
        # This is a 3 by 3 by 2 array, typically images will be (num_px_x, num_px_y,3) where 3 represents the RGB values
        image = np.array(
            [[[ 0.67826139,  0.29380381],
                [ 0.90714982,  0.52835647],
                [ 0.4215251 ,  0.45017551]],
            [[ 0.92814219,  0.96677647],
                [ 0.85304703,  0.52351845],
                [ 0.19981397,  0.27417313]],
            [[ 0.60659855,  0.00533165],
                [ 0.10820313,  0.49978937],
                [ 0.34144279,  0.94630077]]]
        )
        np.testing.assert_allclose(
            image.reshape((-1, 1)),
            np.array(
                [[0.67826139],
                [0.29380381],
                [0.90714982],
                [0.52835647],
                [0.4215251 ],
                [0.45017551],
                [0.92814219],
                [0.96677647],
                [0.85304703],
                [0.52351845],
                [0.19981397],
                [0.27417313],
                [0.60659855],
                [0.00533165],
                [0.10820313],
                [0.49978937],
                [0.34144279],
                [0.94630077]]
            ),
            rtol=1e-6, atol=1e-6,
        )

    def test_normalize_rows(self):
        x = np.array(
            [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]]
        )
        np.testing.assert_allclose(
            np.linalg.norm(x, ord=2, axis=1, keepdims=True),
            np.array(
                [[3.74165739], # sqrt(1^2 + 2^2 + 3^2)
                [ 8.77496439], # sqrt(4^2 + 5^2 + 6^2)
                [13.92838828]] # sqrt(7^2 + 8^2 + 9^2)
            ),
            rtol=1e-6, atol=1e-6,
        )

    def test_dot_product(self):
        x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
        x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
        start = time.process_time()
        dot = 0
        for i in range(len(x1)):
            dot += x1[i] * x2[i]
        diff_classic = (time.process_time() - start) * 1000
        self.assertEqual(dot, 278)

        start = time.process_time()
        dot = np.dot(x1, x2)
        diff_vectorized = (time.process_time() - start) * 1000
        self.assertEqual(dot, 278)
        print("Time of dot product of vectors> classic:", diff_classic, "vectorized:", diff_vectorized, "ms")

    def test_outer_product(self):
        x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
        x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
        result = [[81, 18, 18, 81,  0, 81, 18, 45,  0,  0, 81, 18, 45,  0,  0],
            [18,  4,  4, 18,  0, 18,  4, 10,  0,  0, 18,  4, 10,  0,  0],
            [45, 10, 10, 45,  0, 45, 10, 25,  0,  0, 45, 10, 25,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [63, 14, 14, 63,  0, 63, 14, 35,  0,  0, 63, 14, 35,  0,  0],
            [45, 10, 10, 45,  0, 45, 10, 25,  0,  0, 45, 10, 25,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [81, 18, 18, 81,  0, 81, 18, 45,  0,  0, 81, 18, 45,  0,  0],
            [18,  4,  4, 18,  0, 18,  4, 10,  0,  0, 18,  4, 10,  0,  0],
            [45, 10, 10, 45,  0, 45, 10, 25,  0,  0, 45, 10, 25,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]]

        start = time.process_time()
        outer = np.zeros((len(x1),len(x2))) # we create a len(x1)*len(x2) matrix with only zeros
        for i in range(len(x1)):
            for j in range(len(x2)):
                outer[i,j] = x1[i] * x2[j]
        diff_classic = (time.process_time() - start) * 1000
        self.assertEqual(outer.tolist(), result)

        start = time.process_time()
        outer = np.outer(x1, x2)
        diff_vectorized = (time.process_time() - start) * 1000
        self.assertEqual(outer.tolist(), result)
        print("Time of outer product of vectors> classic:", diff_classic, "vectorized:", diff_vectorized, "ms")

    def test_multiply(self):
        x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
        x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
        result = [81,  4, 10,  0,  0, 63, 10,  0,  0,  0, 81,  4, 25,  0,  0]

        start = time.process_time()
        mul = np.zeros(len(x1))
        for i in range(len(x1)):
            mul[i] = x1[i] * x2[i]
        diff_classic = (time.process_time() - start) * 1000
        self.assertEqual(mul.tolist(), result)

        start = time.process_time()
        mul = np.multiply(x1,x2)
        diff_vectorized = (time.process_time() - start) * 1000
        self.assertEqual(mul.tolist(), result)
        print("Time of multiply of vectors> classic:", diff_classic, "vectorized:", diff_vectorized, "ms")

    def test_random_dot_product(self):
        x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
        np.random.seed(1)
        W = np.random.rand(3,len(x1)) # Random 3*len(x1) numpy array
        result = [12.937541002644906, 28.52560850833297, 23.95619952995058]

        start = time.process_time()
        gdot = np.zeros(W.shape[0])
        for i in range(W.shape[0]):
            for j in range(len(x1)):
                gdot[i] += W[i,j] * x1[j]
        diff_classic = (time.process_time() - start) * 1000
        np.testing.assert_allclose(gdot, np.array(result), rtol=1e-6, atol=1e-6)

        start = time.process_time()
        gdot = np.dot(W,x1)
        diff_vectorized = (time.process_time() - start) * 1000
        np.testing.assert_allclose(gdot, np.array(result), rtol=1e-6, atol=1e-6)
        print("Time of random dot product of vectors> classic:", diff_classic, "vectorized:", diff_vectorized, "ms")


class UtilsTest(unittest.TestCase):
    def test_sigmoid(self):
        x = np.array([1, 2, 3])
        np.testing.assert_allclose(
            sigmoid(x),
            np.array([0.73105858, 0.88079708, 0.95257413]),
            rtol=1e-6, atol=1e-6,
        )
        np.testing.assert_allclose(
            sigmoid_derivative(x),
            np.array([0.19661193, 0.10499359, 0.04517666]),
            rtol=1e-6, atol=1e-6,
        )

    def test_softmax(self):
        x = np.array([
            [9, 2, 5, 0, 0],
            [7, 5, 0, 0 ,0]])
        np.testing.assert_allclose(
            softmax(x),
            np.array(
                [[9.808977e-01, 8.944629e-04, 1.796577e-02, 1.210524e-04, 1.210524e-04],
                [8.786799e-01, 1.189164e-01, 8.012523e-04, 8.012523e-04, 8.012523e-04]]
            ),
            rtol=1e-6, atol=1e-6,
        )

    def test_L1(self):
        yhat = np.array([.9, 0.2, 0.1, .4, .9])
        y = np.array([1, 0, 0, 1, 1])
        np.testing.assert_allclose(
            L1(yhat, y),
            1.1,
            rtol=1e-6, atol=1e-6,
        )

    def test_L2(self):
        yhat = np.array([.9, 0.2, 0.1, .4, .9])
        y = np.array([1, 0, 0, 1, 1])
        np.testing.assert_allclose(
            L2(yhat, y),
            0.43,
            rtol=1e-6, atol=1e-6,
        )
