import unittest
import numpy as np
import matplotlib.pyplot as plt
from utils import printer
from nn3_check import LinearCheck, BaseNNCheck


class LinearCheckTest(unittest.TestCase):
    def test_common(self):
        x, theta = 2, 4
        J = LinearCheck.forward_propagation(x, theta)
        printer("J =", J) # 8

        dtheta = LinearCheck.backward_propagation(x, theta)
        printer("dtheta =", dtheta) # 2

        difference = LinearCheck.gradient_check(x, theta)
        printer("difference =", difference) # 2.919335883291695e-10


class BaseNNCheckTest(unittest.TestCase):
    def setUp(self):
        self.X, self.Y, self.parameters = self.dataset_and_parameters()

    def dataset_and_parameters(self):
        np.random.seed(1)
        x = np.random.randn(4,3)
        y = np.array([1, 1, 0])
        W1 = np.random.randn(5,4)
        b1 = np.random.randn(5,1)
        W2 = np.random.randn(3,5)
        b2 = np.random.randn(3,1)
        W3 = np.random.randn(1,3)
        b3 = np.random.randn(1,1)
        parameters = {"W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2,
                    "W3": W3,
                    "b3": b3}
        return x, y, parameters

    def test_common(self):
        X, Y, parameters = self.X, self.Y, self.parameters
        cost, cache = BaseNNCheck.forward_propagation(X, Y, parameters)
        gradients = BaseNNCheck.backward_propagation(X, Y, cache)
        difference = BaseNNCheck.gradient_check(parameters, gradients, X, Y)
        printer("difference =", difference) # 0.33333334789859204
