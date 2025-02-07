import unittest
import numpy as np
import matplotlib.pyplot as plt
from utils import printer
from datasets import BaseDataSet
from rnn import BaseOperation


def setUpModule():
    BaseDataSet.init_plot()


class BaseOperationTest(unittest.TestCase):
    def test_cell_forward(self):
        np.random.seed(1)
        xt = np.random.randn(3,10)
        a_prev = np.random.randn(5,10)
        Waa = np.random.randn(5,5)
        Wax = np.random.randn(5,3)
        Wya = np.random.randn(2,5)
        ba = np.random.randn(5,1)
        by = np.random.randn(2,1)
        parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

        a_next, yt_pred, cache = BaseOperation.cell_forward(xt, a_prev, parameters)
        print("a_next[4] = ", a_next[4])
        print("a_next.shape = ", a_next.shape)
        print("yt_pred[1] =", yt_pred[1])
        print("yt_pred.shape = ", yt_pred.shape)
