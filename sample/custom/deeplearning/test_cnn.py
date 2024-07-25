import unittest
import numpy as np
import matplotlib.pyplot as plt
from utils import printer
from datasets import BaseDataSet
from cnn import BaseOperation


def setUpModule():
    BaseDataSet.init_plot()


class BaseOperationTest(unittest.TestCase):
    def test_zero_pad(self):
        np.random.seed(1)
        x = np.random.randn(4, 3, 3, 2)
        x_pad = BaseOperation.zero_pad(x, 2)
        printer("x.shape =", x.shape)
        printer("x_pad.shape =", x_pad.shape)
        printer("x[1,1] =", x[1,1])
        printer("x_pad[1,1] =", x_pad[1,1])
        printer("x_pad[1,3] =", x_pad[1,3])

        fig, axarr = plt.subplots(1, 2)
        axarr[0].set_title('x')
        axarr[0].imshow(x[0,:,:,0])
        axarr[1].set_title('x_pad')
        axarr[1].imshow(x_pad[0,:,:,0])
        plt.show()

    def test_conv_single_step(self):
        np.random.seed(1)
        a_slice_prev = np.random.randn(4, 4, 3)
        W = np.random.randn(4, 4, 3)
        b = np.random.randn(1, 1, 1)

        Z = BaseOperation.conv_single_step(a_slice_prev, W, b)
        print("Z =", Z)

    def test_create_mask_from_window(self):
        np.random.seed(1)
        x = np.random.randn(2,3)
        mask = BaseOperation.create_mask_from_window(x)
        print('x = ', x)
        print("mask = ", mask)

    def test_distribute_value(self):
        a = BaseOperation.distribute_value(2, (2,2))
        print('distributed value =', a)

    def test_conv(self):
        np.random.seed(1)
        A_prev = np.random.randn(10,4,4,3)
        W = np.random.randn(2,2,3,8)
        b = np.random.randn(1,1,1,8)
        hparameters = {"pad" : 2,
                    "stride": 1}

        Z, cache_conv = BaseOperation.conv_forward(A_prev, W, b, hparameters)
        print("Z's mean =", np.mean(Z))
        print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])

        dA, dW, db = BaseOperation.conv_backward(Z, cache_conv)
        print("dA_mean =", np.mean(dA))
        print("dW_mean =", np.mean(dW))
        print("db_mean =", np.mean(db))

    def test_pool_forward(self):
        np.random.seed(1)
        A_prev = np.random.randn(2, 4, 4, 3)
        hparameters = {"stride" : 1, "f": 4}

        A, cache = BaseOperation.pool_forward(A_prev, hparameters, mode = "max")
        print("mode = max")
        print("A =", A)
        print()
        A, cache = BaseOperation.pool_forward(A_prev, hparameters, mode = "average")
        print("mode = average")
        print("A =", A)

    def test_pool(self):
        np.random.seed(1)
        A_prev = np.random.randn(5, 5, 3, 2)
        hparameters = {"stride" : 1, "f": 2}
        A, cache = BaseOperation.pool_forward(A_prev, hparameters, mode = "max")
        dA = np.random.randn(5, 4, 2, 2)

        dA_prev = BaseOperation.pool_backward(dA, cache, mode = "max")
        print("mode = max")
        print('mean of dA = ', np.mean(dA))
        print('dA_prev[1,1] = ', dA_prev[1,1])
        print()
        dA_prev = BaseOperation.pool_backward(dA, cache, mode = "average")
        print("mode = average")
        print('mean of dA = ', np.mean(dA))
        print('dA_prev[1,1] = ', dA_prev[1,1])
