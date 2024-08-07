import unittest
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
import keras.backend as K
from utils import printer
from datasets import SignDataSet as DataSet
from cnn_resnet import ResnetBlock, ResnetModel


def setUpModule():
    tf_v1.disable_v2_behavior()


class ResnetBlockTest(unittest.TestCase):
    def test_identity(self):
        tf_v1.reset_default_graph()
        with tf_v1.Session() as session:
            np.random.seed(1)
            A_prev = tf_v1.placeholder("float", [3, 4, 4, 6])
            X = np.random.randn(3, 4, 4, 6)
            A = ResnetBlock.identity(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
            session.run(tf_v1.global_variables_initializer())
            out = session.run([A], feed_dict={A_prev: X})
            printer("out = ", out[0][1][1][0])

    def test_convolutional(self):
        tf_v1.reset_default_graph()
        with tf_v1.Session() as session:
            np.random.seed(1)
            A_prev = tf_v1.placeholder("float", [3, 4, 4, 6])
            X = np.random.randn(3, 4, 4, 6)
            A = ResnetBlock.convolutional(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
            session.run(tf_v1.global_variables_initializer())
            out = session.run([A], feed_dict={A_prev: X})
            printer("out = ", out[0][1][1][0])


class ResnetModelTest(unittest.TestCase):
    def setUp(self):
        self.X_train_orig, self.Y_train_orig, self.X_test_orig, self.Y_test_orig, self.classes = DataSet.load()
        DataSet.show_sign(self.X_train_orig, self.Y_train_orig, 3)

        self.X_train, self.Y_train, self.X_test, self.Y_test = self.scalling_dataset(
            self.X_train_orig, self.Y_train_orig,
            self.X_test_orig, self.Y_test_orig
        )

    def convert_to_one_hot(self, Y, C):
        Y = np.eye(C)[Y.reshape(-1)].T
        return Y

    def scalling_dataset(self, X_train_orig, Y_train_orig, X_test_orig, Y_test_orig):
        # Normalize image vectors
        X_train = X_train_orig / 255.
        X_test = X_test_orig / 255.
        # Convert training and test labels to one hot matrices
        Y_train = self.convert_to_one_hot(Y_train_orig, 6).T
        Y_test = self.convert_to_one_hot(Y_test_orig, 6).T
        print("number of training examples = " + str(X_train.shape[0]))
        print("number of test examples = " + str(X_test.shape[0]))
        print("X_train shape: " + str(X_train.shape))
        print("Y_train shape: " + str(Y_train.shape))
        print("X_test shape: " + str(X_test.shape))
        print("Y_test shape: " + str(Y_test.shape))
        print("data classes: ", self.classes)
        return X_train, Y_train, X_test, Y_test

    def test_model(self):
        model = ResnetModel.model(self.X_train, self.Y_train, self.X_test, self.Y_test)
        ResnetModel.summary(model)
        ResnetModel.plot(model, 'datasets/resnet50.png')
