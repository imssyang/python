import unittest
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
from utils import printer
from datasets import SignDataSet as DataSet
from cnn_tensorflow import TensorflowOperation, TensorflowCNN


def setUpModule():
    tf_v1.disable_v2_behavior()


class TensorflowOperationTest(unittest.TestCase):
    def test_create_placeholders(self):
        X, Y = TensorflowOperation.create_placeholders(64, 64, 3, 6)
        printer("X =", X)
        printer("Y =", Y)

    def test_initialize_parameters(self):
        tf_v1.reset_default_graph()
        with tf_v1.Session() as sess:
            parameters = TensorflowOperation.initialize_parameters()
            init = tf_v1.global_variables_initializer()
            sess.run(init)
            print("W1 = " + str(parameters["W1"].eval()[1,1,1]))
            print("W2 = " + str(parameters["W2"].eval()[1,1,1]))

    def test_forward_propagation(self):
        # The forward propagation doesn't output any cache.
        tf_v1.reset_default_graph()
        with tf_v1.Session() as sess:
            np.random.seed(1)
            X, Y = TensorflowOperation.create_placeholders(64, 64, 3, 6)
            parameters = TensorflowOperation.initialize_parameters()
            Z3 = TensorflowOperation.forward_propagation(X, parameters)
            init = tf_v1.global_variables_initializer()
            sess.run(init)
            a = sess.run(Z3, {X: np.random.randn(2,64,64,3), Y: np.random.randn(2,6)})
            printer("Z3 =", a)

    def test_compute_cost(self):
        tf_v1.reset_default_graph()
        with tf_v1.Session() as sess:
            np.random.seed(1)
            X, Y = TensorflowOperation.create_placeholders(64, 64, 3, 6)
            parameters = TensorflowOperation.initialize_parameters()
            Z3 = TensorflowOperation.forward_propagation(X, parameters)
            cost = TensorflowOperation.compute_cost(Z3, Y)
            init = tf_v1.global_variables_initializer()
            sess.run(init)
            a = sess.run(cost, {X: np.random.randn(4,64,64,3), Y: np.random.randn(4,6)})
            printer("cost =", a)


class TensorflowCNNTest(unittest.TestCase):
    def setUp(self):
        self.X_train_orig, self.Y_train_orig, self.X_test_orig, self.Y_test_orig, self.classes = DataSet.load()
        DataSet.show_sign(self.X_train_orig, self.Y_train_orig, 6)

        self.X_train, self.Y_train, self.X_test, self.Y_test = self.scalling_dataset(
            self.X_train_orig, self.Y_train_orig,
            self.X_test_orig, self.Y_test_orig
        )
        self.learning_rate = 0.009
        self.loss_xlable = 'iterations (per tens)'

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
        parameters, costs = TensorflowCNN.model(self.X_train, self.Y_train, self.X_test, self.Y_test)
        DataSet.show_loss(costs, self.learning_rate, self.loss_xlable)

        image, image64 = DataSet.load_thumb_image(flatten=False)
        image_prediction = TensorflowCNN.predict(image64, parameters)
        print("Model predicts: y = " + str(np.squeeze(image_prediction)))
        DataSet.show_image(image)
