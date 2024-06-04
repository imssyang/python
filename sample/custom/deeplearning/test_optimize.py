import unittest
import numpy as np
from utils import printer
from datasets import MoonDataSet as DataSet
from nn import BaseOperation
from nn_optimize import (
    MiniBatchOperation,
    MomentumOperation,
    AdamOperation,
    OptimizeNN,
)


class GradientDescentTest(unittest.TestCase):
    def update_parameters_test_case(self):
        np.random.seed(1)
        learning_rate = 0.01
        W1 = np.random.randn(2,3)
        b1 = np.random.randn(2,1)
        W2 = np.random.randn(3,3)
        b2 = np.random.randn(3,1)

        dW1 = np.random.randn(2,3)
        db1 = np.random.randn(2,1)
        dW2 = np.random.randn(3,3)
        db2 = np.random.randn(3,1)

        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return parameters, grads, learning_rate

    def test_update_parameters(self):
        parameters, grads, learning_rate = self.update_parameters_test_case()
        parameters = BaseOperation.update_parameters(parameters, grads, learning_rate)
        printer("W1 =", parameters["W1"])
        printer("b1 =", parameters["b1"])
        printer("W2 =", parameters["W2"])
        printer("b2 =", parameters["b2"])


class MiniBatchOperationTest(unittest.TestCase):
    def random_mini_batches_test_case(self):
        np.random.seed(1)
        mini_batch_size = 64
        X = np.random.randn(12288, 148)
        Y = np.random.randn(1, 148) < 0.5
        return X, Y, mini_batch_size

    def test_mini_batches(self):
        X_assess, Y_assess, mini_batch_size = self.random_mini_batches_test_case()
        mini_batches = MiniBatchOperation.random_mini_batches(X_assess, Y_assess, mini_batch_size)
        printer("shape of the 1st mini_batch_X: ", mini_batches[0][0].shape)
        printer("shape of the 2nd mini_batch_X: ", mini_batches[1][0].shape)
        printer("shape of the 3rd mini_batch_X: ", mini_batches[2][0].shape)
        printer("shape of the 1st mini_batch_Y: ", mini_batches[0][1].shape)
        printer("shape of the 2nd mini_batch_Y: ", mini_batches[1][1].shape)
        printer("shape of the 3rd mini_batch_Y: ", mini_batches[2][1].shape)
        printer("mini batch sanity check: ", mini_batches[0][0][0][0:3])


class MomentumOperationTest(unittest.TestCase):
    def initialize_velocity_test_case(self):
        np.random.seed(1)
        W1 = np.random.randn(2,3)
        b1 = np.random.randn(2,1)
        W2 = np.random.randn(3,3)
        b2 = np.random.randn(3,1)
        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        return parameters

    def update_parameters_test_case(self):
        np.random.seed(1)
        W1 = np.random.randn(2,3)
        b1 = np.random.randn(2,1)
        W2 = np.random.randn(3,3)
        b2 = np.random.randn(3,1)

        dW1 = np.random.randn(2,3)
        db1 = np.random.randn(2,1)
        dW2 = np.random.randn(3,3)
        db2 = np.random.randn(3,1)
        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        v = {'dW1': np.array([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.]]), 'dW2': np.array([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.]]), 'db1': np.array([[ 0.],
            [ 0.]]), 'db2': np.array([[ 0.],
            [ 0.],
            [ 0.]])}
        return parameters, grads, v

    def test_initialize_velocity(self):
        parameters = self.initialize_velocity_test_case()
        v = MomentumOperation.initialize_velocity(parameters)
        printer("v[\"dW1\"] =", v["dW1"])
        printer("v[\"db1\"] =", v["db1"])
        printer("v[\"dW2\"] =", v["dW2"])
        printer("v[\"db2\"] =", v["db2"])

    def test_update_parameters(self):
        parameters, grads, v = self.update_parameters_test_case()
        parameters, v = MomentumOperation.update_parameters(parameters, grads, v, beta = 0.9, learning_rate = 0.01)
        printer("W1 =", parameters["W1"])
        printer("b1 =", parameters["b1"])
        printer("W2 =", parameters["W2"])
        printer("b2 =", parameters["b2"])
        printer("v[\"dW1\"] =", v["dW1"])
        printer("v[\"db1\"] =", v["db1"])
        printer("v[\"dW2\"] =", v["dW2"])
        printer("v[\"db2\"] =", v["db2"])


class AdamOperationTest(unittest.TestCase):
    def initialize_adam_test_case(self):
        np.random.seed(1)
        W1 = np.random.randn(2,3)
        b1 = np.random.randn(2,1)
        W2 = np.random.randn(3,3)
        b2 = np.random.randn(3,1)
        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        return parameters

    def update_parameters_test_case(self):
        np.random.seed(1)
        v, s = ({'dW1': np.array([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.]]), 'dW2': np.array([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.]]), 'db1': np.array([[ 0.],
            [ 0.]]), 'db2': np.array([[ 0.],
            [ 0.],
            [ 0.]])}, {'dW1': np.array([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.]]), 'dW2': np.array([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.]]), 'db1': np.array([[ 0.],
            [ 0.]]), 'db2': np.array([[ 0.],
            [ 0.],
            [ 0.]])})
        W1 = np.random.randn(2,3)
        b1 = np.random.randn(2,1)
        W2 = np.random.randn(3,3)
        b2 = np.random.randn(3,1)

        dW1 = np.random.randn(2,3)
        db1 = np.random.randn(2,1)
        dW2 = np.random.randn(3,3)
        db2 = np.random.randn(3,1)

        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return parameters, grads, v, s

    def test_initialize_adam(self):
        parameters = self.initialize_adam_test_case()
        v, s = AdamOperation.initialize_adam(parameters)
        printer("v[\"dW1\"] =", v["dW1"])
        printer("v[\"db1\"] =", v["db1"])
        printer("v[\"dW2\"] =", v["dW2"])
        printer("v[\"db2\"] =", v["db2"])
        printer("s[\"dW1\"] =", s["dW1"])
        printer("s[\"db1\"] =", s["db1"])
        printer("s[\"dW2\"] =", s["dW2"])
        printer("s[\"db2\"] =", s["db2"])

    def test_update_parameters(self):
        parameters, grads, v, s = self.update_parameters_test_case()
        parameters, v, s  = AdamOperation.update_parameters(parameters, grads, v, s, t = 2)
        printer("W1 =", parameters["W1"])
        printer("b1 =", parameters["b1"])
        printer("W2 =", parameters["W2"])
        printer("b2 =", parameters["b2"])
        printer("v[\"dW1\"] =", v["dW1"])
        printer("v[\"db1\"] =", v["db1"])
        printer("v[\"dW2\"] =", v["dW2"])
        printer("v[\"db2\"] =", v["db2"])
        printer("s[\"dW1\"] =", s["dW1"])
        printer("s[\"db1\"] =", s["db1"])
        printer("s[\"dW2\"] =", s["dW2"])
        printer("s[\"db2\"] =", s["db2"])


class OptimizeNNTest(unittest.TestCase):
    def setUp(self):
        self.train_X, self.train_Y = DataSet.load()
        self.learning_rate = 0.0007
        DataSet.show_data(self.train_X, self.train_Y)

    def test_minibatch_gradient_descent(self):
        train_X, train_Y = self.train_X, self.train_Y
        layers_dims = [train_X.shape[0], 5, 2, 1]  # train 3-layer model
        parameters, costs = OptimizeNN.model(train_X, train_Y, layers_dims, optimizer = "gd")
        predictions = OptimizeNN.predict(train_X, train_Y, parameters)
        printer("predictions_train =", predictions)

        DataSet.show_loss(
            costs,
            self.learning_rate,
            'epochs (per 100)',
        )
        DataSet.show_boundary(
            "Model with Gradient Descent optimization",
            train_X,
            train_Y.ravel(),
            parameters,
            OptimizeNN.predict_decision,
        )

    def test_minibatch_gradient_descent_with_momentum(self):
        train_X, train_Y = self.train_X, self.train_Y
        layers_dims = [train_X.shape[0], 5, 2, 1]
        parameters, costs = OptimizeNN.model(train_X, train_Y, layers_dims, beta = 0.9, optimizer = "momentum")
        predictions = OptimizeNN.predict(train_X, train_Y, parameters)
        printer("predictions_train =", predictions)

        DataSet.show_loss(
            costs,
            self.learning_rate,
            'epochs (per 100)',
        )
        DataSet.show_boundary(
            "Model with Momentum Gradient Descent optimization",
            train_X,
            train_Y.ravel(),
            parameters,
            OptimizeNN.predict_decision,
        )

    def test_minibatch_gradient_descent_with_adam(self):
        train_X, train_Y = self.train_X, self.train_Y
        layers_dims = [train_X.shape[0], 5, 2, 1]
        parameters, costs = OptimizeNN.model(train_X, train_Y, layers_dims, optimizer = "adam")
        predictions = OptimizeNN.predict(train_X, train_Y, parameters)
        printer("predictions_train =", predictions)

        DataSet.show_loss(
            costs,
            self.learning_rate,
            'epochs (per 100)',
        )
        DataSet.show_boundary(
            "Model with Adam Gradient Descent optimization",
            train_X,
            train_Y.ravel(),
            parameters,
            OptimizeNN.predict_decision,
        )
