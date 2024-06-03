import unittest
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from utils import printer
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


class DataSet:
    def __init__(self):
        self.train_X, self.train_Y = self.load()

    def load(self):
        np.random.seed(3)
        train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2) #300 #0.2
        # Visualize the data
        plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
        train_X = train_X.T
        train_Y = train_Y.reshape((1, train_Y.shape[0]))
        return train_X, train_Y

    def show(self, X, Y):
        print(f"X = {X}")
        print(f"Y = {Y}")
        plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.cmap'] = 'gray'
        plt.scatter(X[0], X[1], c=Y, s=40, cmap=plt.cm.Spectral);
        plt.show()

    def show_loss(self, costs, learning_rate):
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs (per 100)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    def show_boundary(self, title, X, Y, model):
        # Set min and max values and give it some padding
        x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
        y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
        h = 0.01
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Predict the function value for the whole grid
        Z = model(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.title(title)
        axes = plt.gca()
        axes.set_xlim([-1.5,2.5])
        axes.set_ylim([-1,1.5])
        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.scatter(X[0, :], X[1, :], c=Y, cmap=plt.cm.Spectral)
        plt.show()


class OptimizeNNTest(unittest.TestCase):
    def setUp(self):
        self.dataset = DataSet()
        self.dataset.show(self.dataset.train_X, self.dataset.train_Y)

    def test_minibatch_gradient_descent(self):
        train_X, train_Y = self.dataset.train_X, self.dataset.train_Y

        # train 3-layer model
        layers_dims = [train_X.shape[0], 5, 2, 1]
        parameters, costs = OptimizeNN.model(train_X, train_Y, layers_dims, optimizer = "gd")
        predictions = OptimizeNN.predict(train_X, train_Y, parameters)
        printer("predictions_train =", predictions)

        self.dataset.show_loss(costs, learning_rate = 0.0007)
        self.dataset.show_boundary(
            "Model with Gradient Descent optimization",
            train_X,
            train_Y.ravel(),
            lambda x: OptimizeNN.predict_decision(parameters, x.T),
        )

    def test_minibatch_gradient_descent_with_momentum(self):
        train_X, train_Y = self.dataset.train_X, self.dataset.train_Y

        # train 3-layer model
        layers_dims = [train_X.shape[0], 5, 2, 1]
        parameters, costs = OptimizeNN.model(train_X, train_Y, layers_dims, beta = 0.9, optimizer = "momentum")
        predictions = OptimizeNN.predict(train_X, train_Y, parameters)
        printer("predictions_train =", predictions)

        self.dataset.show_loss(costs, learning_rate = 0.0007)
        self.dataset.show_boundary(
            "Model with Gradient Descent optimization",
            train_X,
            train_Y.ravel(),
            lambda x: OptimizeNN.predict_decision(parameters, x.T),
        )

    def test_minibatch_gradient_descent_with_adam(self):
        train_X, train_Y = self.dataset.train_X, self.dataset.train_Y

        # train 3-layer model
        layers_dims = [train_X.shape[0], 5, 2, 1]
        parameters, costs = OptimizeNN.model(train_X, train_Y, layers_dims, optimizer = "adam")
        predictions = OptimizeNN.predict(train_X, train_Y, parameters)
        printer("predictions_train =", predictions)

        self.dataset.show_loss(costs, learning_rate = 0.0007)
        self.dataset.show_boundary(
            "Model with Gradient Descent optimization",
            train_X,
            train_Y.ravel(),
            lambda x: OptimizeNN.predict_decision(parameters, x.T),
        )
