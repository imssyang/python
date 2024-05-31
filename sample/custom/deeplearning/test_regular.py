import unittest
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from utils import printer
from nn_regular import L2Operation, DropoutOperation, RegularNN


class DataSet:
    def __init__(self):
        self.train_X, self.train_Y, self.test_X, self.test_Y = self.load()

    def load(self):
        data = scipy.io.loadmat('datasets/data.mat')
        train_X = data['X'].T
        train_Y = data['y'].T
        test_X = data['Xval'].T
        test_Y = data['yval'].T
        return train_X, train_Y, test_X, test_Y

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
        plt.xlabel('iterations (x1,000)')
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
        axes.set_xlim([-0.75,0.40])
        axes.set_ylim([-0.75,0.65])
        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.scatter(X[0, :], X[1, :], c=Y, cmap=plt.cm.Spectral)
        plt.show()


class L2OperationTest(unittest.TestCase):
    def compute_cost_test_case(self):
        np.random.seed(1)
        Y_assess = np.array([[1, 1, 0, 1, 0]])
        W1 = np.random.randn(2, 3)
        b1 = np.random.randn(2, 1)
        W2 = np.random.randn(3, 2)
        b2 = np.random.randn(3, 1)
        W3 = np.random.randn(1, 3)
        b3 = np.random.randn(1, 1)
        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
        a3 = np.array([[ 0.40682402,  0.01629284,  0.16722898,  0.10118111,  0.40682402]])
        return a3, Y_assess, parameters

    def backward_propagation_test_case(self):
        np.random.seed(1)
        X_assess = np.random.randn(3, 5)
        Y_assess = np.array([[1, 1, 0, 1, 0]])
        cache = (np.array([[-1.52855314,  3.32524635,  2.13994541,  2.60700654, -0.75942115],
                        [-1.98043538,  4.1600994 ,  0.79051021,  1.46493512, -0.45506242]]),
                np.array([[ 0.        ,  3.32524635,  2.13994541,  2.60700654,  0.        ],
                        [ 0.        ,  4.1600994 ,  0.79051021,  1.46493512,  0.        ]]),
                np.array([[-1.09989127, -0.17242821, -0.87785842],
                        [ 0.04221375,  0.58281521, -1.10061918]]),
                np.array([[ 1.14472371],
                        [ 0.90159072]]),
                np.array([[ 0.53035547,  5.94892323,  2.31780174,  3.16005701,  0.53035547],
                        [-0.69166075, -3.47645987, -2.25194702, -2.65416996, -0.69166075],
                        [-0.39675353, -4.62285846, -2.61101729, -3.22874921, -0.39675353]]),
                np.array([[ 0.53035547,  5.94892323,  2.31780174,  3.16005701,  0.53035547],
                        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
                        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ]]),
                np.array([[ 0.50249434,  0.90085595],
                        [-0.68372786, -0.12289023],
                        [-0.93576943, -0.26788808]]),
                np.array([[ 0.53035547],
                        [-0.69166075],
                        [-0.39675353]]),
                np.array([[-0.3771104 , -4.10060224, -1.60539468, -2.18416951, -0.3771104 ]]),
                np.array([[ 0.40682402,  0.01629284,  0.16722898,  0.10118111,  0.40682402]]),
                np.array([[-0.6871727 , -0.84520564, -0.67124613]]),
                np.array([[-0.0126646]]))
        return X_assess, Y_assess, cache

    def test_common(self):
        A3, Y_assess, parameters = self.compute_cost_test_case()
        cost = L2Operation.compute_loss(A3, Y_assess, parameters, lambd = 0.1)
        printer("cost =", cost)

        X_assess, Y_assess, cache = self.backward_propagation_test_case()
        grads = L2Operation.backward_propagation(X_assess, Y_assess, cache, lambd = 0.7)
        printer("dW1 =", grads["dW1"])
        printer("dW2 =", grads["dW2"])
        printer("dW3 =", grads["dW3"])


class DropoutOperationTest(unittest.TestCase):
    def forward_propagation_test_case(self):
        np.random.seed(1)
        X_assess = np.random.randn(3, 5)
        W1 = np.random.randn(2, 3)
        b1 = np.random.randn(2, 1)
        W2 = np.random.randn(3, 2)
        b2 = np.random.randn(3, 1)
        W3 = np.random.randn(1, 3)
        b3 = np.random.randn(1, 1)
        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
        return X_assess, parameters

    def backward_propagation_test_case(self):
        np.random.seed(1)
        X_assess = np.random.randn(3, 5)
        Y_assess = np.array([[1, 1, 0, 1, 0]])
        cache = (np.array([[-1.52855314,  3.32524635,  2.13994541,  2.60700654, -0.75942115],
            [-1.98043538,  4.1600994 ,  0.79051021,  1.46493512, -0.45506242]]), np.array([[ True, False,  True,  True,  True],
            [ True,  True,  True,  True, False]], dtype=bool), np.array([[ 0.        ,  0.        ,  4.27989081,  5.21401307,  0.        ],
            [ 0.        ,  8.32019881,  1.58102041,  2.92987024,  0.        ]]), np.array([[-1.09989127, -0.17242821, -0.87785842],
            [ 0.04221375,  0.58281521, -1.10061918]]), np.array([[ 1.14472371],
            [ 0.90159072]]), np.array([[ 0.53035547,  8.02565606,  4.10524802,  5.78975856,  0.53035547],
            [-0.69166075, -1.71413186, -3.81223329, -4.61667916, -0.69166075],
            [-0.39675353, -2.62563561, -4.82528105, -6.0607449 , -0.39675353]]), np.array([[ True, False,  True, False,  True],
            [False,  True, False,  True,  True],
            [False, False,  True, False, False]], dtype=bool), np.array([[ 1.06071093,  0.        ,  8.21049603,  0.        ,  1.06071093],
            [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ]]), np.array([[ 0.50249434,  0.90085595],
            [-0.68372786, -0.12289023],
            [-0.93576943, -0.26788808]]), np.array([[ 0.53035547],
            [-0.69166075],
            [-0.39675353]]), np.array([[-0.7415562 , -0.0126646 , -5.65469333, -0.0126646 , -0.7415562 ]]), np.array([[ 0.32266394,  0.49683389,  0.00348883,  0.49683389,  0.32266394]]), np.array([[-0.6871727 , -0.84520564, -0.67124613]]), np.array([[-0.0126646]]))
        return X_assess, Y_assess, cache

    def test_common(self):
        X_assess, parameters = self.forward_propagation_test_case()
        A3, cache = DropoutOperation.forward_propagation(X_assess, parameters, keep_prob = 0.7)
        printer("A3 =", A3)

        X_assess, Y_assess, cache = self.backward_propagation_test_case()
        gradients = DropoutOperation.backward_propagation(X_assess, Y_assess, cache, keep_prob = 0.8)
        printer("dA1 =", gradients["dA1"])
        printer("dA2 =", gradients["dA2"])


class RegularNNTest(unittest.TestCase):
    def setUp(self):
        self.dataset = DataSet()
        self.dataset.show(self.dataset.train_X, self.dataset.train_Y)

    def test_without_regularization(self):
        parameters, costs = RegularNN.model(
            self.dataset.train_X,
            self.dataset.train_Y,
        )
        predictions_train = RegularNN.predict(
            self.dataset.train_X,
            self.dataset.train_Y,
            parameters
        )
        printer("predictions_train =", predictions_train)
        predictions_test = RegularNN.predict(
            self.dataset.test_X,
            self.dataset.test_Y,
            parameters
        )
        printer("predictions_test =", predictions_test)

        self.dataset.show_loss(costs, learning_rate = 0.3)

        # The non-regularized model is obviously overfitting the training set.
        # It is fitting the noisy points!.
        self.dataset.show_boundary(
            "Model without regularization",
            self.dataset.train_X,
            self.dataset.train_Y.ravel(),
            lambda x: RegularNN.predict_decision(parameters, x.T),
        )

    def test_l2_regularization(self):
        """[introduce]
        The standard way to avoid overfitting is called L2 regularization.
        It consists of appropriately modifying your cost function.
        L2-regularization relies on the assumption that a model with small weights
        is simpler than a model with large weights.
        """
        parameters, costs = RegularNN.model(
            self.dataset.train_X,
            self.dataset.train_Y,
            lambd = 0.7,
        )
        predictions_train = RegularNN.predict(
            self.dataset.train_X,
            self.dataset.train_Y,
            parameters,
        )
        printer("predictions_train =", predictions_train)
        predictions_test = RegularNN.predict(
            self.dataset.test_X,
            self.dataset.test_Y,
            parameters
        )
        printer("predictions_test =", predictions_test)

        self.dataset.show_loss(costs, learning_rate = 0.3)

        # You are not overfitting the training data anymore.
        self.dataset.show_boundary(
            "Model with L2-regularization",
            self.dataset.train_X,
            self.dataset.train_Y.ravel(),
            lambda x: RegularNN.predict_decision(parameters, x.T),
        )

    def test_dropout_regularization(self):
        """[introduce]
        dropout is a widely used regularization technique that is specific to deep learning.
        It randomly shuts down some neurons in each iteration.
        - A common mistake when using dropout is to use it both in training and testing.
          You should use dropout (randomly eliminate nodes) only in training.
        """
        parameters, costs = RegularNN.model(
            self.dataset.train_X,
            self.dataset.train_Y,
            keep_prob = 0.86,
            learning_rate = 0.3,
        )
        predictions_train = RegularNN.predict(
            self.dataset.train_X,
            self.dataset.train_Y,
            parameters,
        )
        printer("predictions_train =", predictions_train)
        predictions_test = RegularNN.predict(
            self.dataset.test_X,
            self.dataset.test_Y,
            parameters
        )
        printer("predictions_test =", predictions_test)

        # The test accuracy has increased again (to 95%)! Your model is not overfitting the training set
        # and does a great job on the test set.
        self.dataset.show_loss(costs, learning_rate = 0.3)

        self.dataset.show_boundary(
            "Model with L2-regularization",
            self.dataset.train_X,
            self.dataset.train_Y.ravel(),
            lambda x: RegularNN.predict_decision(parameters, x.T),
        )
