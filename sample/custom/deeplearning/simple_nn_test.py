import pprint
import unittest
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from simple_nn import OperationNN, SimpleNN


def printer(*args, **kwargs):
    def numpy2str(data):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    data[key] = value.tolist()
        return data

    pp = pprint.PrettyPrinter(indent=1, width=80, compact=False)
    kvargs = dict()
    for key, value in kwargs.items():
        kvargs[key] = numpy2str(value)
    argl = list()
    for arg in args:
        argl.append(numpy2str(arg))
    if len(argl) > 0 and len(kvargs) > 0:
        argl.append(kvargs)
        pp.pprint(argl)
    elif len(argl) > 0:
        pp.pprint(argl)
    else:
        pp.pprint(kvargs)


class DataSet:
    def __init__(self):
        self.train_X, self.train_Y, self.test_X, self.test_Y = self.load()

    def load(self):
        np.random.seed(1)
        train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
        np.random.seed(2)
        test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
        train_X = train_X.T
        train_Y = train_Y.reshape((1, train_Y.shape[0]))
        test_X = test_X.T
        test_Y = test_Y.reshape((1, test_Y.shape[0]))
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
        plt.xlabel('iterations (per hundreds)')
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
        axes.set_xlim([-1.5,1.5])
        axes.set_ylim([-1.5,1.5])
        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.scatter(X[0, :], X[1, :], c=Y, cmap=plt.cm.Spectral)
        plt.show()


class OperationNNTest(unittest.TestCase):
    def test_initialize_parameters(self):
        parameters = OperationNN.initialize_parameters([3, 2, 1], "zeros")
        printer(f"parameters_zeros:", parameters)
        parameters = OperationNN.initialize_parameters([3, 2, 1], "random")
        printer(f"parameters_random:", parameters)
        parameters = OperationNN.initialize_parameters([2, 4, 1], "he")
        printer(f"parameters_he:", parameters)


class SimpleNNTest(unittest.TestCase):
    def setUp(self):
        self.dataset = DataSet()
        self.dataset.show(self.dataset.train_X, self.dataset.train_Y)

    def test_parameters_zeros(self):
        """[introduce]
        The performance is really bad, and the cost does not really decrease,
        and the algorithm performs no better than random guessing.
        """
        parameters, costs = SimpleNN.model(
            self.dataset.train_X,
            self.dataset.train_Y,
            initialization = "zeros"
        )
        predictions_train = SimpleNN.predict(
            self.dataset.train_X,
            self.dataset.train_Y,
            parameters
        )
        printer("predictions_train =", predictions_train)
        predictions_test = SimpleNN.predict(
            self.dataset.test_X,
            self.dataset.test_Y,
            parameters
        )
        printer("predictions_test =", predictions_test)

        # The performance is really bad, and the cost does not really decrease,
        # and the algorithm performs no better than random guessing.
        self.dataset.show_loss(costs, learning_rate = 0.01)

        # The model is predicting 0 for every example.
        # In general, initializing all the weights to zero results in the network failing to break symmetry.
        # This means that every neuron in each layer will learn the same thing, and you might as well be
        # training a neural network with $n^{[l]}=1$ for every layer, and the network is no more powerful
        # than a linear classifier such as logistic regression.
        self.dataset.show_boundary(
            "Model with Zeros initialization",
            self.dataset.train_X,
            self.dataset.train_Y,
            lambda x: SimpleNN.predict_decision(parameters, x.T),
        )

    def test_parameters_random(self):
        """[introduce]
        To break symmetry, lets intialize the weights randomly.
        Following random initialization, each neuron can then proceed to learn a different function of its inputs.
        In this exercise, you will see what happens if the weights are intialized randomly, but to very large values.
        """
        parameters, costs = SimpleNN.model(self.dataset.train_X, self.dataset.train_Y, initialization = "random")
        predictions_train = SimpleNN.predict(self.dataset.train_X, self.dataset.train_Y, parameters)
        printer("predictions_train =", predictions_train)
        predictions_test = SimpleNN.predict(self.dataset.test_X, self.dataset.test_Y, parameters)
        printer("predictions_test =", predictions_test)

        # Anyway, it looks like you have broken symmetry, and this gives better results.
        # than before. The model is no longer outputting all 0s.
        # - The cost starts very high. This is because with large random-valued weights, the last activation (sigmoid) outputs
        #   results that are very close to 0 or 1 for some examples, and when it gets that example wrong it incurs a very high loss
        #   for that example. Indeed, when $\log(a^{[3]}) = \log(0)$, the loss goes to infinity.
        # - Poor initialization can lead to vanishing/exploding gradients, which also slows down the optimization algorithm.
        # - If you train this network longer you will see better results, but initializing with overly large random numbers
        #   slows down the optimization.
        self.dataset.show_loss(costs, learning_rate = 0.01)

        # **In summary**:
        # - Initializing weights to very large random values does not work well.
        # - Hopefully intializing with small random values does better. The important question is:
        #   how small should be these random values be? Lets find out in the next part!
        self.dataset.show_boundary(
            "Model with large random initialization",
            self.dataset.train_X,
            self.dataset.train_Y,
            lambda x: SimpleNN.predict_decision(parameters, x.T),
        )

    def test_parameters_he(self):
        """[summary]
        "He Initialization"; this is named for the first author of He et al., 2015.
        (If you have heard of "Xavier initialization", this is similar except Xavier initialization uses
        a scaling factor for the weights 𝑊[𝑙] of sqrt(1./layers_dims[l-1]) where He initialization would
        use sqrt(2./layers_dims[l-1]).)
        """
        parameters, costs = SimpleNN.model(self.dataset.train_X, self.dataset.train_Y, initialization = "he")
        predictions_train = SimpleNN.predict(self.dataset.train_X, self.dataset.train_Y, parameters)
        printer("predictions_train =", predictions_train)
        predictions_test = SimpleNN.predict(self.dataset.test_X, self.dataset.test_Y, parameters)
        printer("predictions_test =", predictions_test)

        self.dataset.show_loss(costs, learning_rate = 0.01)

        # The model with He initialization separates the blue and the red dots very well in a small number of iterations.
        self.dataset.show_boundary(
            "Model with He initialization",
            self.dataset.train_X,
            self.dataset.train_Y,
            lambda x: SimpleNN.predict_decision(parameters, x.T),
        )
