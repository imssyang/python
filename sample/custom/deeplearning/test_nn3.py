import unittest
import numpy as np
from utils import printer
from datasets import CircleDataSet as DataSet
from nn3 import BaseOperation, BaseNN


class BaseOperationTest(unittest.TestCase):
    def test_initialize_parameters(self):
        parameters = BaseOperation.initialize_parameters([3, 2, 1], "zeros")
        printer(f"parameters_zeros:", parameters)
        parameters = BaseOperation.initialize_parameters([3, 2, 1], "random")
        printer(f"parameters_random:", parameters)
        parameters = BaseOperation.initialize_parameters([2, 4, 1], "he")
        printer(f"parameters_he:", parameters)


class BaseNNTest(unittest.TestCase):
    def setUp(self):
        self.train_X, self.train_Y, self.test_X, self.test_Y = DataSet.load()
        self.learning_rate = 0.01
        self.loss_xlable = 'iterations (per hundreds)'
        self.axe_xlim = [-1.5, 1.5]
        self.axe_ylim = [-1.5, 1.5]
        DataSet.show_data(self.train_X, self.train_Y)

    def test_parameters_zeros(self):
        """[introduce]
        The performance is really bad, and the cost does not really decrease,
        and the algorithm performs no better than random guessing.
        """
        parameters, costs = BaseNN.model(self.train_X, self.train_Y, initialization = "zeros")
        predictions_train = BaseNN.predict(self.train_X, self.train_Y, parameters)
        printer("predictions_train =", predictions_train)
        predictions_test = BaseNN.predict(self.test_X, self.test_Y, parameters)
        printer("predictions_test =", predictions_test)

        # The performance is really bad, and the cost does not really decrease,
        # and the algorithm performs no better than random guessing.
        DataSet.show_loss(costs, self.learning_rate, self.loss_xlable)

        # The model is predicting 0 for every example.
        # In general, initializing all the weights to zero results in the network failing to break symmetry.
        # This means that every neuron in each layer will learn the same thing, and you might as well be
        # training a neural network with $n^{[l]}=1$ for every layer, and the network is no more powerful
        # than a linear classifier such as logistic regression.
        DataSet.show_boundary(
            "Model with Zeros initialization",
            self.train_X,
            self.train_Y,
            parameters,
            BaseNN.predict_decision,
            self.axe_xlim,
            self.axe_ylim,
        )

    def test_parameters_random(self):
        """[introduce]
        To break symmetry, lets intialize the weights randomly.
        Following random initialization, each neuron can then proceed to learn a different function of its inputs.
        In this exercise, you will see what happens if the weights are intialized randomly, but to very large values.
        """
        parameters, costs = BaseNN.model(self.train_X, self.train_Y, initialization = "random")
        predictions_train = BaseNN.predict(self.train_X, self.train_Y, parameters)
        printer("predictions_train =", predictions_train)
        predictions_test = BaseNN.predict(self.test_X, self.test_Y, parameters)
        printer("predictions_test =", predictions_test)

        # Anyway, it looks like you have broken symmetry, and this gives better results.
        # than before. The model is no longer outputting all 0s.
        # - The cost starts very high. This is because with large random-valued weights, the last activation (sigmoid) outputs
        #   results that are very close to 0 or 1 for some examples, and when it gets that example wrong it incurs a very high loss
        #   for that example. Indeed, when $\log(a^{[3]}) = \log(0)$, the loss goes to infinity.
        # - Poor initialization can lead to vanishing/exploding gradients, which also slows down the optimization algorithm.
        # - If you train this network longer you will see better results, but initializing with overly large random numbers
        #   slows down the optimization.
        DataSet.show_loss(costs, self.learning_rate, self.loss_xlable)

        # **In summary**:
        # - Initializing weights to very large random values does not work well.
        # - Hopefully intializing with small random values does better. The important question is:
        #   how small should be these random values be? Lets find out in the next part!
        DataSet.show_boundary(
            "Model with large random initialization",
            self.train_Y,
            self.train_X,
            parameters,
            BaseNN.predict_decision,
            self.axe_xlim,
            self.axe_ylim,
        )

    def test_parameters_he(self):
        """[summary]
        "He Initialization"; this is named for the first author of He et al., 2015.
        (If you have heard of "Xavier initialization", this is similar except Xavier initialization uses
        a scaling factor for the weights 𝑊[𝑙] of sqrt(1./layers_dims[l-1]) where He initialization would
        use sqrt(2./layers_dims[l-1]).)
        """
        parameters, costs = BaseNN.model(self.train_X, self.train_Y, initialization = "he")
        predictions_train = BaseNN.predict(self.train_X, self.train_Y, parameters)
        printer("predictions_train =", predictions_train)
        predictions_test = BaseNN.predict(self.test_X, self.test_Y, parameters)
        printer("predictions_test =", predictions_test)

        DataSet.show_loss(costs, self.learning_rate, self.loss_xlable)

        # The model with He initialization separates the blue and the red dots very well in a small number of iterations.
        DataSet.show_boundary(
            "Model with He initialization",
            self.train_X,
            self.train_Y,
            parameters,
            BaseNN.predict_decision,
            self.axe_xlim,
            self.axe_ylim,
        )
