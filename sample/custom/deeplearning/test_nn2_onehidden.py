import unittest
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
from utils import printer
from datasets import FlowerDataSet as DataSet
from nn2_onehidden import OnehiddenOperation, OnehiddenNN


class OnehiddenOperationTest(unittest.TestCase):
    def test_layer_sizes(self):
        X_assess, Y_assess = layer_sizes_test_case()
        (n_x, n_h, n_y) = OnehiddenOperation.layer_sizes(X_assess, Y_assess)
        print("The size of the input layer is: n_x = " + str(n_x))
        print("The size of the hidden layer is: n_h = " + str(n_h))
        print("The size of the output layer is: n_y = " + str(n_y))


class OnehiddenNNTest(unittest.TestCase):
    def setUp(self):
        self.X, self.Y = DataSet.load()
        self.learning_rate = 0.01
        self.loss_xlable = 'iterations (per hundreds)'
        DataSet.show_data(self.X, self.Y)

    def test_logistic_regression(self):
        # Train the logistic regression classifier
        clf = sklearn.linear_model.LogisticRegressionCV()
        clf.fit(self.X.T, self.Y.T.ravel())

        # Print accuracy
        LR_predictions = clf.predict(self.X.T)
        print('Accuracy of logistic regression: %d '
            % float((np.dot(self.Y, LR_predictions) + np.dot(1-self.Y, 1-LR_predictions)) / float(self.Y.size)*100)
            + '% '
            + "(percentage of correctly labelled datapoints)")
        DataSet.show_boundary(
            "Logistic Regression",
            self.X,
            self.Y.ravel(),
            None,
            clf.predict,
        )

    def test_model(self):
        parameters = OnehiddenNN.model(self.X, self.Y, n_h = 4, num_iterations = 10000, print_cost=True)

        # Accuracy is really high compared to Logistic Regression.
        # Neural networks are able to learn even highly non-linear decision boundaries, unlike logistic regression.
        predictions = OnehiddenNN.predict(parameters, self.X)
        print('Accuracy: %d'
            % float((np.dot(self.Y, predictions.T) + np.dot(1-self.Y, 1-predictions.T))/float(self.Y.size)*100)
            + '%'
        )
        DataSet.show_boundary(
            "Decision Boundary for hidden layer size " + str(4),
            self.X,
            self.Y.ravel(),
            parameters,
            OnehiddenNN.predict,
            xlim=None,
            ylim=None,
        )

    def test_hidden_layer_sizes(self):
        # The larger models (with more hidden units) are able to fit the training set better, until eventually the largest models overfit the data.
        # The best hidden layer size seems to be around n_h = 5. Indeed, a value around here seems to fits the data well without also incurring noticable overfitting.
        plt.figure(figsize=(16, 32))
        hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
        for i, n_h in enumerate(hidden_layer_sizes):
            plt.subplot(5, 2, i+1)
            plt.title('Hidden Layer of size %d' % n_h)
            parameters = OnehiddenNN.model(self.X, self.Y, n_h, num_iterations = 5000)
            DataSet.add_plot_boundary(lambda x: OnehiddenNN.predict(parameters, x.T), self.X, self.Y.ravel())

            predictions = OnehiddenNN.predict(parameters, self.X)
            accuracy = float((np.dot(self.Y, predictions.T) + np.dot(1-self.Y, 1-predictions.T))/float(self.Y.size)*100)
            print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
        plt.show()
