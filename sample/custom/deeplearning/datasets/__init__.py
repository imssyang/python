import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import sklearn
import sklearn.datasets


class BaseDataSet:
    @classmethod
    def init_plot(cls):
        plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.cmap'] = 'gray'

    @classmethod
    def show_data(cls, X, Y):
        cls.init_plot()
        print(f"X = {X}")
        print(f"Y = {Y}")
        plt.scatter(X[0], X[1], c=Y, s=40, cmap=plt.cm.Spectral)
        plt.show()

    @classmethod
    def show_loss(cls, costs, learning_rate, xlabel):
        cls.init_plot()
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel(xlabel)
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    @classmethod
    def show_boundary(
        cls,
        title,
        X,
        Y,
        parameters,
        predict,
        xlim=(-1.5, 2.5),
        ylim=(-1, 1.5),
    ):
        cls.init_plot()
        # Set min and max values and give it some padding
        x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
        y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
        h = 0.01
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Predict the function value for the whole grid
        model = lambda x: predict(parameters, x.T)
        Z = model(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.title(title)
        axes = plt.gca()
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.scatter(X[0, :], X[1, :], c=Y, cmap=plt.cm.Spectral)
        plt.show()


class CircleDataSet(BaseDataSet):
    @classmethod
    def load(cls):
        np.random.seed(1)
        train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
        np.random.seed(2)
        test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
        train_X = train_X.T
        train_Y = train_Y.reshape((1, train_Y.shape[0]))
        test_X = test_X.T
        test_Y = test_Y.reshape((1, test_Y.shape[0]))
        return train_X, train_Y, test_X, test_Y


class FootballDataSet(BaseDataSet):
    @classmethod
    def load(cls):
        data = scipy.io.loadmat('datasets/football.dat')
        train_X = data['X'].T
        train_Y = data['y'].T
        test_X = data['Xval'].T
        test_Y = data['yval'].T
        return train_X, train_Y, test_X, test_Y


class MoonDataSet(BaseDataSet):
    @classmethod
    def load(cls):
        np.random.seed(3)
        train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2) #300 #0.2
        # Visualize the data
        plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
        train_X = train_X.T
        train_Y = train_Y.reshape((1, train_Y.shape[0]))
        return train_X, train_Y

