import h5py
import numpy as np
import matplotlib.pyplot as plt
import imageio
import scipy.io
import sklearn
import sklearn.datasets
from PIL import Image


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


class SignDataSet(BaseDataSet):
    @classmethod
    def load(cls):
        # Training set: 1080 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (180 pictures per number).
        train_dataset = h5py.File('datasets/sign_trains.h5', "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
        train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

        # Test set: 120 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (20 pictures per number).
        test_dataset = h5py.File('datasets/sign_tests.h5', "r")
        test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
        test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

        classes = np.array(test_dataset["list_classes"][:]) # the list of classes
        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

    @classmethod
    def load_thumb_image(cls):
        image_path = "datasets/sign_thumb.jpg"
        image = np.array(imageio.imread(image_path))
        image64 = np.array(
            Image.fromarray(image).resize(size=(64,64)),
        ).reshape((1, 64*64*3)).T
        return image, image64

    @classmethod
    def show_sign(cls, X, Y, index):
        print("y = " + str(np.squeeze(Y[:, index])))
        plt.imshow(X[index])
        plt.show()

    @classmethod
    def show_image(cls, image):
        plt.imshow(image)
        plt.show()
