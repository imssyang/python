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
        if parameters:
            model = lambda x: predict(parameters, x.T)
            if xlim or ylim:
                axes = plt.gca()
                axes.set_xlim(xlim)
                axes.set_ylim(ylim)
        else:
            model = lambda x: predict(x)
        Z = model(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        plt.title(title)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.scatter(X[0, :], X[1, :], c=Y, cmap=plt.cm.Spectral)
        plt.show()

    @classmethod
    def load_image(cls, path):
        image = np.array(imageio.imread(path))
        image64 = np.array(
            Image.fromarray(image).resize(size=(64,64)),
        ).reshape((1, 64*64*3)).T
        return image, image64

    @classmethod
    def show_image(cls, image):
        plt.imshow(image)
        plt.show()


class CatDataSet(BaseDataSet):
    @classmethod
    def load(cls):
        # A training set of m_train images labeled as cat (y=1) or non-cat (y=0)
        # A test set of m_test images labeled as cat or non-cat.
        # Each image is of shape (num_px=64, num_px=64, 3) where 3 is for the 3 channels (RGB).
        # Thus, each image is square (height = num_px) and (width = num_px).
        train_dataset = h5py.File('datasets/cat_trains.h5', "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
        train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

        test_dataset = h5py.File('datasets/cat_tests.h5', "r")
        test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
        test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

        classes = np.array(test_dataset["list_classes"][:]) # the list of classes
        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
        cls.show_info(train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig)
        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

    @classmethod
    def show_info(cls, train_X, train_Y, test_X, test_Y):
        num_train = train_X.shape[0]
        num_test = test_X.shape[0]
        num_px = train_X.shape[1]
        print("Number of training examples: num_train = " + str(num_train)) # 209
        print("Number of testing examples: num_test = " + str(num_test))    # 50
        print("Height/Width of each image: num_px = " + str(num_px))        # 64
        print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)") # (64, 64, 3)
        print("train_set_x shape: " + str(train_X.shape))  # (209, 64, 64, 3)
        print("train_set_y shape: " + str(train_Y.shape))  # (1, 209)
        print("test_set_x shape: " + str(test_X.shape))    # (50, 64, 64, 3)
        print("test_set_y shape: " + str(test_Y.shape))    # (1, 50)

    @classmethod
    def show_cat(cls, X, Y, classes, index):
        print("y = " + str(Y[:, index]) + ", it's a '" + classes[np.squeeze(Y[:, index])].decode("utf-8") +  "' picture.")
        plt.imshow(X[index])
        plt.show()

    @classmethod
    def show_learning_rates(cls, learning_rates, models):
        plt.ylabel('cost')
        plt.xlabel('iterations')

        for i in learning_rates:
            plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

        legend = plt.legend(loc='upper center', shadow=True)
        frame = legend.get_frame()
        frame.set_facecolor('0.90')
        plt.show()


class FlowerDataSet(BaseDataSet):
    @classmethod
    def load(cls):
        # The data looks like a "flower" with some red (label y=0) and some blue (y=1) points.
        np.random.seed(1)
        m = 400      # number of examples
        N = int(m/2) # number of points per class
        D = 2        # dimensionality
        X = np.zeros((m,D)) # data matrix where each row is a single example
        Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
        a = 4        # maximum ray of the flower
        for j in range(2):
            ix = range(N*j,N*(j+1))
            t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
            r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
            X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
            Y[ix] = j
        X = X.T
        Y = Y.T
        cls.show_info(X, Y)
        return X, Y

    @classmethod
    def show_info(cls, X, Y):
        shape_X = X.shape
        shape_Y = Y.shape
        m = shape_X[1]  # training set size
        print('The shape of X is: ' + str(shape_X))
        print('The shape of Y is: ' + str(shape_Y))
        print('I have m = %d training examples!' % (m))

    @classmethod
    def add_plot_boundary(cls, model, X, y):
        # Set min and max values and give it some padding
        x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
        y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
        h = 0.01
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Predict the function value for the whole grid
        Z = model(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)



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
