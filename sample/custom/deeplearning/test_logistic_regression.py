import unittest
import numpy as np
from utils import printer
from datasets import CatDataSet as DataSet
from logistic_regression import LROperation, LRModel


class LROperationTest(unittest.TestCase):
    def test_initialize_parameters(self):
        # initialized your parameters.
        dim = 2
        w, b = LROperation.initialize_with_zeros(dim)
        print("w = " + str(w)) # [[0.], [0.]]
        print("b = " + str(b)) # 0

    def test_propagate(self):
        # compute a cost function and its gradient.
        w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
        grads, cost = LROperation.propagate(w, b, X, Y)
        print("dw = " + str(grads["dw"])) # [[0.99993216], [1.99980262]]
        print("db = " + str(grads["db"])) # 0.49993523062470574
        print("cost = " + str(cost))      # 6.000064773192205

    def test_gradient_descent(self):
        # update the parameters using gradient descent.
        w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
        params, grads, costs = LROperation.optimize(
            w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False
        )
        print("w = " + str(params["w"]))  # [[0.1124579 ], [0.23106775]]
        print("b = " + str(params["b"]))  # 1.5593049248448891
        print("dw = " + str(grads["dw"])) # [[0.90158428], [1.76250842]]
        print("db = " + str(grads["db"])) # 0.4304620716786828
        print("costs = " + str(costs))    # [6.000064773192205]

        # use w and b to predict the labels for a dataset X.
        w = params["w"]
        b = params["b"]
        print("predictions = " + str(LROperation.predict(w, b, X))) # [[1, 1]]


class LRModelTest(unittest.TestCase):
    def setUp(self):
        self.train_set_x_orig, self.train_set_y, self.test_set_x_orig, self.test_set_y, self.classes = DataSet.load()
        DataSet.show_cat(self.train_set_x_orig, self.train_set_y, self.classes, 10)
        DataSet.show_cat(self.test_set_x_orig, self.test_set_y, self.classes, 1)
        self.train_set_x, self.train_set_y, self.test_set_x, self.test_set_y = self.preprocessing(
            self.train_set_x_orig, self.train_set_y, self.test_set_x_orig, self.test_set_y
        )
        self.learning_rate = 0.005
        self.loss_xlable = 'iterations (per hundreds)'

    def preprocessing(self, train_set_x_orig, train_set_y, test_set_x_orig, test_set_y):
        # Common steps for pre-processing a new dataset are:
        # 1. Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
        # 2. Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)
        # 3. To center and "Standardize" the data (color from 0 to 255)
        train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T # (209, 64, 64, 3) -> (12288, 209)
        test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T    # (50, 64, 64, 3) -> (12288, 50)
        print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape)) # (12288, 209)
        print("train_set_y shape: " + str(train_set_y.shape)) # (1, 209)
        print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))   # (12288, 50)
        print("test_set_y shape: " + str(test_set_y.shape))   # (1, 50)
        print("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0])) # [17 31 56 22 33]
        train_set_x = train_set_x_flatten/255.
        test_set_x = test_set_x_flatten/255.
        return train_set_x, train_set_y, test_set_x, test_set_y

    def test_model(self):
        d = LRModel.model(
            self.train_set_x, self.train_set_y, self.test_set_x, self.test_set_y,
            num_iterations = 2000, learning_rate = self.learning_rate, print_cost = True
        )
        # train accuracy: 99.04306220095694 %
        # test accuracy: 70.0 %

        # You can see the cost decreasing. It shows that the parameters are being learned.
        DataSet.show_loss(d['costs'], self.learning_rate, self.loss_xlable)

        # We preprocess the image to fit your algorithm.
        image, image64 = DataSet.load_image("datasets/cat_iran.jpg")
        image_predicted = LRModel.predict(d["w"], d["b"], image64)
        print("y = " + str(np.squeeze(image_predicted))
              + ", algorithm predicts a \""
              + self.classes[int(np.squeeze(image_predicted)),].decode("utf-8")
              +  "\" picture."
        )
        DataSet.show_image(image)

    def test_learning_rates(self):
        # The learning rate 𝛼 determines how rapidly we update the parameters.
        # If the learning rate is too large we may "overshoot" the optimal value. Similarly,
        # if it is too small we will need too many iterations to converge to the best values.
        learning_rates = [0.01, 0.001, 0.0001]
        models = {}
        for i in learning_rates:
            print("\nlearning rate is: " + str(i))
            models[str(i)] = LRModel.model(
                self.train_set_x, self.train_set_y, self.test_set_x, self.test_set_y,
                num_iterations = 1500, learning_rate = i, print_cost = False,
            )

        # Different learning rates give different costs and thus different predictions results.
        # If the learning rate is too large (0.01), the cost may oscillate up and down.
        # A lower cost doesn't mean a better model. You have to check if there is possibly overfitting.
        DataSet.show_learning_rates(learning_rates, models)

