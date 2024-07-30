import unittest
from datasets import HappyDataSet as DataSet
from cnn_keras import KerasCNN


class KerasCNNTest(unittest.TestCase):
    def setUp(self):
        self.X_train_orig, self.Y_train_orig, self.X_test_orig, self.Y_test_orig, self.classes = DataSet.load()
        DataSet.show_face(self.X_train_orig, self.Y_train_orig, 3)

        self.X_train, self.Y_train, self.X_test, self.Y_test = self.scalling_dataset(
            self.X_train_orig, self.Y_train_orig,
            self.X_test_orig, self.Y_test_orig
        )
        self.learning_rate = 0.009
        self.loss_xlable = 'iterations (per tens)'

    def scalling_dataset(self, X_train_orig, Y_train_orig, X_test_orig, Y_test_orig):
        # Normalize image vectors
        X_train = X_train_orig / 255.
        X_test = X_test_orig / 255.
        # Reshape
        Y_train = Y_train_orig.T
        Y_test = Y_test_orig.T
        print("number of training examples = " + str(X_train.shape[0]))
        print("number of test examples = " + str(X_test.shape[0]))
        print("X_train shape: " + str(X_train.shape))
        print("Y_train shape: " + str(Y_train.shape))
        print("X_test shape: " + str(X_test.shape))
        print("Y_test shape: " + str(Y_test.shape))
        print("data classes: ", self.classes)
        return X_train, Y_train, X_test, Y_test

    def test_model(self):
        model = KerasCNN.model(self.X_train, self.Y_train, self.X_test, self.Y_test)
        KerasCNN.summary(model)
        KerasCNN.plot(model, 'datasets/happy_model.png')
