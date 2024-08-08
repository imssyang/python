import unittest
import numpy as np
from utils import printer
from datasets import SignDataSet as DataSet
from cnn_resnet import ResnetModel


class ResnetModelTest(unittest.TestCase):
    def setUp(self):
        self.X_train_orig, self.Y_train_orig, self.X_test_orig, self.Y_test_orig, self.classes = DataSet.load()
        DataSet.show_sign(self.X_train_orig, self.Y_train_orig, 3)

        self.X_train, self.Y_train, self.X_test, self.Y_test = self.scalling_dataset(
            self.X_train_orig, self.Y_train_orig,
            self.X_test_orig, self.Y_test_orig
        )

    def convert_to_one_hot(self, Y, C):
        Y = np.eye(C)[Y.reshape(-1)].T
        return Y

    def scalling_dataset(self, X_train_orig, Y_train_orig, X_test_orig, Y_test_orig):
        # Normalize image vectors
        X_train = X_train_orig / 255.
        X_test = X_test_orig / 255.
        # Convert training and test labels to one hot matrices
        Y_train = self.convert_to_one_hot(Y_train_orig, 6).T
        Y_test = self.convert_to_one_hot(Y_test_orig, 6).T
        print("number of training examples = " + str(X_train.shape[0]))
        print("number of test examples = " + str(X_test.shape[0]))
        print("X_train shape: " + str(X_train.shape))
        print("Y_train shape: " + str(Y_train.shape))
        print("X_test shape: " + str(X_test.shape))
        print("Y_test shape: " + str(Y_test.shape))
        print("data classes: ", self.classes)
        return X_train, Y_train, X_test, Y_test

    def test_model(self):
        model = ResnetModel.model(self.X_train, self.Y_train, self.X_test, self.Y_test)
        ResnetModel.summary(model)
        ResnetModel.plot(model, 'datasets/resnet50.png')

        image, image64 = DataSet.load_two_image(flatten=False)
        image_prediction = ResnetModel.predict(image64, model)
        print("Model predicts: y = " + str(np.squeeze(image_prediction)))
        DataSet.show_image(image)
