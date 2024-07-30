from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.utils import plot_model


class KerasCNN:
    @classmethod
    def HappyModelSimple(cls, input_shape):
        # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
        X_input = Input(input_shape)

        # Zero-Padding: pads the border of X_input with zeroes
        X = ZeroPadding2D((3, 3))(X_input)

        # CONV -> BN -> RELU Block applied to X
        X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
        X = BatchNormalization(axis = 3, name = 'bn0')(X)
        X = Activation('relu')(X)

        # MAXPOOL
        X = MaxPooling2D((2, 2), name='max_pool')(X)

        # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
        X = Flatten()(X)
        X = Dense(1, activation='sigmoid', name='fc')(X)

        # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
        model = Model(inputs = X_input, outputs = X, name='HappyModel')
        return model

    @classmethod
    def HappyModelImpl(cls, input_shape):
        """
        Implementation of the HappyModel.

        Arguments:
        input_shape -- shape of the images of the dataset

        Returns:
        model -- a Model() instance in Keras
        """
        # Feel free to use the suggested outline in the text above to get started, and run through the whole
        # exercise (including the later portions of this notebook) once. The come back also try out other
        # network architectures as well.
        X_input = Input(input_shape)

        # layer1, result:32*32*4 (CONV -> BN -> RELU -> MAXPOOL Block applied to X)
        X = Conv2D(4, kernel_size=(3,3), strides=(1,1), padding='same')(X_input)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        X = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(X)

        # layer2, result:16*16*8
        X = Conv2D(8, kernel_size=(3,3), strides=(1,1), padding='same')(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        X = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(X)

        # layer3, result:8*8*4
        X = Conv2D(4, kernel_size=(3,3), strides=(1,1), padding='same')(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)

        # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
        X = Flatten()(X)
        X = Dense(4, activation='sigmoid', name='fc1')(X)
        Y = Dense(1, activation='sigmoid', name='fc2')(X)

        # Create model. This creates your Keras model instance.
        model = Model(inputs = X_input, outputs = Y, name='HappyModel')
        return model

    @classmethod
    def model(cls, X_train, Y_train, X_test, Y_test, num_epochs = 20, minibatch_size = 16):
        # step 1, create the model.
        happyModel = cls.HappyModelImpl((64,64,3))
        # step 2, compile the model to configure the learning process.
        happyModel.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
        # step 3, train the model. Choose the number of epochs and the batch size.
        happyModel.fit(x = X_train, y = Y_train, epochs = num_epochs, batch_size = minibatch_size)
        # step 4, test/evaluate the model.
        preds = happyModel.evaluate(x = X_test, y = Y_test)
        print("Loss = " + str(preds[0]))
        print("Test Accuracy = " + str(preds[1]))
        return happyModel

    @classmethod
    def predict(cls, X, model):
        return model.predict(X)

    @classmethod
    def summary(cls, model):
        # Prints the details of your layers in a table with the sizes of its inputs/outputs
        return model.summary()

    @classmethod
    def plot(cls, model, path):
        plot_model(model, to_file=path)
