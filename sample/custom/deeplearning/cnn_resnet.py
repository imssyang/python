import tensorflow as tf
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.utils import plot_model
import keras.backend as K


K.set_image_data_format('channels_last')


class ResnetBlock:

    @classmethod
    def identity(cls, X, f, filters, stage, block):
        """
        Implementation of the identity block as defined in Figure 3

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network

        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2, F3 = filters

        # Save the input value. You'll need this later to add back to the main path. 
        X_shortcut = X

        # First component of main path
        X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of main path (≈3 lines)
        X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path (≈2 lines)
        X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)
        return X

    @classmethod
    def convolutional(cls, X, f, filters, stage, block, s = 2):
        """
        Implementation of the convolutional block as defined in Figure 4

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        s -- Integer, specifying the stride to be used

        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2, F3 = filters

        # Save the input value
        X_shortcut = X

        ##### MAIN PATH #####
        # First component of main path
        X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of main path (≈3 lines)
        X = Conv2D(F2, (f, f), strides = (1,1), padding='same', name = conv_name_base + '2b', kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path (≈2 lines)
        X = Conv2D(F3, (1, 1), strides = (1,1), name = conv_name_base + '2c', kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

        ##### SHORTCUT PATH #### (≈2 lines)
        X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1', kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)
        return X


class ResnetModel:
    @classmethod
    def ResNet50Impl(cls, input_shape = (64, 64, 3), classes = 6):
        """
        Implementation of the popular ResNet50 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

        Arguments:
        input_shape -- shape of the images of the dataset
        classes -- integer, number of classes

        Returns:
        model -- a Model() instance in Keras
        """
        # Define the input as a tensor with shape input_shape
        X_input = Input(input_shape)

        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X_input)

        # Stage 1
        X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        # Stage 2
        X = ResnetBlock.convolutional(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
        X = ResnetBlock.identity(X, 3, [64, 64, 256], stage=2, block='b')
        X = ResnetBlock.identity(X, 3, [64, 64, 256], stage=2, block='c')

        # Stage 3 (≈4 lines)
        X = ResnetBlock.convolutional(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
        X = ResnetBlock.identity(X, 3, [128, 128, 512], stage=3, block='b')
        X = ResnetBlock.identity(X, 3, [128, 128, 512], stage=3, block='c')
        X = ResnetBlock.identity(X, 3, [128, 128, 512], stage=3, block='d')

        # Stage 4 (≈6 lines)
        X = ResnetBlock.convolutional(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
        X = ResnetBlock.identity(X, 3, [256, 256, 1024], stage=4, block='b')
        X = ResnetBlock.identity(X, 3, [256, 256, 1024], stage=4, block='c')
        X = ResnetBlock.identity(X, 3, [256, 256, 1024], stage=4, block='d')
        X = ResnetBlock.identity(X, 3, [256, 256, 1024], stage=4, block='e')
        X = ResnetBlock.identity(X, 3, [256, 256, 1024], stage=4, block='f')

        # Stage 5 (≈3 lines)
        X = ResnetBlock.convolutional(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
        X = ResnetBlock.identity(X, 3, [512, 512, 2048], stage=5, block='b')
        X = ResnetBlock.identity(X, 3, [512, 512, 2048], stage=5, block='c')

        # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
        X = AveragePooling2D(pool_size=(2, 2),name = 'avg_pool')(X)

        # output layer
        X = Flatten()(X)
        X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0))(X)

        # Create model
        model = Model(inputs = X_input, outputs = X, name='ResNet50')
        return model

    @classmethod
    def model(cls, X_train, Y_train, X_test, Y_test, num_epochs = 2, minibatch_size = 32):
        # step 1, create the model.
        model = cls.ResNet50Impl(input_shape = (64, 64, 3), classes = 6)
        # step 2, compile the model to configure the learning process.
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # step 3, train the model. Choose the number of epochs and the batch size.
        model.fit(x = X_train, y = Y_train, epochs = num_epochs, batch_size = minibatch_size)
        # step 4, test/evaluate the model.
        preds = model.evaluate(x = X_test, y = Y_test)
        print("Loss = " + str(preds[0]))
        print("Test Accuracy = " + str(preds[1]))
        return model

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
