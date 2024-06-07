import unittest
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
from datasets import SignDataSet as DataSet
from nn3_tensorflow import TensorflowOperation, TensorflowNN


class TensorflowTest(unittest.TestCase):
    def setUp(self):
        tf_v1.disable_eager_execution()

    def test_multiply(self):
        # Initialize variables
        a = tf.constant(2)
        b = tf.constant(10)
        c = tf.multiply(a,b)
        print(c) # Tensor("Mul:0", shape=(), dtype=int32)

        # Create a session and run the operations inside the session
        sess = tf_v1.Session()
        print(sess.run(c)) # 20

    def test_placeholder(self):
        # Change the value of x in the feed_dict
        x = tf_v1.placeholder(tf.int64, name = 'x')
        sess = tf_v1.Session()
        print(sess.run(2 * x, feed_dict = {x: 3})) # 6
        sess.close()

    def test_variable(self):
        y_hat = tf.constant(36, name='y_hat')            # Define y_hat constant. Set to 36.
        y = tf.constant(39, name='y')                    # Define y. Set to 39
        loss = tf.Variable((y - y_hat)**2, name='loss')  # Create a variable for the loss
        init = tf_v1.global_variables_initializer()       # When init is run later (session.run(init))
        with tf_v1.Session() as session:                  # Create a session and print the output
            # the loss variable will be initialized and ready to be computed
            session.run(init)                            # Initializes the variables
            print(session.run(loss))                     # Prints the loss

    def linear_function(self):
        """
        Implements a linear function:
                Initializes W to be a random tensor of shape (4,3)
                Initializes X to be a random tensor of shape (3,1)
                Initializes b to be a random tensor of shape (4,1)
        Returns:
        result -- runs the session for Y = WX + b
        """
        np.random.seed(1)
        X = tf.constant(np.random.randn(3,1),name='X')
        W = tf.constant(np.random.randn(4,3),name='W')
        b = tf.constant(np.random.randn(4,1),name='b')
        Y = tf.add(tf.matmul(W,X),b)

        # Create the session using tf.Session() and run it with sess.run(...) on the variable you want to calculate
        sess = tf_v1.Session()
        result = sess.run(Y)
        sess.close()
        return result

    def test_linear_function(self):
        print("result = " + str(self.linear_function()))

    def sigmoid(self, z):
        """
        Computes the sigmoid of z

        Arguments:
        z -- input value, scalar or vector
        Returns:
        results -- the sigmoid of z
        """
        # Create a placeholder for x. Name it 'x'.
        x = tf_v1.placeholder(tf.float32,name='x')

        # compute sigmoid(x)
        sigmoid = tf.sigmoid(x)

        # Create a session, and run it. Please use the method 2 explained above.
        # You should use a feed_dict to pass z's value to x.
        with tf_v1.Session() as sess:
            # Run session and call the output "result"
            result = sess.run(sigmoid, feed_dict={x:z})
        return result

    def test_sigmoid(self):
        print("sigmoid(0) = " + str(self.sigmoid(0)))
        print("sigmoid(12) = " + str(self.sigmoid(12)))

    def cost(self, logits, labels):
        """
        Computes the cost using the sigmoid cross entropy
        Arguments:
        logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
        labels -- vector of labels y (1 or 0)
        Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels"
        in the TensorFlow documentation. So logits will feed into z, and labels into y.
        Returns:
        cost -- runs the session of the cost (formula (2))
        """
        # Create the placeholders for "logits" (z) and "labels" (y) (approx. 2 lines)
        z = tf_v1.placeholder(tf.float32,name='z')
        y = tf_v1.placeholder(tf.float32,name='y')

        # Use the loss function (approx. 1 line)
        cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z,labels=y)
        sess = tf_v1.Session()
        cost = sess.run(cost,feed_dict={z:logits,y:labels})
        sess.close()
        return cost

    def test_cost(self):
        logits = self.sigmoid(np.array([0.2,0.4,0.7,0.9]))
        cost = self.cost(logits, np.array([0,0,1,1]))
        print ("cost = " + str(cost))

    def one_hot_matrix(self, labels, C):
        """
        Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                        corresponds to the jth training example. So if example j had a label i. Then entry (i,j)
                        will be 1.
        Arguments:
        labels -- vector containing the labels
        C -- number of classes, the depth of the one hot dimension
        Returns:
        one_hot -- one hot matrix
        """
        # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
        C = tf.constant(C, name='C')

        # Use tf.one_hot, be careful with the axis (approx. 1 line)
        one_hot_matrix = tf.one_hot(labels,C,axis=0)

        sess = tf_v1.Session()
        one_hot = sess.run(one_hot_matrix)
        sess.close()
        return one_hot

    def test_one_hot_matrix(self):
        labels = np.array([1,2,3,0,2,1])
        one_hot = self.one_hot_matrix(labels, C = 4)
        print("one_hot = " + str(one_hot))

    def ones(self, shape):
        """
        Creates an array of ones of dimension shape
        Arguments:
        shape -- shape of the array you want to create
        Returns:
        ones -- array containing only ones
        """
        # Create "ones" tensor using tf.ones(...). (approx. 1 line)
        ones = tf.ones(shape)
        sess = tf_v1.Session()
        ones = sess.run(ones)
        sess.close()
        return ones

    def test_ones(self):
        print ("ones = " + str(self.ones([3]))) # [1. 1. 1.]


class TensorflowOperationTest(unittest.TestCase):
    def test_create_placeholders(self):
        X, Y = TensorflowOperation.create_placeholders(12288, 6)
        print ("X = " + str(X))
        print ("Y = " + str(Y))

    def test_initialize_parameters(self):
        tf_v1.reset_default_graph()
        with tf_v1.Session() as sess:
            parameters = TensorflowOperation.initialize_parameters()
            print("W1 = " + str(parameters["W1"]))
            print("b1 = " + str(parameters["b1"]))
            print("W2 = " + str(parameters["W2"]))
            print("b2 = " + str(parameters["b2"]))

    def test_forward_propagation(self):
        # The forward propagation doesn't output any cache.
        tf_v1.reset_default_graph()
        with tf_v1.Session() as sess:
            X, Y = TensorflowOperation.create_placeholders(12288, 6)
            parameters = TensorflowOperation.initialize_parameters()
            Z3 = TensorflowOperation.forward_propagation(X, parameters)
            print("Z3 = " + str(Z3))

    def test_compute_cost(self):
        tf_v1.reset_default_graph()
        with tf_v1.Session() as sess:
            X, Y = TensorflowOperation.create_placeholders(12288, 6)
            parameters = TensorflowOperation.initialize_parameters()
            Z3 = TensorflowOperation.forward_propagation(X, parameters)
            cost = TensorflowOperation.compute_cost(Z3, Y)
            print("cost = " + str(cost))


class TensorflowNNTest(unittest.TestCase):
    def setUp(self):
        self.X_train_orig, self.Y_train_orig, self.X_test_orig, self.Y_test_orig, self.classes = DataSet.load()
        DataSet.show_sign(self.X_train_orig, self.Y_train_orig, 0)

        self.X_train, self.Y_train, self.X_test, self.Y_test = self.flatten_image_dataset(
            self.X_train_orig, self.Y_train_orig,
            self.X_test_orig, self.Y_test_orig
        )
        self.learning_rate = 0.0001
        self.loss_xlable = 'iterations (per tens)'

    def convert_to_one_hot(self, Y, C):
        Y = np.eye(C)[Y.reshape(-1)].T
        return Y

    def flatten_image_dataset(self, X_train_orig, Y_train_orig, X_test_orig, Y_test_orig):
        # Flatten the training and test images
        X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
        X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
        # Normalize image vectors
        X_train = X_train_flatten / 255.
        X_test = X_test_flatten / 255.
        # Convert training and test labels to one hot matrices
        Y_train = self.convert_to_one_hot(Y_train_orig, 6)
        Y_test = self.convert_to_one_hot(Y_test_orig, 6)
        print("number of training examples = " + str(X_train.shape[1]))
        print("number of test examples = " + str(X_test.shape[1]))
        print("X_train shape: " + str(X_train.shape))
        print("Y_train shape: " + str(Y_train.shape))
        print("X_test shape: " + str(X_test.shape))
        print("Y_test shape: " + str(Y_test.shape))
        return X_train, Y_train, X_test, Y_test

    def test_model(self):
        parameters, costs = TensorflowNN.model(self.X_train, self.Y_train, self.X_test, self.Y_test)
        DataSet.show_loss(costs, self.learning_rate, self.loss_xlable)

        image, image64 = DataSet.load_thumb_image()
        image_prediction = TensorflowNN.predict(image64, parameters)
        print("Model predicts: y = " + str(np.squeeze(image_prediction)))
        DataSet.show_image(image)
