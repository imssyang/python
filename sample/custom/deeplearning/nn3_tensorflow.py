import math
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
import tensorflow.python.framework as tf_py

tf_v1.disable_v2_behavior()


class TensorflowOperation:
    @classmethod
    def create_placeholders(cls, n_x, n_y):
        """
        Creates the placeholders for the tensorflow session.

        Arguments:
        n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
        n_y -- scalar, number of classes (from 0 to 5, so -> 6)

        Returns:
        X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
        Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

        Tips:
        - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
        In fact, the number of examples during test/train is different.
        """
        X = tf_v1.placeholder(shape=[n_x,None],dtype='float')
        Y = tf_v1.placeholder(shape=[n_y,None],dtype='float')
        return X, Y

    @classmethod
    def initialize_parameters(cls):
        """
        Initializes parameters to build a neural network with tensorflow. The shapes are:
                            W1 : [25, 12288]
                            b1 : [25, 1]
                            W2 : [12, 25]
                            b2 : [12, 1]
                            W3 : [6, 12]
                            b3 : [6, 1]
        Returns:
        parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
        """
        tf_v1.set_random_seed(1)  # so that your "random" numbers match ours
        W1 = tf_v1.get_variable('W1', [25,12288], initializer=tf_v1.initializers.glorot_uniform(seed=1))
        b1 = tf_v1.get_variable('b1', [25,1], initializer=tf_v1.zeros_initializer())
        W2 = tf_v1.get_variable('W2', [12,25], initializer=tf_v1.initializers.glorot_uniform(seed=1))
        b2 = tf_v1.get_variable('b2', [12,1], initializer=tf_v1.zeros_initializer())
        W3 = tf_v1.get_variable('W3', [6,12], initializer=tf_v1.initializers.glorot_uniform(seed=1))
        b3 = tf_v1.get_variable('b3', [6,1], initializer=tf_v1.zeros_initializer())
        parameters = {"W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2,
                    "W3": W3,
                    "b3": b3}
        return parameters

    @classmethod
    def forward_propagation(cls, X, parameters):
        """
        Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

        Arguments:
        X -- input dataset placeholder, of shape (input size, number of examples)
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                    the shapes are given in initialize_parameters

        Returns:
        Z3 -- the output of the last LINEAR unit
        """
        # Retrieve the parameters from the dictionary "parameters"
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        W3 = parameters['W3']
        b3 = parameters['b3']

        # Numpy Equivalents: In tensorflow the last linear layer output is given as input to the function computing the loss.
        Z1 = tf.add(tf.matmul(W1,X),b1)   # Z1 = np.dot(W1, X) + b1
        A1 = tf.nn.relu(Z1)               # A1 = relu(Z1)
        Z2 = tf.add(tf.matmul(W2,A1),b2)  # Z2 = np.dot(W2, a1) + b2
        A2 = tf.nn.relu(Z2)               # A2 = relu(Z2)
        Z3 = tf.add(tf.matmul(W3,Z2),b3)  # Z3 = np.dot(W3,Z2) + b3
        return Z3

    @classmethod
    def compute_cost(cls, Z3, Y):
        """
        Computes the cost

        Arguments:
        Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
        Y -- "true" labels vector placeholder, same shape as Z3
        Returns:
        cost - Tensor of the cost function
        """
        # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
        logits = tf.transpose(Z3)
        labels = tf.transpose(Y)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
        return cost

    @classmethod
    def random_mini_batches(cls, X, Y, mini_batch_size = 64, seed = 0):
        """
        Creates a list of random minibatches from (X, Y)

        Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        mini_batch_size - size of the mini-batches, integer
        seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

        Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
        """
        m = X.shape[1]                  # number of training examples
        mini_batches = []
        np.random.seed(seed)

        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        return mini_batches


class TensorflowNN:
    @classmethod
    def model(cls, X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
            num_epochs = 1500, minibatch_size = 32, print_cost = True):
        """
        Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

        Arguments:
        X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
        Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
        X_test -- training set, of shape (input size = 12288, number of training examples = 120)
        Y_test -- test set, of shape (output size = 6, number of test examples = 120)
        learning_rate -- learning rate of the optimization
        num_epochs -- number of epochs of the optimization loop
        minibatch_size -- size of a minibatch
        print_cost -- True to print the cost every 100 epochs

        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """
        tf_py.ops.reset_default_graph()   # to be able to rerun the model without overwriting tf variables
        tf_v1.set_random_seed(1)       # to keep consistent results
        seed = 3                    # to keep consistent results
        (n_x, m) = X_train.shape    # (n_x: input size, m : number of examples in the train set)
        n_y = Y_train.shape[0]      # n_y : output size
        costs = []                  # To keep track of the cost

        # Create Placeholders of shape (n_x, n_y)
        X, Y = TensorflowOperation.create_placeholders(n_x, n_y)

        # Initialize parameters
        parameters = TensorflowOperation.initialize_parameters()

        # Forward propagation: Build the forward propagation in the tensorflow graph
        Z3 = TensorflowOperation.forward_propagation(X, parameters)

        # Cost function: Add cost function to tensorflow graph
        cost = TensorflowOperation.compute_cost(Z3, Y)

        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
        optimizer = tf_v1.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

        # Initialize all the variables
        init = tf_v1.global_variables_initializer()

        # Start the session to compute the tensorflow graph
        with tf_v1.Session() as sess:
            # Run the initialization
            sess.run(init)

            # Do the training loop
            for epoch in range(num_epochs):
                epoch_cost = 0.                       # Defines a cost related to an epoch
                num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
                seed = seed + 1
                minibatches = TensorflowOperation.random_mini_batches(X_train, Y_train, minibatch_size, seed)
                for minibatch in minibatches:
                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch

                    # IMPORTANT: The line that runs the graph on a minibatch.
                    # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                    # This computes the backpropagation by passing through the tensorflow graph in the reverse order.
                    _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                    epoch_cost += minibatch_cost / num_minibatches

                # Print the cost every epoch
                if print_cost == True and epoch % 100 == 0:
                    print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                if print_cost == True and epoch % 5 == 0:
                    costs.append(epoch_cost)

            # lets save the parameters in a variable
            parameters = sess.run(parameters)
            print ("Parameters have been trained!")

            # Calculate the correct predictions
            correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

            # Calculate accuracy on the test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
            print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
            return parameters, np.squeeze(costs)

    @classmethod
    def predict(self, X, parameters):
        W1 = tf.convert_to_tensor(parameters["W1"])
        b1 = tf.convert_to_tensor(parameters["b1"])
        W2 = tf.convert_to_tensor(parameters["W2"])
        b2 = tf.convert_to_tensor(parameters["b2"])
        W3 = tf.convert_to_tensor(parameters["W3"])
        b3 = tf.convert_to_tensor(parameters["b3"])
        params = {"W1": W1,
                "b1": b1,
                "W2": W2,
                "b2": b2,
                "W3": W3,
                "b3": b3}

        x = tf_v1.placeholder("float", [12288, 1])
        z3 = TensorflowOperation.forward_propagation(x, params)
        p = tf.argmax(z3)
        with tf_v1.Session() as sess:
            prediction = sess.run(p, feed_dict = {x: X})
        return prediction

