import math
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
import tensorflow.python.framework as tf_py
import tf_slim

tf_v1.disable_v2_behavior()


class TensorflowOperation:
    @classmethod
    def create_placeholders(cls, n_H0, n_W0, n_C0, n_y):
        """
        Creates the placeholders for the tensorflow session.

        Arguments:
        n_H0 -- scalar, height of an input image
        n_W0 -- scalar, width of an input image
        n_C0 -- scalar, number of channels of the input
        n_y -- scalar, number of classes

        Returns:
        X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
        Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
        """
        X = tf_v1.placeholder(tf.float32,shape=[None, n_H0, n_W0, n_C0])
        Y = tf_v1.placeholder(tf.float32,shape=[None, n_y])
        return X, Y

    @classmethod
    def initialize_parameters(cls):
        """
        Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                            W1 : [4, 4, 3, 8]
                            W2 : [2, 2, 8, 16]
        Returns:
        parameters -- a dictionary of tensors containing W1, W2
        """
        tf_v1.set_random_seed(1)  # so that your "random" numbers match ours
        W1 = tf_v1.get_variable('W1',[4, 4, 3, 8], initializer=tf_slim.layers.initializers.xavier_initializer(seed = 0))
        W2 = tf_v1.get_variable('W2',[2, 2, 8, 16], initializer=tf_slim.layers.initializers.xavier_initializer(seed = 0))
        parameters = {"W1": W1,
                    "W2": W2}
        return parameters

    @classmethod
    def forward_propagation(cls, X, parameters):
        """
        Implements the forward propagation for the model:
        CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

        Arguments:
        X -- input dataset placeholder, of shape (input size, number of examples)
        parameters -- python dictionary containing your parameters "W1", "W2"
                    the shapes are given in initialize_parameters

        Returns:
        Z3 -- the output of the last LINEAR unit
        """
        # Retrieve the parameters from the dictionary "parameters" 
        W1 = parameters['W1']
        W2 = parameters['W2']

        # CONV2D: stride of 1, padding 'SAME'
        Z1 = tf_v1.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')
        # RELU
        A1 = tf_v1.nn.relu(Z1)
        # MAXPOOL: window 8x8, sride 8, padding 'SAME'
        P1 = tf_v1.nn.max_pool(A1,[1,8,8,1],strides=[1,8,8,1],padding='SAME')
        # CONV2D: filters W2, stride 1, padding 'SAME'
        Z2 = tf_v1.nn.conv2d(P1,W2,strides=[1,1,1,1],padding='SAME')
        # RELU
        A2 = tf_v1.nn.relu(Z2)
        # MAXPOOL: window 4x4, stride 4, padding 'SAME'
        P2 = tf_v1.nn.max_pool(A2,[1,4,4,1],strides=[1,4,4,1],padding='SAME')
        # FLATTEN
        P2 = tf_slim.flatten(P2)
        # FULLY-CONNECTED without non-linear activation function (not not call softmax).
        # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
        Z3 = tf_slim.fully_connected(P2,6,activation_fn=None)
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
        cost = tf.reduce_mean(tf_v1.nn.softmax_cross_entropy_with_logits_v2(logits=Z3, labels=Y))
        return cost

    @classmethod
    def random_mini_batches(cls, X, Y, mini_batch_size = 64, seed = 0):
        """
        Creates a list of random minibatches from (X, Y)

        Arguments:
        X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
        mini_batch_size - size of the mini-batches, integer
        seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

        Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
        """
        m = X.shape[0]                  # number of training examples
        mini_batches = []
        np.random.seed(seed)

        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[permutation,:,:,:]
        shuffled_Y = Y[permutation,:]

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
            mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        return mini_batches


class TensorflowCNN:
    @classmethod
    def model(cls, X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
            num_epochs = 100, minibatch_size = 64, print_cost = True):
        """
        Implements a three-layer ConvNet in Tensorflow:
        CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

        Arguments:
        X_train -- training set, of shape (None, 64, 64, 3)
        Y_train -- test set, of shape (None, n_y = 6)
        X_test -- training set, of shape (None, 64, 64, 3)
        Y_test -- test set, of shape (None, n_y = 6)
        learning_rate -- learning rate of the optimization
        num_epochs -- number of epochs of the optimization loop
        minibatch_size -- size of a minibatch
        print_cost -- True to print the cost every 100 epochs

        Returns:
        train_accuracy -- real number, accuracy on the train set (X_train)
        test_accuracy -- real number, testing accuracy on the test set (X_test)
        parameters -- parameters learnt by the model. They can then be used to predict.
        """
        tf_py.ops.reset_default_graph()           # to be able to rerun the model without overwriting tf variables
        tf_v1.set_random_seed(1)                  # to keep results consistent (tensorflow seed)
        seed = 3                                  # to keep results consistent (numpy seed)
        (m, n_H0, n_W0, n_C0) = X_train.shape
        n_y = Y_train.shape[1]
        costs = []                                # To keep track of the cost

        # Create Placeholders of the correct shape
        X, Y = TensorflowOperation.create_placeholders(n_H0, n_W0, n_C0, n_y)

        # Initialize parameters
        parameters = TensorflowOperation.initialize_parameters()

        # Forward propagation: Build the forward propagation in the tensorflow graph
        Z3 = TensorflowOperation.forward_propagation(X, parameters)

        # Cost function: Add cost function to tensorflow graph
        cost = TensorflowOperation.compute_cost(Z3, Y)

        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
        optimizer = tf_v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Initialize all the variables globally
        init = tf_v1.global_variables_initializer()

        # Start the session to compute the tensorflow graph
        with tf_v1.Session() as sess:
            # Run the initialization
            sess.run(init)

            # Do the training loop
            for epoch in range(num_epochs):
                minibatch_cost = 0.
                num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
                # seed = seed + 1
                minibatches = TensorflowOperation.random_mini_batches(X_train, Y_train, minibatch_size)  # dont use seed，in order to significantly improve the accuracy
                for minibatch in minibatches:
                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch
                    # IMPORTANT: The line that runs the graph on a minibatch.
                    # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                    _ , temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                    minibatch_cost += temp_cost / num_minibatches

                # Print the cost every epoch
                if print_cost == True and epoch % 5 == 0:
                    print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
                if print_cost == True and epoch % 1 == 0:
                    costs.append(minibatch_cost)

            # lets save the parameters in a variable
            parameters = sess.run(parameters)
            print ("Parameters have been trained!")

            # Calculate the correct predictions
            predict_op = tf.argmax(Z3, 1)
            correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

            # Calculate accuracy on the test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
            test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
            print(accuracy)
            print("Train Accuracy:", train_accuracy)
            print("Test Accuracy:", test_accuracy)
            print(parameters.keys())
            return parameters, np.squeeze(costs)

    @classmethod
    def forward_propagation_for_predict(cls, X, parameters):
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
        Z1 = tf.add(tf.matmul(W1, X), b1)      # Z1 = np.dot(W1, X) + b1
        A1 = tf.nn.relu(Z1)                    # A1 = relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)     # Z2 = np.dot(W2, a1) + b2
        A2 = tf.nn.relu(Z2)                    # A2 = relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)     # Z3 = np.dot(W3,Z2) + b3
        return Z3

    @classmethod
    def predict(cls, X, parameters):
        W1 = tf.convert_to_tensor(parameters["W1"])
        W2 = tf.convert_to_tensor(parameters["W2"])
        params = {"W1": W1, "W2": W2}

        x = tf_v1.placeholder(tf.float32, shape=[None, 64, 64, 3])
        z3 = TensorflowOperation.forward_propagation(x, params)
        p = tf.argmax(z3)
        sess = tf_v1.Session()
        prediction = sess.run(p, feed_dict = {x: X})
        return prediction
