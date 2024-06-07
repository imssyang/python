import pprint
import numpy as np


def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s


def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.

    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """
    s = sigmoid(x)
    ds = s * (1 - s)
    return ds


def softmax(x):
    """Calculates the softmax for each row of the input x.

    Your code should work for a row vector and also for matrices of shape (n, m).

    Argument:
    x -- A numpy matrix of shape (n,m)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """
    # Apply exp() element-wise to x. Use np.exp(...).
    x_exp = np.exp(x)

    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    x_sum = np.sum(x_exp, axis=1, keepdims=True)

    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    s = x_exp / x_sum
    return s


def relu(x):
    """
    Compute the relu of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
    """
    s = np.maximum(0,x)
    return s


def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- the value of the L1 loss function defined above
    """
    loss = np.sum(np.abs(y - yhat))
    return loss


def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- the value of the L2 loss function defined above
    """
    loss = np.sum((y-yhat)@(y-yhat))
    return loss


def printer(*args, **kwargs):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def numpy2str(data):
        if isinstance(data, dict):
            for key, value in data.items():
                data[key] = convert(value)
        if isinstance(data, list):
            data_tmp = []
            for value in data:
                data_tmp.append(convert(value))
            data = data_tmp
        else:
            data = convert(data)
        return data

    pp = pprint.PrettyPrinter(indent=1, width=80, compact=False)
    kvargs = dict()
    for key, value in kwargs.items():
        kvargs[key] = numpy2str(value)
    argl = list()
    for arg in args:
        argl.append(numpy2str(arg))
    if len(argl) > 0 and len(kvargs) > 0:
        argl.append(kvargs)
        pp.pprint(argl)
    elif len(argl) > 0:
        pp.pprint(argl)
    else:
        pp.pprint(kvargs)
