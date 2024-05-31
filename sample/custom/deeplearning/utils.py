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


def printer(*args, **kwargs):
    def numpy2str(data):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    data[key] = value.tolist()
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
