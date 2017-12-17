import torch as th
import numpy as np


# This method has been taken from tflearn
def to_categorical(y, n_classes):
    """ to_categorical.

    Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.

    Arguments:
        y: `array`. Class vector to convert.
    """
    return np.eye(n_classes, dtype='uint8')[y]

def get_activation_by_name(activation=None):
    act_module = None
    if activation is None:
        act_module = None
    elif activation == "threshold":
        act_module = th.nn.Threshold
    elif activation == "relu":
        act_module = th.nn.ReLU
    elif activation == "hardtanh":
        act_module = th.nn.Hardtanh
    elif activation == "relu6":
        act_module = th.nn.ReLU6
    elif activation == "elu":
        act_module = th.nn.ELU
    elif activation == "selu":
        act_module = th.nn.SELU
    elif activation == "leaky_relu":
        act_module = th.nn.LeakyReLU
    elif activation == "prelu":
        act_module = th.nn.PReLU
    elif activation == "rrelu":
        act_module = th.nn.RReLU
    elif activation == "glu":
        act_module = th.nn.GLU
    elif activation == "logsigmoid":
        act_module = th.nn.LogSigmoid
    elif activation == "hardshrink":
        act_module = th.nn.Hardshrink
    elif activation == "tanhshrink":
        act_module = th.nn.Tanhshrink
    elif activation == "softsign":
        act_module = th.nn.Softsign
    elif activation == "softplus":
        act_module = th.nn.Softplus
    elif activation == "softmin":
        act_module = th.nn.Softmin
    elif activation == "softmax":
        act_module = th.nn.Softmax
    elif activation == "softshrink":
        act_module = th.nn.Softshrink
    elif activation == "log_softmax":
        act_module = th.nn.LogSoftmax
    elif activation == "tanh":
        act_module = th.nn.Tanh
    elif activation == "sigmoid":
        act_module = th.nn.Sigmoid
    else:
        raise ValueError("{} Activation function has not been implemented in "
                         "the current version or is an invalid activation "
                         "function.".format(activation))
    return act_module