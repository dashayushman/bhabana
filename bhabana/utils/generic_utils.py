import torch as th
import numpy as np


# This method has been taken from tflearn
def to_categorical(y, nb_classes=None):
    """ to_categorical.

    Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.

    Arguments:
        y: `array`. Class vector to convert.
        nb_classes: `unused`. Used for older code compatibility.
    """
    return (y[:, None] == np.unique(y)).astype(np.float32)