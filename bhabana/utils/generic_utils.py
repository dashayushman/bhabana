import torch as th
import numpy as np

from bhabana.utils import constants


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


def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return th.from_numpy(position_enc).type(th.FloatTensor)


def get_attn_padding_mask(seq_q, seq_k):
    ''' Indicate the padding-related part to mask '''
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(constants .PAD).unsqueeze(1)   # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # bxsqxsk
    return pad_attn_mask