import torch.nn as nn

from bhabana.models import Bottle

class BottleSoftmax(Bottle, nn.Softmax):
    ''' Perform the reshape routine before and after a softmax operation'''
    pass