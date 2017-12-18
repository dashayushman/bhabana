from bhabana.models import BatchBottle
from bhabana.models import LayerNormalization


class BottleLayerNormalization(BatchBottle, LayerNormalization):
    ''' Perform the reshape routine before and after a layer normalization'''
    pass