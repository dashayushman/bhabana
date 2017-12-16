import os
import logging
import shutil

import bhabana.utils as utils

from bhabana.models import NGramCNN
from nose.tools import *
from bhabana.utils import data_utils as du

logger = logging.getLogger(__name__)


class TestDataUtils():

    def test_instantiate_cnn_blstm_model(self):
        ngram_cnn = NGramCNN()