import logging

import numpy as np
import torch as th

from bhabana.models import NGramCNN
from torch.autograd import Variable
from nose.tools import *

logger = logging.getLogger(__name__)


class TestDataUtils():
    in_channels = 300
    kernel_dims = 400
    n_samples = 20
    time_steps = 50
    dropout = 0.3
    kernel_sizes = [1, 3, 5, 7, 9]
    inputs = Variable(th.randn(n_samples, time_steps, in_channels))

    def setUp(self):
        self.ngram_cnn = NGramCNN(self.in_channels, self.kernel_dims,
                                  self.kernel_sizes, self.dropout)

    @raises(Exception)
    def test_invalid_forward(self):
        input = np.random.randn(self.n_samples, self.in_channels,
                                self.time_steps)
        self.ngram_cnn(input)

    def test_invalid_input_shape_forward(self):
        data = {"inputs": self.inputs}
        response = self.ngram_cnn(data)

        output_shape = list(response["out"].size())
        ngram_features_shape = list(response["ngram_features"].size())
        assert_not_equals(response, None)
        assert_equals(output_shape[1], self.time_steps)
        assert_equals(ngram_features_shape[1], len(self.kernel_sizes) * self.time_steps)
