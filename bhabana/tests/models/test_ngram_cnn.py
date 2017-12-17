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
        self.ngram_cnn = NGramCNN(in_channels=self.in_channels,
                                  kernel_dim=self.kernel_dims,
                                  kernel_sizes=self.kernel_sizes,
                                  dropout=self.dropout)

    def tearDown(self):
        self.ngram_cnn = None

    @raises(Exception)
    def test_invalid_forward(self):
        input = np.random.randn(self.n_samples, self.in_channels,
                                self.time_steps)
        self.ngram_cnn(input)

    @raises(Exception)
    def test_invalid_data_forward(self):
        input = np.random.randn(self.n_samples, self.in_channels,
                                self.time_steps)
        data = {"inputs": input}
        self.ngram_cnn(data)

    @raises(Exception)
    def test_invalid_constructor_params_in_channels(self):
        self.ngram_cnn = NGramCNN(in_channels=-23,
                                  kernel_dim=self.kernel_dims,
                                  kernel_sizes=self.kernel_sizes,
                                  dropout=self.dropout)
        data = {"inputs": self.inputs, "training": True}
        self.ngram_cnn(data)

    @raises(Exception)
    def test_invalid_constructor_params_n_layers(self):
        self.ngram_cnn = NGramCNN(in_channels=self.in_channels,
                                  kernel_dim=self.kernel_dims,
                                  kernel_sizes=self.kernel_sizes,
                                  dropout=self.dropout,
                                  n_layers=-22)
        data = {"inputs": self.inputs, "training": True}
        self.ngram_cnn(data)

    @raises(Exception)
    def test_invalid_constructor_params_kernel_dim(self):
        self.ngram_cnn = NGramCNN(in_channels=self.in_channels,
                                  kernel_dim=-234,
                                  kernel_sizes=self.kernel_sizes,
                                  dropout=self.dropout,
                                  n_layers=1)
        data = {"inputs": self.inputs, "training": True}
        self.ngram_cnn(data)


    @raises(Exception)
    def test_invalid_constructor_params_kernel_sizes(self):
        self.ngram_cnn = NGramCNN(in_channels=self.in_channels,
                                  kernel_dim=self.kernel_dims,
                                  kernel_sizes=44,
                                  dropout=self.dropout,
                                  n_layers=1)
        data = {"inputs": self.inputs, "training": True}
        self.ngram_cnn(data)

    @raises(Exception)
    def test_invalid_constructor_params_dropout(self):
        self.ngram_cnn = NGramCNN(in_channels=self.in_channels,
                                  kernel_dim=self.kernel_dims,
                                  kernel_sizes=self.kernel_sizes,
                                  dropout=123,
                                  n_layers=1)
        data = {"inputs": self.inputs, "training": True}
        self.ngram_cnn(data)

    @raises(Exception)
    def test_invalid_constructor_params_dropout(self):
        self.ngram_cnn = NGramCNN(in_channels=self.in_channels,
                                  kernel_dim=self.kernel_dims,
                                  kernel_sizes=self.kernel_sizes,
                                  dropout=-234,
                                  n_layers=1)
        data = {"inputs": self.inputs, "training": True}
        self.ngram_cnn(data)

    def test_valid_forward(self):
        data = {"inputs": self.inputs, "training": False}
        response = self.ngram_cnn(data)

        for ret in self.ngram_cnn.provides:
            assert_true(ret in response)

        output_shape = list(response["out"].size())
        ngram_features_shape = list(response["ngram_features"].size())
        assert_not_equals(response, None)
        assert_equals(output_shape[0], self.n_samples)
        assert_equals(output_shape[1], self.time_steps)
        assert_equals(output_shape[2], self.kernel_dims)

        assert_equals(ngram_features_shape[0], self.n_samples)
        assert_equals(ngram_features_shape[1],
                      len(self.kernel_sizes) * self.time_steps)
        assert_equals(ngram_features_shape[2], self.kernel_dims)

    def test_multilayer_forward(self):

        for n_layer in range(5):
            self.ngram_cnn = NGramCNN(in_channels=self.in_channels,
                                      kernel_dim=self.kernel_dims,
                                      kernel_sizes=self.kernel_sizes,
                                      dropout=self.dropout,
                                      n_layers=n_layer+1)
            data = {"inputs": self.inputs, "training": False}
            response = self.ngram_cnn(data)

            for ret in self.ngram_cnn.provides:
                assert_true(ret in response)

            output_shape = list(response["out"].size())
            ngram_features_shape = list(response["ngram_features"].size())
            assert_not_equals(response, None)
            assert_equals(output_shape[0], self.n_samples)
            assert_equals(output_shape[1], self.time_steps)
            assert_equals(output_shape[2], self.kernel_dims)

            assert_equals(ngram_features_shape[0], self.n_samples)
            assert_equals(ngram_features_shape[1],
                          len(self.kernel_sizes) * self.time_steps)
            assert_equals(ngram_features_shape[2], self.kernel_dims)