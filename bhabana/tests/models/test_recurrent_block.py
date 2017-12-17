import logging

import numpy as np
import torch as th

from bhabana.models import RecurrentBlock
from torch.autograd import Variable
from nose.tools import *

logger = logging.getLogger(__name__)


class TestDataUtils():
    input_size = 300
    hidden_size = 100
    n_samples = 20
    time_steps = 50
    dropout = 0.3
    n_layers = 3
    bidirectional = True
    cell_type = ["GRU", "RNN", "LSTM"]
    inputs = Variable(th.randn(n_samples, time_steps, input_size))

    def setUp(self):
        self.rnns = [RecurrentBlock(input_size=self.input_size,
                                    hidden_size=self.hidden_size,
                                    bidirectional=self.bidirectional,
                                    rnn_cell=ctype, n_layers=self.n_layers,
                                    dropout=self.dropout)
                     for ctype in self.cell_type]

    def tearDown(self):
        self.rnns = None

    @raises(Exception)
    def test_invalid_constructor_params_input_size(self):
        self.rnn = RecurrentBlock(input_size=-234,
                                    hidden_size=self.hidden_size,
                                    bidirectional=self.bidirectional,
                                    rnn_cell="GRU", n_layers=self.n_layers,
                                    dropout=self.dropout)
        data = {"inputs": self.inputs, "training": True}
        self.rnn(data)

    @raises(Exception)
    def test_invalid_constructor_params_hidden_size(self):
        self.rnn = RecurrentBlock(input_size=self.input_size,
                                  hidden_size=-453,
                                  bidirectional=self.bidirectional,
                                  rnn_cell="GRU", n_layers=self.n_layers,
                                  dropout=self.dropout)
        data = {"inputs": self.inputs,
                "hidden": self.rnn.init_hidden(self.n_samples)}
        self.rnn(data)

    @raises(Exception)
    def test_invalid_constructor_params_bidirectional(self):
        self.rnn = RecurrentBlock(input_size=self.input_size,
                                  hidden_size=self.hidden_size,
                                  bidirectional="hello",
                                  rnn_cell="GRU", n_layers=self.n_layers,
                                  dropout=self.dropout)
        data = {"inputs": self.inputs,
                "hidden": self.rnn.init_hidden(self.n_samples)}
        self.rnn(data)

    @raises(ValueError)
    def test_invalid_constructor_params_cell_type(self):
        self.rnn = RecurrentBlock(input_size=self.input_size,
                                  hidden_size=self.hidden_size,
                                  bidirectional=self.bidirectional,
                                  rnn_cell="blah", n_layers=self.n_layers,
                                  dropout=self.dropout)
        data = {"inputs": self.inputs,
                "hidden": self.rnn.init_hidden(self.n_samples)}
        self.rnn(data)

    @raises(Exception)
    def test_invalid_constructor_params_n_layers(self):
        self.rnn = RecurrentBlock(input_size=self.input_size,
                                  hidden_size=self.hidden_size,
                                  bidirectional=self.bidirectional,
                                  rnn_cell="GRU", n_layers=-23,
                                  dropout=self.dropout)
        data = {"inputs": self.inputs,
                "hidden": self.rnn.init_hidden(self.n_samples)}
        self.rnn(data)

    @raises(Exception)
    def test_invalid_constructor_params_dropout(self):
        self.rnn = RecurrentBlock(input_size=self.input_size,
                                  hidden_size=self.hidden_size,
                                  bidirectional=self.bidirectional,
                                  rnn_cell="GRU", n_layers=-23,
                                  dropout=self.dropout)
        data = {"inputs": self.inputs,
                "hidden": self.rnn.init_hidden(self.n_samples)}
        self.rnn(data)

    @raises(Exception)
    def test_invalid_forward(self):
        pass

    @raises(Exception)
    def test_invalid_data_forward(self):
        pass

    def test_valid_forward(self):
        pass