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
    bidirectional = False
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

    @classmethod
    def validate_hidden(cls, hidden_out):
        if type(hidden_out) is tuple:
            for hidden in hidden_out:
                hidden_size = list(hidden.size())
                assert_equals(hidden_size[0],
                              cls.n_layers * (2 if cls.bidirectional else 1))
                assert_equals(hidden_size[1], cls.n_samples)
                assert_equals(hidden_size[2], cls.hidden_size)
        else:
            hidden_size = list(hidden_out.size())
            assert_equals(hidden_size[0],
                          cls.n_layers * (2 if cls.bidirectional else 1))
            assert_equals(hidden_size[1], cls.n_samples)
            assert_equals(hidden_size[2], cls.hidden_size)

    @classmethod
    def validate_output(cls, out):
        output_shape = list(out.size())
        assert_equals(output_shape[0], cls.n_samples)
        assert_equals(output_shape[1], cls.time_steps)
        assert_equals(output_shape[2], cls.hidden_size)

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
    def test_invalid_forward_gru(self):
         self.rnns[0](self.inputs)

    @raises(Exception)
    def test_invalid_forward_lstm(self):
        self.rnns[1](self.inputs)

    @raises(Exception)
    def test_invalid_forward_rnn(self):
        self.rnns[2](self.inputs)

    @raises(Exception)
    def test_invalid_data_forward(self):
        in_seq = Variable(th.randn(10, 45, 98))
        data = {"inputs": in_seq, "hidden": self.rnns[0].init_hidden(20)}
        self.rnns[0](data)

    def test_valid_forward(self):
        for rnn in self.rnns:
            data = {
                "inputs": self.inputs,
                "hidden": rnn.init_hidden(self.n_samples)
            }
            response = rnn(data)
            assert_not_equals(response, None)
            self.validate_hidden(response["out_hidden"])
            self.validate_output(response["out"])

    def test_repackage_hidden(self):
        for rnn in self.rnns:
            data = {
                "inputs": self.inputs,
                "hidden": rnn.init_hidden(self.n_samples)
            }
            for i in range(5):
                data["hidden"] = rnn.repackage_hidden(data["hidden"])
                response = rnn(data)
                assert_not_equals(response, None)
                self.validate_hidden(response["out_hidden"])
                self.validate_output(response["out"])
