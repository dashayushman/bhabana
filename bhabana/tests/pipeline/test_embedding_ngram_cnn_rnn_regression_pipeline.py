import torch

import torch.nn as nn
import torch.optim as optim

from nose.tools import *
from torch.autograd import Variable
from bhabana.pipeline import EmbeddingNgramCNNRNNRegression


class TestEMbeddingNgramCNNRNNRegressionPipeline:
    n_samples = 3
    time_steps = 7
    vocab_size = 100
    embedding_dims = 300
    embedding_dropout = 0.3
    pretrained_word_vectors=None
    padding_idx=0
    trainable_embeddings=True
    cnn_kernel_dims = 400
    cnn_kernel_sizes = [1, 3, 5, 7]
    cnn_layers = 3
    cnn_dropout = 0.3
    rnn_hidden_size = 100
    rnn_layers = 3
    bidirectional = True
    return_sequence = False
    rnn_dropout = 0.4
    cell_types = ["GRU", "RNN", "LSTM"]
    activation = None
    input_batch = Variable(torch.LongTensor([[1, 2, 4, 5, 0, 0, 0],
                                             [4, 3, 2, 9, 10, 0, 0],
                                             [4, 1, 3, 4, 11, 0, 0]]),
                           requires_grad=False)
    gt = Variable(torch.FloatTensor([[1], [0], [1]]),
                           requires_grad=False)
    def setUp(self):
        self.pipelines = [EmbeddingNgramCNNRNNRegression(
                vocab_size=self.vocab_size,
                embedding_dims=self.embedding_dims,
                rnn_hidden_size=self.rnn_hidden_size,
                bidirectional=self.bidirectional, rnn_cell=ctype,
                rnn_layers=self.rnn_layers, cnn_layers=self.cnn_layers,
                cnn_kernel_dim=self.cnn_kernel_dims,
                cnn_kernel_sizes=self.cnn_kernel_sizes,
                padding_idx=self.padding_idx, pretrained_word_vectors=None,
                trainable_embeddings=self.trainable_embeddings,
                embedding_dropout=self.embedding_dropout,
                regression_activation=self.activation,
                cnn_dropout=self.cnn_dropout, rnn_dropout=self.rnn_dropout)
            for ctype in self.cell_types]

    def tearDown(self):
        self.pipelines = None

    @classmethod
    def validate_response(cls, response):
        assert_true("1.Embedding.out" in response)
        assert_true("2.NGramCNN.out" in response)
        assert_true("2.NGramCNN.aux" in response)
        assert_true("3.RecurrentBlock.out" in response)
        assert_true("3.RecurrentBlock.aux" in response)
        assert_true("4.Regressor.out" in response)
        assert_true("inputs" in response)
        assert_true("hidden" in response)


    def test_valid_forward(self):
        for pipeline in self.pipelines:
            data = {"inputs": self.input_batch, "training": True, "hidden":
                pipeline.init_rnn_hidden(self.n_samples)}
            for i in range(10):
                data["hidden"] = pipeline.repackage_rnn_hidden(data["hidden"])
                data["inputs"] = Variable(torch.LongTensor([[1, 2, 4, 5, 0, 0, 0],
                                             [4, 3, 2, 9, 10, 0, 0],
                                             [4, 1, 3, 4, 11, 0, 0]]))
                pipeline.zero_grad()
                response = pipeline(data)
                assert_not_equals(response, None)
                self.validate_response(response)

    def test_valid_forward_activation_no_bck(self):
        pipeline = EmbeddingNgramCNNRNNRegression(
                vocab_size=self.vocab_size,
                embedding_dims=self.embedding_dims,
                rnn_hidden_size=self.rnn_hidden_size,
                bidirectional=False, rnn_cell="GRU",
                rnn_layers=self.rnn_layers, cnn_layers=self.cnn_layers,
                cnn_kernel_dim=self.cnn_kernel_dims,
                cnn_kernel_sizes=self.cnn_kernel_sizes,
                padding_idx=self.padding_idx, pretrained_word_vectors=None,
                trainable_embeddings=self.trainable_embeddings,
                embedding_dropout=self.embedding_dropout,
                regression_activation="sigmoid",
                cnn_dropout=self.cnn_dropout, rnn_dropout=self.rnn_dropout)

        data = {"inputs": self.input_batch, "training": True, "hidden":
            pipeline.init_rnn_hidden(self.n_samples)}
        for i in range(10):
            data["hidden"] = pipeline.repackage_rnn_hidden(data["hidden"])
            data["inputs"] = Variable(torch.LongTensor([[1, 2, 4, 5, 0, 0, 0],
                                         [4, 3, 2, 9, 10, 0, 0],
                                         [4, 1, 3, 4, 11, 0, 0]]))
            pipeline.zero_grad()
            response = pipeline(data)
            assert_not_equals(response, None)
            self.validate_response(response)

    def test_valid_train(self):
        for pipeline in self.pipelines:
            loss_function = nn.MSELoss()
            optimizer = optim.Adam(pipeline.parameters(), lr=0.001)
            data = {"inputs": self.input_batch, "training": True, "hidden":
                pipeline.init_rnn_hidden(self.n_samples)}
            for i in range(10):
                data["hidden"] = pipeline.repackage_rnn_hidden(data["hidden"])
                data["inputs"] = Variable(torch.LongTensor([[1, 2, 4, 5, 0, 0, 0],
                                             [4, 3, 2, 9, 10, 0, 0],
                                             [4, 1, 3, 4, 11, 0, 0]]))
                pipeline.zero_grad()
                response = pipeline(data)
                assert_not_equals(response, None)
                self.validate_response(response)
                #data["hidden"] = response["hidden"]
                loss = loss_function(response["out"], self.gt)
                loss.backward()
                optimizer.step()