import torch
import numpy as np

from nose.tools import *
from torch.autograd import Variable
from bhabana.models import Embedding

class TestEmbedding():
    vocab_size = 20
    embedding_dims = 300

    def setUp(self):
        self.emb = Embedding(vocab_size=self.vocab_size,
                        embedding_dims=self.embedding_dims)

    def tearDown(self):
        self.emb = None

    def test_default_params(self):
        assert_equals(self.emb.embedding.embedding_dim, 300)
        assert_equals(self.emb.embedding.num_embeddings, 20)

    def test_init_weights(self):
        weights = np.ones((self.vocab_size, self.embedding_dims))
        self.emb.init_weights(weights, trainable=False)
        assert_equals(self.emb.embedding.weight.requires_grad, False)
        assert_true(np.array_equal(weights,
                                     self.emb.embedding.weight.data.numpy()))

    @raises(Exception)
    def test_invalid_forward(self):
        batch_input = [[1, 4, 2, 6], [3, 5, 2, 7]]
        data = {"inputs": batch_input, "training": True}
        self.emb.forward(data)

    @raises(Exception)
    def test_out_of_bound_forward(self):
        batch_input = Variable(torch.LongTensor([[1,2,4,5],[4,3,2,48]]))
        data = {"inputs": batch_input, "training": True}
        self.emb.forward(data)

    def test_valid_forward(self):
        weights = np.ones((self.vocab_size, self.embedding_dims))
        weights[0,:] = np.zeros((300,))
        self.emb.init_weights(weights, trainable=False)

        batch_input = Variable(torch.LongTensor([[1, 2, 4, 5, 0],
                                                 [4, 3, 2, 9, 10]]))
        data = {"inputs": batch_input, "training": False}
        response = self.emb.forward(data)

        for ret in self.emb.provides:
            assert_true(ret in response)
        gt_1 = np.ones((5, self.embedding_dims))
        gt_1[-1,:] = np.zeros((self.embedding_dims,))
        assert_true(np.array_equal(response["out"][0].data.numpy(),
                                   gt_1))
        assert_true(np.array_equal(response["out"][1].data.numpy(),
                           np.ones((5, 300))))
        assert_true(np.array_equal(response["out"][0].data.numpy()[-1],
                           np.zeros((300,))))

