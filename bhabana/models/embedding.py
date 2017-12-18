import torch as th
import torch.nn as nn


class Embedding(nn.Module):

    def __init__(self, vocab_size, embedding_dims, padding_idx=0,
                 pretrained_word_vectors=None, trainable=True, dropout=0.5):
        super(Embedding, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dims = embedding_dims
        self.embedding = nn.Embedding(vocab_size, embedding_dims,
                                      padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.init_weights(pretrained_word_vectors, trainable)

    def init_weights(self, pretrained_word_vectors, trainable=False):
        if pretrained_word_vectors is not None:
            self.embedding.weight = nn.Parameter(
                th.from_numpy(pretrained_word_vectors).float())
        if not trainable:
            self.embedding.weight.requires_grad = False

    def forward(self, data):

        # Shape B X T X EMBEDDING_DIMS, where all the paddings are 0 vectors
        # of size EMBEDDING_DIMS
        embedding_features = self.embedding(data["inputs"])
        if data["training"]:
            embedding_features = self.dropout(embedding_features)

        resp = {"out": embedding_features}
        return resp