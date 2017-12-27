import torch.nn as nn

from bhabana.models import NGramCNN
from bhabana.models import Linear
from bhabana.models import Embedding
from bhabana.models import RecurrentBlock


class EmbeddingNgramCNNRNNClassification(nn.Module):
    def __init__(self, vocab_size, embedding_dims, rnn_hidden_size, n_classes,
                 bidirectional=True, rnn_cell="LSTM", rnn_layers=1,
                 cnn_layers=1, cnn_kernel_dim=100, cnn_kernel_sizes=(3, 4, 5),
                 padding_idx=0, pretrained_word_vectors=None,
                 trainable_embeddings=True, embedding_dropout=0.5,
                 cnn_dropout=0.5, rnn_dropout=0.5):
        super(EmbeddingNgramCNNRNNClassification, self).__init__()
        
        self.rnn_hidden_size = rnn_hidden_size
        self.pipeline = nn.ModuleList([Embedding(vocab_size=vocab_size,
                                                 embedding_dims=embedding_dims,
                                                 padding_idx=padding_idx,
                                                 pretrained_word_vectors=pretrained_word_vectors,
                                                 trainable=trainable_embeddings,
                                                 dropout=embedding_dropout),
                          NGramCNN(in_channels=embedding_dims,
                                   n_layers=cnn_layers,
                                   kernel_dim=cnn_kernel_dim,
                                   kernel_sizes=cnn_kernel_sizes,
                                   dropout=cnn_dropout),
                          RecurrentBlock(input_size=cnn_kernel_dim,
                                         hidden_size=rnn_hidden_size,
                                         bidirectional=bidirectional,
                                         rnn_cell=rnn_cell, n_layers=rnn_layers,
                                         dropout=rnn_dropout,
                                         return_sequence=False),
                           Linear(rnn_hidden_size, n_classes, bias=True,
                                  activation=None)])

    def init_rnn_hidden(self, batch_size, cuda=False):
        return self.pipeline[2].init_hidden(batch_size, cuda)

    def repackage_rnn_hidden(self, hidden):
        return self.pipeline[2].repackage_hidden(hidden)

    def get_embedding_weights(self):
        return self.pipeline[0].embedding.weight.clone().data.cpu()

    def forward(self, data):
        resp = {}
        for i_m, module in enumerate(self.pipeline):
            layer_resp = module(data)
            resp["{}.{}.out".format(i_m+1, type(module).__name__)] = layer_resp["out"]
            if "aux" in layer_resp:
                resp["{}.{}.aux".format(i_m + 1, type(module).__name__)] = layer_resp["aux"]
                if "hidden" in layer_resp["aux"]:
                    resp["hidden"] = layer_resp["aux"]["hidden"]
            data["inputs"] = layer_resp["out"]
        resp["out"] = layer_resp["out"]
        return resp