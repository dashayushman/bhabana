import torch

import torch.nn as nn
from bhabana.models import Encoder
from bhabana.models import Regressor
from bhabana.models import Linear


class TransformerClassifier(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_max_seq, n_classes, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1,
            pretrained_word_vectors=None, trainable_embeddings=True):
        super(TransformerClassifier, self).__init__()
        self.d_model = d_model
        self.encoder = Encoder(
            n_src_vocab, n_max_seq, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout,
            pretrained_word_vectors=pretrained_word_vectors,
                trainable_embeddings=trainable_embeddings)
        self.dropout = nn.Dropout(dropout)
        self.fcl = Linear(d_model, d_model, activation="relu")
        self.classifier = Linear(d_model, n_classes, bias=True,
                                  activation=None)


        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module output shall be the same.'


    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        enc_freezed_param_ids = set(map(id, self.encoder.position_enc.parameters()))
        freezed_param_ids = enc_freezed_param_ids
        return (p for p in self.parameters() if (id(p) not in
                freezed_param_ids) or p.requires_grad == True)

    def get_embedding_weights(self):
        return self.encoder.src_word_emb.weight.clone().data.cpu()


    def forward(self, data):
        src_seq, src_pos = data["text"], data["text_position"]

        enc_output, attn = self.encoder(src_seq, src_pos, return_attns=True)

        reduced_sum = torch.sum(enc_output, 1)
        data = {"inputs": reduced_sum}
        fcl_out = self.fcl(data)["out"]
        data["inputs"] = fcl_out
        classifier_output = self.classifier(data)["out"]
        output = {"out": classifier_output, "attention": attn}
        return output