import torch

import bhabana.utils.generic_utils as gu
import torch.nn as nn
import bhabana.utils.constants as Constants
from bhabana.models import EncoderLayer


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1,
            pretrained_word_vectors=None, trainable_embeddings=True):

        super(Encoder, self).__init__()

        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        d_word_vec = d_word_vec if pretrained_word_vectors is None else \
            pretrained_word_vectors.shape[-1]

        self.position_enc = nn.Embedding(n_position, d_word_vec,
                                         padding_idx=Constants.PAD)
        self.position_enc.weight.data = gu.position_encoding_init(n_position,
                                                                    d_word_vec)
        self.position_enc.weight.requires_grad = False

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec,
                                         padding_idx=Constants.PAD)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.init_weights(pretrained_word_vectors, trainable_embeddings)

    def init_weights(self, pretrained_word_vectors, trainable=False):
        if pretrained_word_vectors is not None:
            self.src_word_emb.weight = nn.Parameter(
                torch.from_numpy(pretrained_word_vectors).float())
        if not trainable:
            self.src_word_emb.weight.requires_grad = False

    def forward(self, src_seq, src_pos, return_attns=False):
        # Word embedding look up
        enc_input = self.src_word_emb(src_seq)

        # Position Encoding addition
        enc_input += self.position_enc(src_pos)
        if return_attns:
            enc_slf_attns = []

        enc_output = enc_input
        enc_slf_attn_mask = gu.get_attn_padding_mask(src_seq, src_seq)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, slf_attn_mask=enc_slf_attn_mask)
            if return_attns:
                enc_slf_attns += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attns
        else:
            return enc_output