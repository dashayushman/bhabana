import torch as th
import torch.nn as nn
import torch.nn.functional as F

from bhabana.utils import data_utils as du


class NGramCNN(nn.Module):

    requires = ["inputs"]

    provides = ["out", "ngram_features"]

    def __init__(self, in_channels, kernel_dim=100, kernel_sizes=(3, 4, 5),
                 dropout=0.5):
        super(NGramCNN, self).__init__()

        self.in_channels = in_channels
        self.kernel_dim = kernel_dim
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout
        self.convs = nn.ModuleList()
        for k_size in kernel_sizes:
            self.convs.append(nn.Conv1d(in_channels, kernel_dim, k_size))

        self.dropout = nn.Dropout(dropout)

    def ngram_convolve(self, inputs):
        conv_outs = []
        for conv, n_k in zip(self.convs, self.kernel_sizes):
            conv_outs.append(F.relu(conv((du.pad_1dconv_input(inputs,
                             kernel_size=n_k, mode="same")).permute(0, 2, 1))))
        return conv_outs

    def split_ngram_conv_output(self, conv_outs):
        split_list = []
        for conv_out in conv_outs:
            # Shape: B X CHANNELS X TIMESTEPS -> B X TIMESTEPS X CHANNELS
            conv_out = conv_out.permute(0, 2, 1)
            split = [s.squeeze(dim=1) for s in th.split(conv_out, 1, dim=1)]
            split_list.append(split)
        return split_list

    def merge_ngram_splits(self, split_list):
        merged_ngram_splits = zip(*split_list)
        merged_ngram_sequence = []
        for n_grams in merged_ngram_splits:
            merged_ngram_sequence.append(th.stack(n_grams, dim=0))
        merged_ngram_sequence = th.cat(merged_ngram_sequence, dim=0)

        # Shape: TIMESTEPS X B X CHANNELS -> B X CHANNELS X TIMESTEPS
        merged_ngram_sequence = merged_ngram_sequence.permute(1, 2, 0)
        return merged_ngram_sequence

    def pool_features(self, merged_ngram_sequence):
        return F.max_pool1d(merged_ngram_sequence,
                           kernel_size=len(self.kernel_sizes),
                           stride=len(self.kernel_sizes))

    def forward(self, data):
        # Input Shape: B X TIMESTEPS X CHANNELS -> B X CHANNELS X TIMESTEPS
        conv_outs = self.ngram_convolve(data["inputs"])
        split_list = self.split_ngram_conv_output(conv_outs)
        merged_ngram_sequence = self.merge_ngram_splits(split_list)
        out = self.pool_features(merged_ngram_sequence)

        # Shape: B X CHANNELS X TIMESTEPS -> B X TIMESTEPS X CHANNELS
        merged_ngram_sequence = merged_ngram_sequence.permute(0, 2, 1)

        # Shape: B X CHANNELS X TIMESTEPS -> B X TIMESTEPS X CHANNELS
        out = out.permute(0, 2, 1)
        data["out"] = out
        data["ngram_features"] = merged_ngram_sequence
        return data

