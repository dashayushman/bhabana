import torch as th
import torch.nn as nn
import torch.nn.functional as F

from bhabana.utils import data_utils as du


class NGramCNN(nn.Module):

    requires = ["inputs", "training"]

    provides = ["out", "ngram_features", "layer_outs", "ngram_features"]

    def __init__(self, in_channels, n_layers=1, kernel_dim=100,
                 kernel_sizes=(3, 4, 5), dropout=0.5):
        super(NGramCNN, self).__init__()
        self._validate_constructor_params(in_channels, n_layers)
        self.in_channels = in_channels
        self.kernel_dim = kernel_dim
        self.kernel_sizes = kernel_sizes
        self.dropout_p = dropout
        self.conv_blocks = nn.ModuleList()

        for l in range(n_layers):
            convs = nn.ModuleList()
            if l != 0:
                in_channels = kernel_dim
            for k_size in kernel_sizes:
                convs.append(nn.Conv1d(in_channels, self.kernel_dim, k_size))
            self.conv_blocks.append(convs)

        self.dropout = nn.Dropout(dropout)

    def _validate_constructor_params(self, in_channels, n_layers):
        if in_channels is None:
            raise ValueError("in_channel cannot be None. Please provide an "
                             "integer value which is > 0")
        if type(in_channels).__name__ != 'int':
            raise ValueError("in_channel must be an int. Please provide a "
                             "valid int which must be > 0")
        if n_layers is None:
            raise ValueError("n_layers cannot be None. Please provide an "
                             "integer value which is > 0")
        if type(n_layers).__name__ != 'int':
            raise ValueError("n_layers must be an int. Please provide a "
                             "valid int which must be > 0")


    def _ngram_convolve(self, inputs, conv_block):
        conv_outs = []
        for conv, n_k in zip(conv_block, self.kernel_sizes):
            conv_outs.append(F.relu(conv((du.pad_1dconv_input(inputs,
                             kernel_size=n_k, mode="same")).permute(0, 2, 1))))
        return conv_outs

    def _split_ngram_conv_output(self, conv_outs):
        split_list = []
        for conv_out in conv_outs:
            # Shape: B X CHANNELS X TIMESTEPS -> B X TIMESTEPS X CHANNELS
            conv_out = conv_out.permute(0, 2, 1)
            split = [s.squeeze(dim=1) for s in th.split(conv_out, 1, dim=1)]
            split_list.append(split)
        return split_list

    def _merge_ngram_splits(self, split_list):
        merged_ngram_splits = zip(*split_list)
        merged_ngram_sequence = []
        for n_grams in merged_ngram_splits:
            merged_ngram_sequence.append(th.stack(n_grams, dim=0))
        merged_ngram_sequence = th.cat(merged_ngram_sequence, dim=0)

        # Shape: TIMESTEPS X B X CHANNELS -> B X CHANNELS X TIMESTEPS
        merged_ngram_sequence = merged_ngram_sequence.permute(1, 2, 0)
        return merged_ngram_sequence

    def _pool_features(self, merged_ngram_sequence):
        return F.max_pool1d(merged_ngram_sequence,
                           kernel_size=len(self.kernel_sizes),
                           stride=len(self.kernel_sizes))

    def _exec_conv_blocks(self, inputs, training):
        layer_outs, layer_ngram_features = [], []
        prev_input = inputs
        for conv_block in self.conv_blocks:
            conv_outs = self._ngram_convolve(prev_input, conv_block)
            split_list = self._split_ngram_conv_output(conv_outs)
            merged_ngram_sequence = self._merge_ngram_splits(split_list)
            out = self._pool_features(merged_ngram_sequence)

            if training:
                out = self.dropout(out)

            # Shape: B X CHANNELS X TIMESTEPS -> B X TIMESTEPS X CHANNELS
            merged_ngram_sequence = merged_ngram_sequence.permute(0, 2, 1)

            # Shape: B X CHANNELS X TIMESTEPS -> B X TIMESTEPS X CHANNELS
            out = out.permute(0, 2, 1)
            layer_outs.append(out)
            layer_ngram_features.append(merged_ngram_sequence)
            prev_input = out

        return layer_outs, layer_ngram_features

    def forward(self, data):
        # Input Shape: B X TIMESTEPS X CHANNELS -> B X CHANNELS X TIMESTEPS
        layer_outs, layer_ngram_features = self.__exec_conv_blocks(
                data["inputs"], data["training"])
        data["out"] = layer_outs[-1]
        data["ngram_features"] = layer_ngram_features[-1]
        data["layer_outs"] = layer_outs
        data["layer_ngram_features"] = layer_ngram_features

        return data