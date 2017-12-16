import torch
import torch.nn as nn

class NGramCNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, kernel_dim=100,
                 kernel_sizes=(3, 4, 5), dropout=0.5):
        super(NGramCNN, self).__init__()

        self.convs = nn.ModuleList(
                [nn.Conv2d(1, kernel_dim, (K, embedding_dim)) for K in
                 kernel_sizes])

        self.dropout = nn.Dropout(dropout)