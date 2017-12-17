import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MultiFilterCNN(nn.Module):

    requires = ["inputs", "training"]

    provides = ["out"]

    def __init__(self,in_channels, kernel_dim=100, kernel_sizes=(3, 4, 5),
                 dropout=0.5):
        super(MultiFilterCNN, self).__init__()

        self.in_channels = in_channels
        self.kernel_dim = kernel_dim
        self.kernel_sizes = kernel_sizes
        self.dropout_p = dropout
        self.convs = nn.ModuleList(
                [nn.Conv2d(1, kernel_dim, (K, in_channels)) for K in
                 kernel_sizes])

        # kernal_size = (K,D)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        inputs = data["inputs"].unsqueeze(1)  # (B,1,T,D)
        inputs = [F.relu(conv(inputs)).squeeze(3) for conv in
                  self.convs]  # [(N,Co,W), ...]*len(Ks)
        inputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in
                  inputs]  # [(N,Co), ...]*len(Ks)

        concated = th.cat(inputs, 1)
        if data["training"]:
            concated = self.dropout(concated)  # (N,len(Ks)*Co)
        data["out"] = concated
        return data