import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from bhabana.utils import generic_utils as gu


class Linear(nn.Module):
    ''' Simple Linear layer with xavier init '''
    def __init__(self, d_in, d_out, bias=True, activation=None):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        init.xavier_normal(self.linear.weight)
        self.activation = gu.get_activation_by_name(activation)
        if self.activation is not None:
            self.activation = self.activation()

    def forward(self, data):
        resp = {"out": self.linear(data["inputs"])}
        if self.activation is not None:
            logits = resp["out"]
            soft_probab = self.activation(logits)
            resp["out"] = soft_probab
        return resp