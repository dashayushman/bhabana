import torch as th
import torch.nn as nn

from torch.autograd import Variable
from bhabana.utils import generic_utils as gu


class Regressor(nn.Module):
    requires = ["inputs"]

    provides = ["out"]

    def __init__(self, input_size, activation=None):
        super(Regressor, self).__init__()

        self.input_size = input_size
        self.activation = gu.get_activation_by_name(activation)
        if self.activation is not None:
            self.activation = self.activation()
        self.fcl = nn.Linear(input_size, 1)

    def forward(self, data):
        resp = {"out": self.fcl(data["inputs"])}
        if self.activation is not None:
            resp["out"] = self.activation(resp["out"])
        return resp
