import torch as th
import torch.nn as nn

from torch.autograd import Variable


class RecurrentBlock(nn.Module):

    def __init__(self, input_size, hidden_size, bidirectional=False,
                 rnn_cell="LSTM", n_layers=1, dropout=0.5,
                 return_sequence=True):
        super(RecurrentBlock, self).__init__()

        self.input_size = input_size
        self.cell = self._get_rnn_cell(rnn_cell)
        self.hidden_size = hidden_size if not bidirectional else hidden_size // 2
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.dropout = dropout
        self.return_sequence = return_sequence
        self.rnn = self.cell(input_size=self.input_size,
                             hidden_size=self.hidden_size,
                             num_layers=self.n_layers,
                             bidirectional=self.bidirectional,
                             dropout=self.dropout,
                             batch_first=True)

    def _get_rnn_cell(self, rnn_cell):
        cell = None
        if rnn_cell.lower() == "rnn":
            cell = nn.RNN
        elif rnn_cell.lower() == "lstm":
            cell = nn.LSTM
        elif rnn_cell.lower() == "gru":
            cell = nn.GRU
        else:
            raise ValueError("{} cell is currently not supported or is an "
                             "invalid cell type. Supported Cell Types are: "
                             "lstm, rnn and gru.".format(rnn_cell))
        return cell

    def init_hidden(self, batch_size, cuda=False):
        d1 = self.n_layers if not self.bidirectional else self.n_layers * 2
        if cuda:
            h = Variable(th.zeros(d1, batch_size, self.hidden_size)).cuda()
            c = Variable(th.zeros(d1, batch_size, self.hidden_size)).cuda()
        else:
            h = Variable(th.zeros(d1, batch_size, self.hidden_size))
            c = Variable(th.zeros(d1, batch_size, self.hidden_size))
        ret = h if (type(self.rnn) is nn.GRU or type(self.rnn) is nn.RNN) \
                    else  (h, c)
        return ret

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their 
        history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def forward(self, data):
        output, hidden = self.rnn(data["inputs"], data["hidden"])
        resp = {"out": output if self.return_sequence \
                      else output.split(split_size=1, dim=1)[-1].squeeze()}
        resp["aux"] = {"hidden": hidden}
        return resp

