import torch.nn.functional as F
import torch.nn as nn


class RLNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, is_first=False):
        super(RLNet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.is_first = is_first

        self.layers = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.droupout = nn.ModuleList()  # drop out layer for regularization

        current_dim = self.input_dim
        for hdim in self.hidden_dim:
            self.layers.append(nn.Linear(int(current_dim), int(hdim)))
            self.bn.append(nn.BatchNorm1d(int(hdim)))
            self.droupout.append(nn.Dropout(0.25))  # add a dropout layer
            current_dim = hdim

        self.layers.append(nn.Linear(int(current_dim), int(self.output_dim)))

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.is_first == False:
                x = self.bn[i](x)
            x = F.tanh(x)
        out = self.layers[-1](x)
        return out
