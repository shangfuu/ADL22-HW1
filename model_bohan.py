from typing import Dict

import torch
import torch.nn as nn
from torch.nn import Embedding


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num, class_num, batch_size):
        super(LSTM, self).__init__()
        # pass argument
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.class_num = class_num
        self.batch_size = batch_size
        # initial LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            layer_num, dropout=0.2, batch_first=True)
        self.linear = nn.Linear(hidden_dim, class_num)

    def forward(self, x):
        output, (hn, cn) = self.lstm(x, None)
        result = self.linear(hn[-1])
        return result


class LSTM_Tagger(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num, class_num, batch_size,max_len):
        super(LSTM_Tagger, self).__init__()
        # pass argument
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.class_num = class_num
        self.batch_size = batch_size
        self.max_len = max_len
        # initial LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            layer_num, dropout=0.3, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2 * hidden_dim, class_num)

    def forward(self, x):
        output, (hn, cn) = self.lstm(x, None)
        # print(output.shape)
        tag_seq = self.linear(output.view(len(x), self.max_len, -1))
        # print(tag_seq.shape)
        return tag_seq
