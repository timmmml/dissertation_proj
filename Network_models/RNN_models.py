"""This file contains the RNN saved_model used in the mental rotations project.

"""

import torch
import torch.nn as nn
from .BaseNNAgent import BaseNNAgent
import datetime

class CustomRNN(BaseNNAgent):
    """The CustomRNN model - a canonical RNN graph that accepts a cell-type argument.

    Args:
        input_size (int): The number of expected features in the input x
        hidden_size (int): The number of features in the hidden state h
        num_layers (int): Number of recurrent layers
        output_size (int): The number of output features
        cell_type (str): The type of RNN cell to use. Choose from ['RNN', 'GRU', 'LSTM']

    Flow:
        input -> RNN cell (hidden_size x num_layers) -> FC layer -> output
        input: (batch_size, seq_len, input_size)
        output: (batch_size, seq_len, output_size)
    """

    def __init__(self, model_params):
        super(CustomRNN, self).__init__()
        self.hidden_size = model_params["hidden_size"]
        self.num_layers = model_params["num_layers"]
        self.cell_type = model_params["cell_type"]
        self.out = None

        self.input_size = model_params["input_size"]
        self.output_size = model_params["output_size"]
        self.hidden_size = model_params["hidden_size"]
        self.num_layers = model_params["num_layers"]

        if "dt" in model_params:
            self.dt = model_params["dt"]

        if self.cell_type == "RNN":
            self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        elif self.cell_type == "GRU":
            self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        elif self.cell_type == "LSTM":
            self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        else:
            raise ValueError("Invalid cell type. Please choose from ['RNN', 'GRU', 'LSTM']")

        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        # Metricise time complexity of this:
        # print(f'start time: {datetime.datetime.now()}')
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        self.out = out
        #print(f'end forward time: {datetime.datetime.now()}')
        self.predict()
        #print(f'end predict time: {datetime.datetime.now()}')
        return out
