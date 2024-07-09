"""This file contains the RNN saved_model used in the mental rotations project.

"""

import torch
import torch.nn as nn
from .BaseNNAgent import BaseNNAgent
import Rotations as Rot
from numba import jit
from pytorch3d import transforms

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
        self.cell_type = model_params["cell_type"]
        self.input_size = model_params["input_size"]
        self.output_size = model_params["output_size"]
        self.hidden_size = model_params["hidden_size"]
        self.num_layers = model_params["num_layers"]
        self.action_period = 500
        self.total_period = 1000
        self.stepwise = model_params["stepwise"]

        self.out = None
        self.pred = None
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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device).float()

        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        self.out = out.float()
        self.predict()
        return out
    #
    # def forward_pytorch3d(self, x):
    #     h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
    #     out, _ = self.rnn(x, h0)
    #     out = self.fc(out)
    #     self.predict_pytorch3d()
    #     return out
    #
    # @jit(nopython = True)
    # def predict_pytorch3d(self):
    #     pass

    def rotate(self, q_in):
        silence_period = self.total_period - self.action_period
        if q_in[0] < 0:
            q_in = -q_in
        q_in = torch.cat((q_in, -q_in), dim=0)
        q_in = q_in.unsqueeze(0).unsqueeze(0)
        q_in = q_in.repeat(1, silence_period, 1)
        q_in = torch.cat((q_in, torch.zeros(1, self.action_period, 8).to(q_in.device)), dim=1)
        q_in = q_in.to(self.fc.bias.device)
        traj = self(q_in)
        # Take the last 500 steps of outputs and make them a list of 500 quaternions by exponential map
        traj = traj[0, -self.action_period:, :].detach()
        traj = Rot.exp_quat(traj * self.dt)
        return traj
class FC_RNN(BaseNNAgent):
    """CustomRNN + FC layer in front

    Args:
        input_size (int): The number of expected features in the input x
        hidden_size (int): The number of features in the hidden state h
        num_layers (int): Number of recurrent layers
        output_size (int): The number of output features
        cell_type (str): The type of RNN cell to use. Choose from ['RNN', 'GRU', 'LSTM']
        FC_dim (int): The number of features in the FC layer

    Flow:
        input -> RNN cell (hidden_size x num_layers) -> FC layer -> output
        input: (batch_size, seq_len, input_size)
        output: (batch_size, seq_len, output_size)
    """

    def __init__(self, model_params):
        super(FC_RNN, self).__init__()
        self.cell_type = model_params["cell_type"]
        self.input_size = model_params["input_size"]
        self.output_size = model_params["output_size"]
        self.hidden_size = model_params["hidden_size"]
        self.num_layers = model_params["num_layers"]
        self.FC_dim = model_params["FC_dim"]
        self.action_period = 500
        self.total_period = 1000
        self.stepwise = model_params["stepwise"]

        self.out = None
        self.pred = None
        if "dt" in model_params:
            self.dt = model_params["dt"]

        if self.cell_type == "RNN":
            self.rnn = nn.RNN(self.FC_dim, self.hidden_size, self.num_layers, batch_first=True)
        elif self.cell_type == "GRU":
            self.rnn = nn.GRU(self.FC_dim, self.hidden_size, self.num_layers, batch_first=True)
        elif self.cell_type == "LSTM":
            self.rnn = nn.LSTM(self.FC_dim, self.hidden_size, self.num_layers, batch_first=True)
        else:
            raise ValueError("Invalid cell type. Please choose from ['RNN', 'GRU', 'LSTM']")

        self.fc_in = nn.Sequential(
            nn.Linear(self.input_size, self.FC_dim),
            nn.ReLU())
        self.fc_out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x = self.fc_in(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device).float()
        out, _ = self.rnn(x, h0)
        out = self.fc_out(out)
        self.out = out.float()
        self.predict()
        return out

    def rotate(self, q_in):
        silence_period = self.total_period - self.action_period
        if q_in[0] < 0:
            q_in = -q_in
        q_in = torch.cat((q_in, -q_in), dim=0)
        q_in = q_in.unsqueeze(0).unsqueeze(0)
        q_in = q_in.repeat(1, silence_period, 1)
        q_in = torch.cat((q_in, torch.zeros(1, self.action_period, 8).to(q_in.device)), dim=1)
        q_in = q_in.to(self.fc_out.bias.device)
        traj = self(q_in)
        # Take the last 500 steps of outputs and make them a list of 500 quaternions by exponential map
        traj = traj[0, -self.action_period:, :].detach()
        traj = Rot.exp_quat(traj * self.dt)
        return traj
