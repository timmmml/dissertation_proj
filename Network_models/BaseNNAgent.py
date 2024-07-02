'''Implements a base class for all neural network agents used in the present project.

Stage 0: feasibility check
    saved_model used: simple RNN structures (specify within config model type (RNN, GRU, LSTM) and hyperparams).


Stage 1 onwards...

'''
import torch
import torch.nn as nn
import Rotations as Rot

class BaseNNAgent(nn.Module):
    def __init__(self):
        super(BaseNNAgent, self).__init__()
        self.out = None
        self.pred = None # Final rotation made
        self.dt = 1.0 # Time step
        self.action_period = None

    def forward(self, x):
        raise NotImplementedError

    def predict(self):
        """Predict the final rotation made by the agent.

        Returns:
            pred (torch.Tensor(batch_size, 4)): The final rotation made by the agent, in quaternion
        """
        self.out = self.out.permute(1, 0, 2)
        self.pred = Rot.integrate_velocities(self.out[self.action_period:, : :], dt=self.dt)
        self.out = self.out.permute(1, 0, 2)
        return self.pred
