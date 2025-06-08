'''Implements a base class for all neural network agents used in the present project.

Stage 0: feasibility check
    saved_model used: simple RNN structures (specify within config model type (RNN, GRU, LSTM) and hyperparams).


Stage 1 onwards...

'''
import torch
import torch.nn as nn
import Rotations as Rot
import scipy.spatial.transform as R
from utils.path_settings import OBJECT_PATH

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
            case not stepwise (final rotation):
                pred (torch.Tensor(batch_size, 4)): The final rotation made by the agent, in quaternion
            case stepwise (stepwise rotation trajectory):
                pred (torch.Tensor(batch_size, action_period, 4)): The stepwise rotation trajectory made by the agent, in quaternion
        """
        if not self.stepwise:
            self.out = self.out.permute(1, 0, 2)
            self.pred = Rot.integrate_velocities(self.out[-self.action_period:, : :], dt=self.dt) if self.action_period is not None else Rot.integrate_velocities(self.out, dt=self.dt)
            self.out = self.out.permute(1, 0, 2)
        else:
            self.out_quat = Rot.exp_quat(self.out[:, -self.action_period:, :] * self.dt) if self.action_period is not None else Rot.exp_quat(self.out * self.dt)
            self.out_quat = self.out_quat.permute(1, 0, 2)
            self.pred = Rot.integrate_quat_sequential(self.out_quat)
            self.out_quat = self.out_quat.permute(1, 0, 2)
        return self.pred

    def evaluate_q(self, q_in, returned_list = ["z0"], z0_time = None):
        assert hasattr(self, "input_size") and hasattr(self, "action_period")
        # split q_in up into chunks of 4096:
        beg = 0
        chunk_size = 4096
        q_in = q_in.to(self.device)
        q_in_ = q_in.clone()
        q_in_list = []
        while q_in.shape[0] - beg > chunk_size: 
            q_in_list.append(q_in[beg:beg + chunk_size])
            beg += chunk_size
        q_in_list.append(q_in[beg:])
        
        hidden = None
        pred = None
        for q_in in q_in_list:
            q = q_in * q_in[:, :1].sign()
            match self.input_size:
                case 8: 
                    q = torch.cat((q[:, :1], q[:, 1:]), dim=-1)
                case 9: 
                    q = torch.tensor(R.from_quat(q.cpu().numpy(), scalar_first=True).as_matrix()).float().view(-1, 9)
                case 24:
                    mesh = torch.load(OBJECT_PATH + f"/cube_mesh/cube_mesh_full.pth")
                    q = mesh.quat2verts(q).reshape(-1, 24)
                case 16:
                    mesh = torch.load(OBJECT_PATH + f"/cube_mesh/cube_mesh_camera_simple.pth")
                    q = mesh.quat2verts(q).reshape(-1, 24)
            q = q.unsqueeze(1)
            q = q.repeat(1, self.total_period - self.action_period, 1)
            q = torch.cat((q, torch.zeros(q.shape[0], self.action_period, q.shape[-1]).to(q.device)), dim=1)
            # register forward hook to get the rnn trajectory
            self.hidden = []
            returned = {}
            self.rnn.register_forward_hook(self.get_out)
            out = self.forward(q)
            if hidden is None: 
                hidden = self.hidden[0]
            else: 
                hidden = torch.cat((hidden, self.hidden[0]), dim=0)
            if pred is None: 
                pred = self.pred[:, -1, :]
            else: 
                pred = torch.cat((pred, self.pred[:, -1, :]), dim=0)

        if "z0" in returned_list:
            if z0_time is None:
                returned["z0"] = hidden[:, self.total_period - self.action_period, :].detach()
            elif z0_time == "all": 
                returned["z0"] = hidden[:, :, :].detach()
            elif type(z0_time) == int:
                returned["z0"] = hidden[:, z0_time, :].detach()
        if "q_out_error" in returned_list: 
            q_out = pred 
            q_in = q_in_.to(q_out.device)
            returned["q_out_error"] = [Rot.geodesic_distance(q_out[i], q_in[i]) for i in range(q_in.shape[0])]
        return returned

    def get_out(self, module, input, output):
        self.hidden = output
           