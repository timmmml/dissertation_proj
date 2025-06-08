""" This module implements the soc_lognormal_big network. 

"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.path_settings import *
from .BaseNNAgent import BaseNNAgent
from numba import jit
import Rotations as Rot
from scipy.linalg import solve_continuous_lyapunov, expm
from utils import goto_project_root
 

network = pd.read_csv(
    OBJECT_PATH + "/network_dynamics/soc_lognormal_big.txt", sep="\t", header=None
).to_numpy()[:, :-1]
torch.set_default_dtype(torch.float32)


@jit(nopython=True)
def autonomous_dynamics(x0, action_period, W, tau, dt):
    """This method computes the autonomous dynamics of the network given the initial condition x.

    Args:
        x0 (np.array(batch_size, self.state_dim)): the initial condition tensor

    Returns:
        np.array(batch_size, self.state_dim): the final state of the network after the autonomous dynamics
    """
    x = np.zeros((x0.shape[0], action_period, x0.shape[-1]), dtype=np.float32)
    x[:, 0, :] = x0
    for i in range(1, action_period):
        y = x[:, i - 1, :] + (-x[:, i - 1, :] + (W @ x[:, i - 1, :].T).T) / tau * dt
        x[:, i, :] = y
    return x


@jit(nopython=True)
def input_dynamics(x0, action_period, W, tau, dt, I):
    """This method computes the dynamics of the network given the initial condition x and the input dynamics.

    Args:
        I (np.array(batch_size, full_seq_length, self.state_dim)): the input

    Returns:
        np.array(batch_size, self.state_dim): the final state of the network after the dynamics
    """
    x = np.zeros((x0.shape[0], action_period, x0.shape[-1]), dtype=np.float32)
    if I.shape[1] != action_period:
        # concatenate 0s to the input to match the full sequence length
        I = np.concatenate(
            (I, np.zeros((I.shape[0], action_period - I.shape[1], I.shape[-1]))), axis=1
        )
    for i in range(1, action_period):
        x[:, i, :] = (
            x[:, i - 1, :]
            + (-x[:, i - 1, :] + (W @ x[:, i - 1, :].T).T + I[:, i - 1, :]) / tau * dt
        )
    return x

# @torch.jit.script
# def controller_dynamics(x0,  W, tau, dt, controller: torch.jit.ScriptModule, control_period = 0, control_threshold = 0, noise_level = 0):
#     """This method computes the dynamics of the network given the initial condition x and the input dynamics.

#     Args:
#         controller: a controller object that _extends the torch.jit.ScriptModule class_ whose forward method returns a tensor of shape (batch_size, 1, state_dim) that is the control signal to feed in on each timestep
#         control_period: the number of timesteps to control for (default is None);
#         control_threshld: the threshold to stop controlling at (default is None);
#         noise_level: the level of noise to add to the control signal (default is 0);

#     Returns:
#         np.array(batch_size, self.state_dim): the final state of the network after the dynamics
#     """
#     assert control_period != 0 or control_threshold != 0, "Either control_period or control_threshold must be specified"
#     assert control_period == 0 or control_threshold == 0, "Only one of control_period or control_threshold can be specified"

#     x = x0.unsqueeze(1) 
#     time = 0
#     upper_bound = 1000
#     error_trajectory = torch.zeros(x.shape[0], 0)
#     while True: 
#         time += 1
#         if time == control_period: 
#             break
#         control_input, error = controller(x[:, -1, :])
#         error_trajectory = torch.cat([error_trajectory, error.unsqueeze(1)], dim=1)
#         if error < control_threshold: 
#             break
#         if time > upper_bound: 
#             print(f"Control period exceeded upper bound with residual error {error}")
#             break
#         x_new = x[:, -1, :] + (-x[:, -1, :] + (W @ x[:, -1, :].T).T + control_input) / tau * dt
#         x = torch.cat([x, x_new.unsqueeze(1)], dim=1)
#     return x


class DynamicalNetwork(BaseNNAgent):
    def __init__(self, model_params):
        super(DynamicalNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_dtype(torch.float32)
        torch.set_default_device(self.device)
        self.W = torch.tensor(network, dtype=torch.get_default_dtype())
        self.W.requires_grad = False
        self.state_dim = self.W.shape[1]
        self.output_size = model_params.get("output_size", 3)
        self.k = model_params.get("k", 0.05)
        torch.manual_seed(model_params.get("seed", 0))
        self.Optimise_C = model_params.get("Optimise_C", False)
        if self.Optimise_C:
            self.C = nn.Parameter(torch.tensor(self._optimise_C()))
        else:
            self.C = torch.randn(self.output_size, self.W.shape[1])
            self.C = nn.Parameter(self.C / torch.norm(self.C, dim=1).reshape(-1, 1))

        self.freeze_C = model_params.get("freeze_C", False)
        self.C.requires_grad = not self.freeze_C

        self.U = None
        self.calculate_U()

        if self.freeze_C and model_params.get("automatically_plot", False): 
            self.plot_spectra()
        self.forward_mode = model_params.get("mode", "initial_condition")
        self.tau = model_params.get("tau", 100)  # 100 ms
        self.dt = model_params.get("dt", 1)  # 1 ms
        self.total_period = model_params.get("total_period", 1000)
        self.action_period = model_params.get("action_period", 500)  # 500 ms
        # these are going to be altered by the trainer wrapper of the models

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.t_full = np.arange(0, self.total_period, self.dt)
        self.t_action = np.arange(0, self.action_period, self.dt)
        self = self.to(self.device)
        self.stepwise = model_params.get("stepwise", False)
        self.out_accel = model_params.get("out_accel", True)
        self.tau_accel = model_params.get("tau_accel", self.tau)

        self.leak_method = model_params.get("accel_leak_method", 0)
        self.input_scalar = model_params.get("input_scalar", 1)
        self.controller = None # controller object will have a built-in predictor. 

    
    def forward(self, x, controller = None, control_period = None, control_threshold = None):
        """This method computes a forward pass of the network given a few situations.

        case self.forward_mode = "initial_condition" -> x is the initial condition

        Computes the autonomous dynamics of the network given the initial condition x;

        Args:
            x (torch.Tensor(batch_size, self.state_dim)): the initial condition tensor


        case: self.forward_mode = "dynamics_input" -> x is the dynamics input

        Accept inputs as a time series of injected currents and integrate for an initial condition, then release the dynamics on action phase;

        Args:
            x (torch.Tensor(batch_size, seq_len, self.state_dim)): the dynamics input tensor

        Returns:
            ?
        """

        assert (
            x.dtype == torch.get_default_dtype()
        ), f"Input tensor must be of type {torch.get_default_dtype()}"

        if x.dim() == 1:
            x = x.unsqueeze(0)
        match self.forward_mode:
            case "initial_condition":
                z = torch.tensor(
                    autonomous_dynamics(
                        x.cpu().numpy(),
                        self.action_period,
                        self.W.cpu().numpy(),
                        self.tau,
                        self.dt,
                    ),
                    dtype=torch.get_default_dtype(),
                    device=self.device,
                )
            case "dynamics_input":
                z = torch.tensor(
                    input_dynamics(
                        x.cpu().numpy(),
                        self.total_period,
                        self.W.cpu().numpy(),
                        self.tau,
                        self.dt,
                    ),
                    dtype=torch.get_default_dtype(),
                    device=self.device,
                )
            # case "controller": 
            #     assert controller is not None, "Controller must be specified"
            #     z = controller_dynamics(x, self.W, self.tau, self.dt, controller, control_period, control_threshold).to(self.device)
            #     z_init_condition = z[:, -1, :]
            #     z = torch.tensor(
            #         autonomous_dynamics(
            #             z_init_condition.cpu().numpy(),
            #             self.action_period,
            #             self.W.cpu().numpy(),
            #             self.tau,
            #             self.dt,
            #         ),
            #         dtype=torch.get_default_dtype(),
            #         device=self.device,
            #     )
        self.z = z
        z_y = z[:, -self.action_period :, :]
        # z_y is of shape (batch_size, action_period, state_dim)
        # C is of shape (output_size, state_dim)
        out_accel = torch.einsum("ij, bjk->bik", self.C, z_y.permute(0, 2, 1)).permute(
            0, 2, 1
        )
        if getattr(self, "out_accel", True): 
            print('Output acceleration')
            if self.leak_method == 0:
                out = torch.cumsum(out_accel, dim=1) * self.dt 
            if self.leak_method == 1: 
                out = torch.cumsum(out_accel, dim=1) * self.dt * self.k_traj
            if self.leak_method == 2: 
                out = torch.zeros_like(out_accel)
                for i in range(out_accel.shape[1]): 
                    if i == 0: 
                        out[:, i, :] = out_accel[:, i, :] * self.dt/self.tau_accel
                    else: 
                        out[:, i, :] = (1 - self.dt/self.tau_accel) * out[:, i-1, :] + out_accel[:, i, :] * self.dt/self.tau_accel * self.input_scalar
        else: 
            print('Output velocity')
            out = out_accel * self.dt * self.k_traj
        self.out = out.float()
        self.predict()
        return out
    
    def discretise(self, x, dT, steps): 
        """per some initial condition, calculate a sequence of states for the network. 
        
        Args:
            x (torch.Tensor(batch_size, self.state_dim)): the initial condition tensor
            dT (float): bin size for discretisation  
            steps (int): number of steps to discretise for.
        """ 
        eW = torch.tensor(expm((self.W.cpu().numpy() - np.eye(self.W.shape[0])) * dT/self.tau), dtype = torch.get_default_dtype())
        eW = eW.to(x.device)
        for i in range(steps): 
            x = torch.cat((x, (eW @ x[-1,:]).unsqueeze(0)), dim=0)
        return x
        
    def rotate(self, q_in):
        # Currently implemented is a proof-of-concept example rotation.torch.tensor(
        # Now, this is a random rotation just to see the outputs of the model
        print(self.k)
        silence_period = self.total_period - self.action_period
        traj = self(
            self.sample_initial_condition(1) * self.input_scalar,
        )
        # Take the last 500 steps of outputs and make them a list of 500 quaternions by exponential map
        traj = traj[0, -self.action_period :, :].detach()
        traj = Rot.exp_quat(traj * self.dt)
        return traj

    def rotate_z0(self, z0):
        # Currently implemented is a proof-of-concept example rotation.
        # Now, this is a random rotation just to see the outputs of the model
        traj = self(
            z0 
        )
        # Take the last 500 steps of outputs and make them a list of 500 quaternions by exponential map
        traj = traj[0, -self.action_period :, :].detach()
        traj = Rot.exp_quat(traj * self.dt)
        return traj

    def sample_initial_condition(self, batch_size, use_gramian = True):
        if use_gramian:
            if self.U is None:
                W = self.W.cpu().numpy()
                C = self.C.cpu().numpy()
                Q = solve_continuous_lyapunov(W.T, -C.T @ C)
                self.U, _, _ = np.linalg.svd(Q)
            # Utop = U[:,0:4]
            samples = self.U @ np.random.randn(self.U.shape[0], batch_size)
        else: 
            samples = np.random.randn(self.state_dim, batch_size)
        return torch.tensor(
            samples.T, dtype=torch.get_default_dtype(), device=self.device
        )

    def calculate_U(self):
        W = self.W.cpu().numpy()
        C = self.C.cpu().numpy()
        Q = solve_continuous_lyapunov(W.T, -C.T @ C)
        self.U, _, _ = np.linalg.svd(Q)

    def eval_k(self, k=None):
        if k is None:
            k = self.k
        # produce a train of e^{-kt}
        t = np.arange(0, self.action_period * self.dt, self.dt)
        self.k_traj = (
            torch.tensor(
                np.exp(-k * t), dtype=torch.get_default_dtype(), device=self.device
            )
            .unsqueeze(0)
            .unsqueeze(-1)
        )
    
    def _optimise_C(self):
        # Solve a few Lyaponov equations to get C: 
        W = self.W.cpu().numpy()
        Q0 = solve_continuous_lyapunov(W.T, -np.eye(W.shape[0]))
        P0 = solve_continuous_lyapunov(W, -np.eye(W.shape[0]))
        U, _, _ = np.linalg.svd(P0)
        Utop = U[:,0:3]
        C = Utop.T
        return C

    def plot_spectra(self): 
        # Plot the singular values of the W matrix and fraction of variance explained on each step, two subplots
        import matplotlib.pyplot as plt
        W = self.W.cpu().numpy()
        C = self.C.cpu().numpy()
        Q = solve_continuous_lyapunov(W.T, -C.T @ C)

        # Do W first
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        _, S, _ = np.linalg.svd(W)
        ax[0].plot(S)
        ax[0].set_title('singular values of W')
        ax[1].plot(np.cumsum(S)/np.sum(S))
        ax[1].set_title('Fraction of variance explained by W')
        
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        _, S, _ = np.linalg.svd(Q)
        ax[0].plot(S)
        ax[0].set_title('singular values of Q')
        ax[1].plot(np.cumsum(S)/np.sum(S))
        ax[1].set_title('Fraction of variance explained by Q')

    def refresh(self): 
        self.out = None
        self.pred = None


