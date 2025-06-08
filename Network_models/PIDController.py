import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from typing import Mapping, Optional, Dict, Union, List
from utils.path_settings import *
from copy import deepcopy
from scipy.linalg import expm
from .Activations import *
import numpy as np

import time


# Let's try simple PID control on the cube now.
# Define a controller
class PIDController(torch.jit.ScriptModule):
    def __init__(
        self,
        tau,
        dt,
        Kp,
        Ki,
        Kd,
        W,
        predictive_model,
        output_size=16,
        target_size=200,
        device="cuda",
    ):
        super(PIDController, self).__init__()
        self.tau = tau
        self.predictive_model = torch.jit.script(predictive_model)
        for param in self.predictive_model.parameters():
            param.requires_grad = False
        self.dt = dt

        self.Kp = nn.Parameter(torch.tensor(Kp, device=device))
        # self.Ki = nn.Parameter(torch.tensor(Ki, device=device))
        # self.Kd = nn.Parameter(torch.tensor(Kd, device=device))
        self.Ki = 0
        self.Kd = 0

        self.output_size = output_size  # NOTE This may be 16 or 24 - depending on the configuration of the camera (we could either have 2d image or an extra dimension)
        self.target_size = target_size
        self.integral = torch.zeros(1, output_size, device=device)
        self.prev_error = torch.zeros(1, output_size, device=device)
        self.B = nn.Parameter(torch.zeros(target_size, output_size))

        # Set the non-zero values according to the sparsity
        sparsity = 0.1
        num_nonzero = int(target_size * output_size * sparsity)
        row_indices = torch.randint(0, target_size, (num_nonzero,))
        col_indices = torch.randint(0, output_size, (num_nonzero,))
        indices = torch.stack([row_indices, col_indices])
        values = torch.randn(num_nonzero)

        # Populate the dense tensor with sparse-like values
        self.B.data[indices[0], indices[1]] = values

        self.W = W.to(device)
        self.device = device
        self.control_input = torch.zeros(1, output_size, device=device)

    @torch.jit.script_method
    def forward(
        self,
        x0,
        target,
        control_period: Optional[int] = None,
        control_threshold: Optional[float] = None,
        period_upperlimit: int = 10000,
    ):
        """Forward control loop, defined either to run a number of steps or reach a certain threshold.

        NOTE in the training loop it may be more straightforward to constrain the number of steps, then push the control dynamics to be short in the loss function

        Though we can add noise to this, as a first step we assume there is no noise whatsoever.
        """
        # brings x0 to target either for control period or until control threshold is reached;
        # or, when unsuccessful, until period_upperlimit is reached.
        assert control_period is not None or control_threshold is not None
        assert control_period is None or control_threshold is None
        self.integral = torch.zeros(x0.shape[0], self.output_size, device=self.device)
        self.prev_error = torch.zeros(x0.shape[0], self.output_size, device=self.device)
        x = x0.unsqueeze(1)
        # print(f"x shape: {x.shape}")
        time = 0
        error_trajectory = torch.zeros(x.shape[0], 0, device=self.device)
        while True:
            time += 1
            print(f"Time: {time}")
            if isinstance(control_period, int) and time == control_period:
                break
            if (
                isinstance(control_threshold, float)
                and (x[:, -1, :] - target).norm(p=2).mean() < control_threshold
            ):
                break
            control_input, error = self.step_controller(x[:, -1, :], target)
            error_trajectory = torch.cat([error_trajectory, error.unsqueeze(1)], dim=1)
            if time > period_upperlimit:
                print(f"Control period exceeded upper bound with residual error")
                break
            x_new = (
                x[:, -1, :]
                + (-x[:, -1, :] + (self.W @ x[:, -1, :].T).T + control_input)
                / self.tau
                * self.dt
            )
            x = torch.cat([x, x_new], dim=1)
        # error_trajectory = (x[:, 1:, :] - target.unsqueeze(1)).norm(p = 2, dim = -1)
        return x, error_trajectory

    @torch.jit.script_method
    def forward_no_predictor(
        self,
        x0,
        target,
        control_period: Optional[int] = None,
        control_threshold: Optional[float] = None,
        period_upperlimit: int = 1000,
    ):
        # brings x0 to target either for control period or until control threshold is reached;
        # or, when unsuccessful, until period_upperlimit is reached.
        assert control_period is not None or control_threshold is not None
        assert control_period is None or control_threshold is None

        x = x0.unsqueeze(1)
        time = 0
        error_trajectory = torch.zeros(x.shape[0], 0, device=self.device)
        self.integral = torch.zeros(x0.shape[0], 200, device=self.device)
        self.prev_error = torch.zeros(x0.shape[0], 200, device=self.device)
        while True:
            time += 1
            if isinstance(control_period, int) and time == control_period:
                break
            control_input, error = self.step_controller_no_predictor(
                x[:, -1, :], target
            )
            self.control_input = control_input
            error_trajectory = torch.cat([error_trajectory, error.unsqueeze(1)], dim=1)
            if isinstance(control_threshold, float) and error < control_threshold:
                break
            if time > period_upperlimit:
                print(
                    f"Control period exceeded upper bound with residual error {error}"
                )
                break
            x_new = (
                x[:, -1, :]
                + (-x[:, -1, :] + (self.W @ x[:, -1, :].T).T + control_input)
                / self.tau
                * self.dt
            )
            x = torch.cat([x, x_new], dim=1)
        return x, error_trajectory

    @torch.jit.script_method
    def step_controller(self, zt, target=None):
        """Gives one step update of the controller"""
        if target is None:
            target = self.target  # in that case resort to default target
        predicted = self.predictive_model(zt)
        # print(f"taget shape {target.shape}; predicted shape {predicted.shape}")
        predicted = predicted[..., -1, :].view(-1, self.output_size)
        # print(f"predicted shape: {predicted.shape}")
        # print(f"target shape {target.shape}; predicted shape {predicted.shape}")

        error = target - predicted
        # self.integral = self.integral + error * self.dt
        # derivative = (error - self.prev_error) / self.dt
        # self.prev_error = error
        u = self.Kp * error  # + self.Ki * self.integral + self.Kd * derivative
        return (torch.exp(self.B.unsqueeze(0)) @ u.unsqueeze(-1)).permute(
            0, 2, 1
        ), error.norm(p=2, dim=-1) ** 2

    @torch.jit.script_method
    def step_controller_no_predictor(self, zt, target=None):
        """Gives one step update of the controller; thit time target is in state space"""
        if target is None:
            target = self.target  # in that case resort to default target

        # print(f"taget shape {target.shape}; predicted shape {predicted.shape}")
        error = target - zt
        self.integral = self.integral + error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        u = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        # print(f"u shape: {u.shape}")

        return u.unsqueeze(0), error.norm(p=2, dim=-1) ** 2


# NOTE: tackling plan for these RNN-based control networks: several knobs to turn
# 1. Whether we operate on state space directly (or inject inputs to the original dynamics).
# 2. Whether we use feedforward or recurrent control. (feedforward learns an inverse function directly; recurrent uses a few steps to take us there)
# 3. Whether we control from a random initial condition or from fixed initial condition (0).

# NOTE: Loss functions:
# BASE: we use MSE to constrain the output image difference (between predicted and actual).
# 1. we can constrain input energy
# 2. we can constrain state likelihood (from the initial conditions)
#   or just the state energy (let's imaging a flat state prior)
#       Constraining the state to fall in the prior ensures that states we arrive at will be suitbale for continual control (if we train based on this prior)
#       We can train this prior too. Start with the Gramian from which the samples are drawn to train the predictor.


class ControlNet(nn.Module):  # (torch.jit.ScriptModule):
    """This module is an RNN controller that takes concatenated reference signal (picture space) and predicted state output (picture space) as inputs and holds the current state as hidden.

    - NOTE: the thing operates directly in state space and hence disregards the original system dynamics.

    """

    def __init__(
        self,
        predictive_model,
        dt=1,
        tau=100,
        output_size=16,
        target_size=200,
        W=None,
        inject_inputs=False,  # set this to True to inject inputs; else set states
        recurrent_control=False,  # set this to toggle if the previous state information is fed along with the prediction.
        pure_error=True,  # if we want only allow error feedback
        hidden_control_dim=0,  # if this is more than 0, we would require using a readout matrix.
        readout_layer=False,
        activation="tanh",
        device="cuda",
    ):
        super(ControlNet, self).__init__()

        assert (hidden_control_dim and readout_layer) or (not hidden_control_dim)

        self.predictive_model = (
            torch.jit.script(predictive_model)
            if isinstance(self, torch.jit.ScriptModule)
            else predictive_model
        )

        for param in self.predictive_model.parameters():
            param.requires_grad = False

        self.dt = dt
        self.tau = tau
        self.W = W
        self.output_size = output_size
        self.feedback_size = (
            output_size if pure_error else 2 * output_size
        )  # NOTE This may be 16 or 24 - depending on the configuration of the camera (we could either have 2d image or an extra dimension)
        self.hidden_size = (
            hidden_control_dim if hidden_control_dim != 0 else target_size
        )

        # collate the knobs that we specify
        self.inject_inputs = inject_inputs
        self.recurrent_control = recurrent_control

        self.W_hh = nn.Linear(self.hidden_size, self.hidden_size) 
        self.W_xh = nn.Linear(self.feedback_size, self.hidden_size)

        self.activation = (
            nn.Tanh()
            if activation == "tanh"
            else (nn.ReLU() if activation == "relu" else nn.Identity())
        )

        if readout_layer:
            self.output = nn.Sequential(
                nn.ReLU(), nn.Linear(self.hidden_size, target_size)
            )
        else:
            self.output = nn.Identity()

        self.device = device
        self.W_hh = self.W_hh.to(device)
        self.W_xh = self.W_xh.to(device)
        self.mesh = None
        self.mode = None
        self.dynamics_model = None

    # @torch.jit.script_method
    def forward(
        self,
        x0,
        target,
        h0=None,
        control_period: Optional[int] = None,
        control_threshold: Optional[float] = None,
        period_upperlimit: int = 100,
        return_control_trajectory=False,
    ):
        """Forward control loop, defined either to run a number of steps or reach a certain threshold.

        Here, we could train the system to perform as little steps as possible or constrain the total step number.
        """
        assert control_period is not None or control_threshold is not None
        assert control_period is None or control_threshold is None

        tiny_segment_length = 2048  # to avoid memory overflow, we divide inputs into tiny segments when it's too big
        error_trajectory = None
        if x0.shape[0] > tiny_segment_length:
            left = x0.shape[0]
            segment_start = 0
            while left > 0:
                x_segment = x0[
                    segment_start : (segment_start + min(tiny_segment_length, left))
                ]
                target_segment = target[
                    segment_start : (segment_start + min(tiny_segment_length, left))
                ]
                if return_control_trajectory:
                    x_, control_trajectory_, error_trajectory_ = self.forward(
                        x_segment,
                        target_segment,
                        h0,
                        control_period,
                        control_threshold,
                        period_upperlimit,
                        return_control_trajectory,
                    )
                    x, control_trajectory, error_trajectory = (
                        (x_, control_trajectory_, error_trajectory_)
                        if error_trajectory is None
                        else (
                            torch.cat([x, x_], dim=0),
                            torch.cat([control_trajectory, control_trajectory_], dim=0),
                            torch.cat([error_trajectory, error_trajectory_], dim=0),
                        )
                    )
                else:
                    x_, error_trajectory_ = self.forward(
                        x_segment,
                        target_segment,
                        h0,
                        control_period,
                        control_threshold,
                        period_upperlimit,
                        return_control_trajectory,
                    )
                    x, error_trajectory = (
                        (x_, error_trajectory_)
                        if error_trajectory is None
                        else (
                            torch.cat([x, x_], dim=0),
                            torch.cat([error_trajectory, error_trajectory_], dim=0),
                        )
                    )
                left -= tiny_segment_length
                segment_start += tiny_segment_length
            if return_control_trajectory:
                return x, control_trajectory, error_trajectory
            return x, error_trajectory

        if h0 is None:
            h0 = torch.zeros(x0.shape[0], self.hidden_size, device=self.device)

        x = x0.unsqueeze(1)
        time = 0
        error_trajectory = torch.zeros(x.shape[0], 0, device=self.device)
        control_trajectory = torch.zeros(x.shape[0], 0, x.shape[-1], device=self.device)
        ht = h0
        while True:
            if isinstance(control_period, int) and time == control_period:
                break
            if (
                isinstance(control_threshold, float)
                and (x[:, -1, :] - target).norm(p=2).mean() < control_threshold
            ):
                break

            control_input, ht, error = self.step_controller(x[:, -1, :], ht, target)
            error_trajectory = torch.cat([error_trajectory, error.unsqueeze(1)], dim=1)
            if time > period_upperlimit:
                print(f"Control period exceeded upper bound with residual error")
                break
            if self.inject_inputs:
                x_new = (
                    x[:, -1, :]
                    + (-x[:, -1, :] + (self.W @ x[:, -1, :].T).T) / self.tau * self.dt
                    + control_input
                )
                control_trajectory = torch.cat(
                    [control_trajectory, control_input], dim=1
                )
            else:
                x_new = control_input
                if return_control_trajectory:
                    # print(f"control input shape {control_input.shape}")
                    # print(f"control trajectory shape {control_trajectory.shape}")
                    control_input = control_input - (
                        x[:, -1, :]
                        + (-x[:, -1, :] + (self.W @ x[:, -1, :].T).T)
                        / self.tau
                        * self.dt
                    ).unsqueeze(1)
                    control_trajectory = torch.cat(
                        [control_trajectory, control_input], dim=1
                    )
            x = torch.cat([x, x_new], dim=1)
            time += 1
        # error_trajectory = (x[:, 1:, :] - target.unsqueeze(1)).norm(p = 2, dim = -1)
        if return_control_trajectory:
            return x, control_trajectory, error_trajectory
        return x, error_trajectory

    # @torch.jit.script_method
    def step_controller(self, zt, ht, target=None):
        """Gives one step update of the controller"""
        if target is None:
            target = self.target

        predicted = self.predictive_model(zt)
        predicted = predicted[..., -1, :].view(-1, self.output_size)
        if self.feedback_size == target.shape[-1]:
            x_xh = self.W_xh(target - predicted)
        else:
            x_xh = self.W_xh(torch.cat([target, predicted], dim=-1))

        if self.recurrent_control:
            x_hh = self.W_hh(ht)
            ht = self.activation(x_hh + x_xh)
        else:
            ht = self.activation(x_xh)

        yt = self.output(ht).unsqueeze(1)
        return yt, ht, (target - predicted).norm(p=2, dim=-1) ** 2

    def rotate(self, q_in):
        """This method prepares a preparatory state and sets the dynamical network to unroll"""
        assert q_in.view(-1, 4).shape[0] == 1
        image = self.mesh.quat2verts(q_in / q_in.norm(dim=-1, keepdim=True)).view(
            -1, 24 if self.mode != "camera_simple" else 16
        )
        z, _ = self.forward(
            torch.zeros(1, 200, device=self.device), image, control_period=16
        )
        traj = self.dynamics_model.rotate_z0(
            z[:, -1:].detach() * self.dynamics_model.input_scalar
        )
        self.pred = self.dynamics_model.pred
        return traj

    def evaluate_q(
        self,
        q_in,
        returned_list=["z0"],
        control_period=16,
        control_threshold=None,
        random_init=False,
    ):
        """This method provides an easy handle for the globe plots

        Args:
            q_in (torch.Tensor): the input quaternions (..., 4)
            returned_list (list[str]): list of names of objects to return
                choices:
                    "z0"
                    "error_traj"
                    "I_pred"
                    "I_pred_error"
                    "I_out"
                    "I_out_error"
                    "q_out"
                    "q_out_error"
                    "control_norm"
                    "state_norm"
                    "state_log_likelihood"
            control_period (int): the number of steps to control the system for
            control_threshold (float): the threshold to stop the control at (not implemented)

        Returns:
            returned (dict[torch.Tensor]): named with keys provided in the returned_list

        """
        assert q_in.shape[-1] == 4
        q_in = q_in.to(self.device)
        if type(random_init) == bool:
            init_state = (
                torch.randn(*q_in.shape[:-1], 200, device=self.device)
                if random_init
                else torch.zeros(*q_in.shape[:-1], 200, device=self.device)
            )
        else:
            assert type(random_init) == torch.Tensor
            init_state = random_init.to(self.device)
        if returned_list == ["all"]:
            returned_list = [
                "z0",
                "I_pred",
                "I_pred_error",
                "I_out",
                "I_out_error",
                "q_out",
                "q_out_error",
                "control_norm",
                "state_norm",
                "state_log_likelihood",
                "pred_error",
            ]
        returned = {}
        image = self.mesh.quat2verts(q_in / q_in.norm(dim=-1, keepdim=True)).view(
            -1, 24 if self.mode != "camera_simple" else 16
        )

        if "control_norm" not in returned_list and "control_traj" not in returned_list:
            z, error_traj = self.forward(
                init_state, image, control_period=control_period
            )
        else:
            z, control_traj, error_traj = self.forward(
                init_state,
                image,
                control_period=control_period,
                return_control_trajectory=True,
            )
            if "control_norm" in returned_list:
                returned["control_norm"] = control_traj.norm(p=2, dim=-1).sum(dim=-1)
            if "control_traj" in returned_list:
                returned["control_traj"] = control_traj.norm(p=2, dim=-1)

        if "error_traj" in returned_list:
            returned["error_traj"] = error_traj

        if "z0" in returned_list:
            returned["z0"] = z[:, -1, :].detach()
        if "I_pred_error" in returned_list:
            returned["I_pred_error"] = (
                image
                - self.predictive_model(z[:, -1, :].detach())[..., -1, :].view(
                    -1, 24 if self.mode != "camera_simple" else 16
                )
            ).norm(p=2, dim=-1)
        if "I_pred" in returned_list:
            returned["I_pred"] = self.predictive_model(z[:, -1, :].detach())

        out = None
        if any(
            [
                x in returned_list
                for x in ["I_out", "I_out_error", "q_out", "q_out_error"]
            ]
        ):
            out = self.dynamics_model(
                z[:, -1, :].detach() * self.dynamics_model.input_scalar
            )
            if "I_out" in returned_list:
                returned["I_out"] = self.mesh.quat2verts(
                    self.dynamics_model.pred[..., -1, :].detach()
                ).view(-1, 24 if self.mode != "camera_simple" else 16)
            if "I_out_error" in returned_list:
                returned["I_out_error"] = (
                    image
                    - self.mesh.quat2verts(
                        self.dynamics_model.pred[..., -1, :].detach()
                    ).view(-1, 24 if self.mode != "camera_simple" else 16)
                ).norm(p=2, dim=-1)
            if "q_out" in returned_list:
                returned["q_out"] = self.dynamics_model.pred[..., -1, :].detach()
            if "q_out_error" in returned_list:
                # return geodesic distance between the q's
                returned["q_out_error"] = 2 * torch.acos(
                    torch.clamp(
                        torch.abs(
                            (q_in * self.dynamics_model.pred[..., -1, :].detach()).sum(
                                dim=-1
                            )
                        ),
                        -1,
                        1,
                    )
                )
        if "pred_error" in returned_list:
            if not hasattr(returned, "I_out"):
                if out is None:
                    out = self.dynamics_model(
                        z[:, -1, :].detach() * self.dynamics_model.input_scalar
                    )
                iout = self.mesh.quat2verts(
                    self.dynamics_model.pred[..., -1, :].detach()
                ).view(-1, 24 if self.mode != "camera_simple" else 16)
            else:
                iout = returned["I_out"]

            if not hasattr(returned, "I_pred"):
                ipred = self.predictive_model(z[:, -1, :].detach())[..., -1, :].view(
                    -1, 24 if self.mode != "camera_simple" else 16
                )
            else:
                ipred = returned["I_pred"]
            print(f"iout shape: {iout.shape}; ipred shape: {ipred.shape}")
            returned["pred_error"] = (ipred - iout).norm(p=2, dim=-1)

        if "state_norm" in returned_list:
            returned["state_norm"] = z[:, -1, :].norm(p=2, dim=-1)

        if "state_log_likelihood" in returned_list:
            if not hasattr(self, "U2_inv") or self.U2_inv is None:
                U = torch.tensor(self.dynamics_model.U, device=self.device)
                self.U2_inv = (U @ U.T).inverse()
            returned["state_log_likelihood"] = (
                z[:, -1, :] @ self.U2_inv @ (z[:, -1, :].T)
            ).mean()
        return returned

    def load_state_dict(self, state_dict, strict=True, assign=False):
        temp_dm = deepcopy(self.dynamics_model)
        self.dynamics_model = None
        super().load_state_dict(state_dict, strict, assign)
        self.dynamics_model = temp_dm


class ControlInjectNet(nn.Module):  # (torch.jit.ScriptModule):
    """This module is an RNN controller that takes concatenated reference signal (picture space) and predicted state output (picture space) as inputs and holds the current state as hidden.

    - NOTE: the thing operates directly in state space and hence disregards the original system dynamics.

    """

    def __init__(
        self,
        predictive_model,
        dt=1,
        tau=100,
        output_size=16,
        target_size=200,
        dT=500 / 32,  # time step for the dynamics model;
        W=None,
        pure_error=True,  # if we want only allow error feedback
        hidden_control_dim=0,  # if this is more than 0, we would require using a readout matrix.
        activation="tanh",
        network_type="default",
        device="cuda",
    ):
        super(ControlInjectNet, self).__init__()

        self.predictive_model = (
            torch.jit.script(predictive_model)
            if isinstance(self, torch.jit.ScriptModule)
            else predictive_model
        )

        for param in self.predictive_model.parameters():
            param.requires_grad = False

        self.dt = dt
        self.dT = dT
        self.tau = tau
        self.W = W
        self.eW = torch.tensor(
            expm((W - torch.eye(W.shape[0])).cpu().numpy() * self.dT / self.tau),
            device=device,
        )

        self.output_size = output_size
        self.feedback_size = (
            output_size if pure_error else 2 * output_size
        )  # NOTE This may be 16 or 24 - depending on the configuration of the camera (we could either have 2d image or an extra dimension)
        self.hidden_size = (
            hidden_control_dim if hidden_control_dim != 0 else target_size
        )

        # collate the knobs that we specify
        self.inject_inputs = True
        self.recurrent_control = True
        self.network_type = network_type

        if network_type == "default":
            # TODO: see if I can do this by the simplest setup (no GRU or LSTM)
            self.h = None
            self.W_hh = nn.Linear(self.hidden_size, self.hidden_size)
            self.W_xh = nn.Linear(self.feedback_size, self.hidden_size)

        elif network_type.lower() == "gru":
            self.z = torch.zeros(self.hidden_size)
            self.r = torch.zeros(self.hidden_size)
            self.n = torch.zeros(self.hidden_size)
            self.h = torch.zeros(self.hidden_size)
            self.W_rx = nn.Linear(self.feedback_size, self.hidden_size, bias=False)
            self.W_rh = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.b_r = nn.Parameter(torch.zeros(self.hidden_size))
            self.W_zx = nn.Linear(self.feedback_size, self.hidden_size, bias=False)
            self.W_zh = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.b_z = nn.Parameter(torch.zeros(self.hidden_size))
            self.W_nx = nn.Linear(self.feedback_size, self.hidden_size, bias=False)
            self.W_nh = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.b_n = nn.Parameter(torch.zeros(self.hidden_size))

        self.construct_graph()

        self.activation = (
            nn.Tanh()
            if activation == "tanh"
            else (nn.ReLU() if activation == "relu" else nn.Identity())
        )

        self.output = nn.Sequential(nn.Linear(self.hidden_size, target_size))

        self.device = device
        for i in self.parameters():
            i = i.to(device)

        self.mesh = None
        self.mode = None
        self.dynamics_model = None

    # @torch.jit.script_method
    def forward(
        self,
        x0,
        target,
        h0=None,
        control_period: Optional[int] = None,
        control_threshold: Optional[float] = None,
        period_upperlimit: int = 100,
        return_control_trajectory=False,
    ):
        """Forward control loop, defined either to run a number of steps or reach a certain threshold.

        Here, we could train the system to perform as little steps as possible or constrain the total step number.
        """
        assert control_period is not None or control_threshold is not None
        assert control_period is None or control_threshold is None

        chunk_length = 4096  # to avoid memory overflow, we divide inputs into tiny segments when it's too big
        error_trajectory = None
        x_segment = []
        target_segment = []
        if x0.shape[0] > chunk_length:
            left = x0.shape[0]
            segment_start = 0
            while left > 0:
                x_segment.append(
                    x0[segment_start : (segment_start + min(chunk_length, left))]
                )
                target_segment.append(
                    target[segment_start : (segment_start + min(chunk_length, left))]
                )
                left -= chunk_length
                segment_start += chunk_length
            del x0, target
            torch.cuda.empty_cache()

            for i in range(len(x_segment)):
                x_seg = x_segment.pop(0)
                target_seg = target_segment.pop(0)
                if return_control_trajectory:
                    x_, control_trajectory_, error_trajectory_ = self.forward(
                        x_seg,
                        target_seg,
                        h0,
                        control_period,
                        control_threshold,
                        period_upperlimit,
                        return_control_trajectory,
                    )
                    x, control_trajectory, error_trajectory = (
                        (x_, control_trajectory_, error_trajectory_)
                        if error_trajectory is None
                        else (
                            torch.cat([x, x_], dim=0),
                            torch.cat([control_trajectory, control_trajectory_], dim=0),
                            torch.cat([error_trajectory, error_trajectory_], dim=0),
                        )
                    )
                    print(x.shape)
                else:
                    x_, error_trajectory_ = self.forward(
                        x_seg,
                        target_seg,
                        h0,
                        control_period,
                        control_threshold,
                        period_upperlimit,
                        return_control_trajectory,
                    )
                    x, error_trajectory = (
                        (x_, error_trajectory_)
                        if error_trajectory is None
                        else (
                            torch.cat([x, x_], dim=0),
                            torch.cat([error_trajectory, error_trajectory_], dim=0),
                        )
                    )
                    print(x.shape)

            if return_control_trajectory:
                return x, control_trajectory, error_trajectory
            return x, error_trajectory

        original_device = self.device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        x0 = x0.to(device)
        target = target.to(device)
        h0 = torch.zeros(x0.shape[0], self.hidden_size, device=self.device)

        x = x0.unsqueeze(1)
        time = 0
        error_trajectory = torch.zeros(x.shape[0], 0, device=self.device)
        control_trajectory = torch.zeros(x.shape[0], 0, x.shape[-1], device=self.device)
        ht = h0
        while True:
            if isinstance(control_period, int) and time == control_period:
                break
            if (
                isinstance(control_threshold, float)
                and (x[:, -1, :] - target).norm(p=2).mean() < control_threshold
            ):
                break

            control_input, ht, error = self.step_controller(x[:, -1, :], ht, target)
            error_trajectory = torch.cat([error_trajectory, error.unsqueeze(1)], dim=1)
            if time > period_upperlimit:
                print(f"Control period exceeded upper bound with residual error")
                break
            x_new = (self.eW @ x[:, -1, :].T).T.unsqueeze(1) + control_input
            control_trajectory = torch.cat([control_trajectory, control_input], dim=1)

            x = torch.cat([x, x_new], dim=1)
            time += 1
        # error_trajectory = (x[:, 1:, :] - target.unsqueeze(1)).norm(p = 2, dim = -1)
        self.to(original_device)
        if return_control_trajectory:
            return x, control_trajectory, error_trajectory
        return x, error_trajectory

    # @torch.jit.script_method
    def step_controller(self, zt, ht, target=None):
        """Gives one step update of the controller"""
        if target is None:
            target = self.target

        predicted = self.predictive_model(zt)
        predicted = predicted[..., -1, :].view(-1, self.output_size)

        self.h = ht

        if self.feedback_size == target.shape[-1]:
            et = target - predicted
        else:
            et = torch.cat([target, predicted], dim=-1)

        if hasattr(self, "graph"):
            ht = self.graph(et)
        else:
            x_xh = self.W_xh(et)
            x_hh = self.W_hh(self.h)
            ht = self.activation(x_hh + x_xh)
        self.h = ht
        yt = self.output(ht).unsqueeze(1)
        return yt, ht, (target - predicted).norm(p=2, dim=-1) ** 2

    def construct_graph(self):
        if self.network_type == "default":
            self.graph = self.graph_recurrent
        elif self.network_type.lower() == "gru":
            self.graph = self.graph_gru
        else:
            raise ValueError("Unknown network type")

    def graph_recurrent(self, et):
        # Uses self.h as the current state step
        x_xh = self.W_xh(
            et
        )  # et should be precomputed (decided if to concatenate or take differnece)
        x_hh = self.W_hh(self.h)
        h = self.activation(x_hh + x_xh)
        return h

    def graph_gru(self, et):
        self.r = torch.sigmoid(self.W_rh(self.h) + self.W_rx(et) + self.b_r)
        self.z = torch.sigmoid(self.W_zh(self.h) + self.W_zx(et) + self.b_z)
        self.n = self.activation(self.W_nh(self.r * self.h) + self.W_nx(et) + self.b_n)
        h = (1 - self.z) * self.n + self.z * self.h
        return h

    def rotate(self, q_in):
        """This method prepares a preparatory state and sets the dynamical network to unroll"""
        assert q_in.view(-1, 4).shape[0] == 1
        image = self.mesh.quat2verts(q_in / q_in.norm(dim=-1, keepdim=True)).view(
            -1, 24 if self.mode != "camera_simple" else 16
        )
        z, _ = self.forward(
            torch.zeros(1, 200, device=self.device), image, control_period=16
        )
        traj = self.dynamics_model.rotate_z0(
            z[:, -1:].detach() * self.dynamics_model.input_scalar
        )
        self.pred = self.dynamics_model.pred
        return traj

    def evaluate_q(
        self,
        q_in,
        returned_list=["z0"],
        control_period=16,
        control_threshold=None,
        random_init=False,
    ):
        """This method provides an easy handle for the globe plots

        Args:
            q_in (torch.Tensor): the input quaternions (..., 4)
            returned_list (list[str]): list of names of objects to return
                choices:
                    "z0"
                    "error_traj"
                    "I_pred"
                    "I_pred_error"
                    "I_out"
                    "I_out_error"
                    "q_out"
                    "q_out_error"
                    "control_norm"
                    "state_norm"
                    "state_log_likelihood"
            control_period (int): the number of steps to control the system for
            control_threshold (float): the threshold to stop the control at (not implemented)

        Returns:
            returned (dict[torch.Tensor]): named with keys provided in the returned_list

        """
        assert q_in.shape[-1] == 4
        q_in = q_in.to(self.device)
        if type(random_init) == bool:
            init_state = (
                torch.randn(*q_in.shape[:-1], 200, device=self.device)
                if random_init
                else torch.zeros(*q_in.shape[:-1], 200, device=self.device)
            )
        else:
            assert type(random_init) == torch.Tensor
            init_state = random_init.to(self.device)
        if returned_list == ["all"]:
            returned_list = [
                "z0",
                "I_pred",
                "I_pred_error",
                "I_out",
                "I_out_error",
                "q_out",
                "q_out_error",
                "control_norm",
                "state_norm",
                "state_log_likelihood",
                "pred_error",
            ]
        returned = {}
        image = self.mesh.quat2verts(q_in / q_in.norm(dim=-1, keepdim=True)).view(
            -1, 24 if self.mode != "camera_simple" else 16
        )

        if "control_norm" not in returned_list and "control_traj" not in returned_list:
            z, error_traj = self.forward(
                init_state, image, control_period=control_period
            )
        else:
            z, control_traj, error_traj = self.forward(
                init_state,
                image,
                control_period=control_period,
                return_control_trajectory=True,
            )
            if "control_norm" in returned_list:
                returned["control_norm"] = control_traj.norm(p=2, dim=-1).sum(dim=-1)
            if "control_traj" in returned_list:
                returned["control_traj"] = control_traj.norm(p=2, dim=-1)

        if "error_traj" in returned_list:
            returned["error_traj"] = error_traj

        if "z0" in returned_list:
            returned["z0"] = z[:, -1, :].detach()
        if "I_pred_error" in returned_list:
            returned["I_pred_error"] = (
                image
                - self.predictive_model(z[:, -1, :].detach())[..., -1, :].view(
                    -1, 24 if self.mode != "camera_simple" else 16
                )
            ).norm(p=2, dim=-1)
        if "I_pred" in returned_list:
            returned["I_pred"] = self.predictive_model(z[:, -1, :].detach())

        self.to("cuda")
        out = None
        if any(
            [
                x in returned_list
                for x in ["I_out", "I_out_error", "q_out", "q_out_error"]
            ]
        ):
            out = self.dynamics_model(
                z[:, -1, :].detach() * self.dynamics_model.input_scalar
            )
            if "I_out" in returned_list:
                returned["I_out"] = self.mesh.quat2verts(
                    self.dynamics_model.pred[..., -1, :].detach()
                ).view(-1, 24 if self.mode != "camera_simple" else 16)
            if "I_out_error" in returned_list:
                returned["I_out_error"] = (
                    image
                    - self.mesh.quat2verts(
                        self.dynamics_model.pred[..., -1, :].detach()
                    ).view(-1, 24 if self.mode != "camera_simple" else 16)
                ).norm(p=2, dim=-1)
            if "q_out" in returned_list:
                returned["q_out"] = self.dynamics_model.pred[..., -1, :].detach()
            if "q_out_error" in returned_list:
                # return geodesic distance between the q's
                returned["q_out_error"] = 2 * torch.acos(
                    torch.clamp(
                        torch.abs(
                            (q_in * self.dynamics_model.pred[..., -1, :].detach()).sum(
                                dim=-1
                            )
                        ),
                        -1,
                        1,
                    )
                )
        if "pred_error" in returned_list:
            if not hasattr(returned, "I_out"):
                if out is None:
                    out = self.dynamics_model(
                        z[:, -1, :].detach() * self.dynamics_model.input_scalar
                    )
                iout = self.mesh.quat2verts(
                    self.dynamics_model.pred[..., -1, :].detach()
                ).view(-1, 24 if self.mode != "camera_simple" else 16)
            else:
                iout = returned["I_out"]

            if not hasattr(returned, "I_pred"):
                ipred = self.predictive_model(z[:, -1, :].detach())[..., -1, :].view(
                    -1, 24 if self.mode != "camera_simple" else 16
                )
            else:
                ipred = returned["I_pred"]
            print(f"iout shape: {iout.shape}; ipred shape: {ipred.shape}")
            returned["pred_error"] = (ipred - iout).norm(p=2, dim=-1)

        if "state_norm" in returned_list:
            returned["state_norm"] = z[:, -1, :].norm(p=2, dim=-1)

        if "state_log_likelihood" in returned_list:
            if not hasattr(self, "U2_inv") or self.U2_inv is None:
                U = torch.tensor(self.dynamics_model.U, device=self.device)
                self.U2_inv = (U @ U.T).inverse()
            returned["state_log_likelihood"] = (
                z[:, -1, :] @ self.U2_inv @ (z[:, -1, :].T)
            ).mean()
        return returned

    def load_state_dict(self, state_dict, strict=True, assign=False):
        temp_dm = deepcopy(self.dynamics_model)
        self.dynamics_model = None
        super().load_state_dict(state_dict, strict, assign)
        self.dynamics_model = temp_dm

    def to(self, device):
        super().to(device)
        self.device = device
        self.mesh.device = device
        if hasattr(self.mesh, "to"):
            self.mesh.to(device)
            self.mesh.device = device
        if hasattr(self.predictive_model, "to"):
            self.predictive_model.to(device)
            self.predictive_model.device = device
        if hasattr(self.dynamics_model, "to"):
            self.dynamics_model.to(device)
            self.dynamics_model.device = device
        self.W = self.W.to(device)
        self.eW = self.eW.to(device)

        return self


# now the control net is purposed to take on the (more straightforward) task to control a linked CAN to a certain statea


class ControlNetCAN(nn.Module):  # (torch.jit.ScriptModule):
    # NOTE: currently discrete time: this means that the control has the same time step as the system.
    """
    Implements a neural network controller to control a representation net to some desired state.
    - note: the reference signal may be in a subspace of the state space.
    - implement simple state loss (activation MSE) + control loss (control norm, control net hidden state norm)
    - can then learn some forward model (control net hidden state directly to control the representation net, when the actual velocities require integration &c. )

    this is one sheet of neurons, followed by readout connections to 2N output nodes where N is the dimensionality of the tangent space
    """

    def __init__(
        self,
        rep_net_path,
        rep_units_only=False,  # read from representation units only
        recurrent_control=False,  # use simple recurrence here (RNN)
        control_mode="velocity",  # "velocity" or "acceleration"
        activation="relu",
        hidden_control_dim=0,
        device="cuda",
    ):
        super(ControlNetCAN, self).__init__()
        self.rep_net = torch.load(rep_net_path)  # instance of a PartitionedNet
        for param in self.rep_net.parameters():
            param.requires_grad = False

        self.rep_units_only = rep_units_only
        if self.rep_units_only:
            self.ref_size = self.rep_net.rnn.rep_units
        else:
            self.ref_size = self.rep_net.rnn.hidden_size
        self.state_size = (
            self.rep_net.rnn.hidden_size
        )  # note the possibility of the hidden size being different from the state size
        self.output_size = self.rep_net.rnn.input_size

        self.recurrent_control = recurrent_control
        self.control_mode = control_mode
        self.hidden_control_dim = hidden_control_dim
        self.device = device
        self.activation = activation
        try:
            self.activation = getattr(nn, self.activation)()
        except:
            try:
                self.activation = getattr(F, self.activation)()
            except:
                self.activation = globals().get(self.activation)()
        self.hidden_size = (
            hidden_control_dim
            if hidden_control_dim != 0
            else self.rep_net.rnn.input_size
        )
        self.output_p = nn.Sequential(
            nn.Linear(self.hidden_control_dim, self.output_size, bias=False),
            self.activation,
        )
        self.output_m = nn.Sequential(
            nn.Linear(self.hidden_control_dim, self.output_size, bias=False),
            self.activation,
        )

        self.construct_graph()
        self.W_hh = (
            nn.Linear(self.hidden_control_dim, self.hidden_control_dim)
            if self.recurrent_control
            else None
        )

        self.W_xh = nn.Linear(self.state_size + self.ref_size, self.hidden_control_dim)
        # nn.init.xavier_normal_(self.W_xh.weight)
        # self.W_xh = nn.Sequential(
        #     nn.Linear(self.state_size + self.ref_size, self.hidden_control_dim * 3),
        #     self.activation,
        #     nn.Linear(self.hidden_control_dim * 2, self.hidden_control_dim),
        # )

        self.h = torch.zeros(1, self.hidden_control_dim, device=self.device)

    def construct_graph(self):
        if self.recurrent_control:
            self.graph = self.graph_recurrent
        else:
            self.graph = self.graph_feedforward

    def graph_recurrent(self, et):
        # et is the external input
        x_xh = self.W_xh(et)
        x_hh = self.W_hh(self.h)
        self.h = self.activation(x_hh + x_xh)
        out = self.output_p(self.h) - self.output_m(self.h)
        return self.h, out

    def graph_feedforward(self, et):
        h = self.activation(self.W_xh(et))
        output_p = self.output_p(h)
        output_m = self.output_m(h)
        out = output_p - output_m
        return h, out

    def forward(
        self,
        h0,
        ref,
        control_period: Optional[int] = None,
        control_threshold: Optional[float] = None,
        period_upperlimit: int = 100,
        return_control_trajectory=False,
    ):
        """Forward loop. Controlled for a number of steps or until a certain threshold is reached."""
        assert control_period is not None or control_threshold is not None
        assert control_period is None or control_threshold is None

        chunk_length = 4096  # to avoid memory overflow, we divide inputs into tiny segments when it's too big
        error_trajectory = None
        x_segment = []
        target_segment = []
        time = 0
        if h0.shape[0] > chunk_length:
            left = h0.shape[0]
            segment_start = 0
            while left > 0:
                x_segment.append(
                    h0[segment_start : (segment_start + min(chunk_length, left))]
                )
                target_segment.append(
                    ref[segment_start : (segment_start + min(chunk_length, left))]
                )
                left -= chunk_length
                segment_start += chunk_length
            del h0, ref
            torch.cuda.empty_cache()

            for i in range(len(x_segment)):
                x_seg = x_segment.pop(0)
                target_seg = target_segment.pop(0)
                if return_control_trajectory:
                    x_, control_trajectory_, error_trajectory_ = self.forward(
                        x_seg,
                        target_seg,
                        h0,
                        control_period,
                        control_threshold,
                        period_upperlimit,
                        return_control_trajectory,
                    )
                    x, control_trajectory, error_trajectory = (
                        (x_, control_trajectory_, error_trajectory_)
                        if error_trajectory is None
                        else (
                            torch.cat([x, x_], dim=0),
                            torch.cat([control_trajectory, control_trajectory_], dim=0),
                            torch.cat([error_trajectory, error_trajectory_], dim=0),
                        )
                    )
                    print(x.shape)
                else:
                    x_, error_trajectory_ = self.forward(
                        x_seg,
                        target_seg,
                        h0,
                        control_period,
                        control_threshold,
                        period_upperlimit,
                        return_control_trajectory,
                    )
                    x, error_trajectory = (
                        (x_, error_trajectory_)
                        if error_trajectory is None
                        else (
                            torch.cat([x, x_], dim=0),
                            torch.cat([error_trajectory, error_trajectory_], dim=0),
                        )
                    )
                    print(x.shape)

            if return_control_trajectory:
                return x, control_trajectory, error_trajectory, ht_trajectory
            return x, error_trajectory

        original_device = self.device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        hi = h0.to(device)  # Note, this is the hidden state of the rep net!
        ref = ref.to(device)
        error_trajectory = torch.zeros(hi.shape[0], 0, device=device)
        control_trajectory = torch.zeros(
            hi.shape[0], 0, self.output_size, device=device
        )
        ht_trajectory = torch.zeros(
            hi.shape[0], 0, self.hidden_control_dim, device=device
        )
        hi_trajectory = torch.zeros(hi.shape[0], 0, self.state_size, device=device)
        while True:
            if isinstance(control_period, int) and time == control_period:
                break
            if (
                isinstance(control_threshold, float)
                and (hi - ref).norm(p=2).mean() < control_threshold
            ):
                break
            hi, out, ht, error = self.step_controller(hi, ref)
            error_trajectory = torch.cat([error_trajectory, error.unsqueeze(1)], dim=1)
            if time > period_upperlimit:
                print(f"Control period exceeded upper bound with residual error")
                break
            control_trajectory = torch.cat(
                [control_trajectory, out.unsqueeze(1)], dim=1
            )
            ht_trajectory = torch.cat([ht_trajectory, ht.unsqueeze(1)], dim=1)
            hi_trajectory = torch.cat([hi_trajectory, hi.unsqueeze(1)], dim=1)
            time += 1
        self.to(original_device)
        if return_control_trajectory:
            return hi_trajectory, control_trajectory, error_trajectory, ht_trajectory
        return hi_trajectory, error_trajectory

    def step_controller(self, hi, ref):
        """
        hi: shape(batch_size, input_units * input_size + rep_units)
        ref: shape(batch_size, ref_size)

        Returns:
        hi: shape(batch_size, rep_units), current state after update
        out: shape(batch_size, output_size), output of the controller
        ht: shape(batch_size, hidden_control_dim), hidden state of the controller
        """
        # print((ref - hi[:, -ref.shape[-1]:]).norm(p = 2, dim = -1))
        et = torch.cat([hi, ref], dim=-1).float()
        # print(f"et shape: {et.shape}")
        ht, out = self.graph(et)
        out *= 0.25
        # out *= 0
        # print(out)
        hi = self.step_CAN(hi, out)
        # print(f"ref shape: {ref.shape}")
        # print(f"hi shape: {hi.shape}")
        error = (ref - hi[:, -ref.shape[-1] :]).norm(p=2, dim=-1) ** 2
        # print(error)
        return hi, out, ht, error

    def step_CAN(self, hi, velocity_input):
        """
        hi_input: shape(batch_size, input_units * input_size)
        hi_rep: shape(batch_size, rep_units)
        velocity_input: shape(batch_size, input_size)

        Returns:
        hi: shape(batch_size, rep_units), current state after update
        """
        hi = self.rep_net.rnn.step(hi, velocity_input)
        return hi


class PredControlNetCAN(nn.Module):  # (torch.jit.ScriptModule):
    def __init__(
        self,
        rep_net,  # these all should be nn.Modules with forward methods
        plant,
        operator,  # path-integrates in the environment
        IM1,  # internal model 1
        IM2,  # internal model 2
        readoutnet,
        output_size=None,
        rep_units_only=False,  # read from representation units only
        recurrent_control=False,  # use simple recurrence here (RNN)
        activation="relu",
        recurrent_control_dim=0,
        hidden_control_dim=0,
        device="cuda",
    ):
        super(PredControlNetCAN, self).__init__()
        self.rep_net = rep_net
        for param in self.rep_net.parameters():
            param.requires_grad = False

        self.rep_units_only = rep_units_only
        if self.rep_units_only:
            self.ref_size = self.rep_net.rnn.rep_units
        else:
            self.ref_size = self.rep_net.rnn.hidden_size
        self.state_size = (
            self.rep_net.rnn.hidden_size
        )  # note the possibility of the hidden size being different from the state size
        # self.plant = self._init_plant(plant_specs)
        self.plant = plant.to(device)
        self.readoutnet = readoutnet.to(device)

        if output_size is None:
            self.output_size = self.plant.state_dim
        else:
            self.output_size = output_size

        self.action_size = self.rep_net.rnn.input_size

        # self.operator = self._init_operator(operator_specs) # must match the operator's input dimension to that of the rep net
        # self.im1, self.hidden_pred_dim1 = self._init_IM(IM1_specs)
        # self.im2, self.hidden_pred_dim2 = self._init_IM(IM2_specs)
        self.operator = operator.to(device)
        self.im1, self.hidden_im1_dim = IM1.to(device), IM1.recurrent_size
        self.im2, self.hidden_im2_dim = IM2.to(device), IM2.recurrent_size

        self.recurrent_control = recurrent_control
        self.recurrent_control_dim = recurrent_control_dim

        self.device = device
        self.activation = activation
        try:
            self.activation = getattr(nn, self.activation)()
        except:
            try:
                self.activation = getattr(F, self.activation)()
            except:
                self.activation = globals().get(self.activation)()
        self.hidden_size = (
            recurrent_control_dim
            if recurrent_control_dim != 0
            else self.rep_net.rnn.input_size
        )

        # TODO: fix these
        # self.output_p = nn.Sequential(
        #     nn.Linear(self.recurrent_control_dim, self.output_size, bias=False),
        #     self.activation)
        # self.output_m = nn.Sequential(
        #     nn.Linear(self.recurrent_control_dim, self.output_size, bias = False),
        #     self.activation)

        self.W_hh = (
            nn.Linear(self.recurrent_control_dim, self.recurrent_control_dim)
            if self.recurrent_control
            else None
        )
        self.W_xh = nn.Linear(self.state_size + self.ref_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.construct_graph()
        self.ht0 = nn.Parameter(
            torch.zeros(1, self.recurrent_control_dim, device=self.device)
        )
        self.ht_im10 = nn.Parameter(
            torch.zeros(1, self.hidden_im1_dim, device=self.device)
        )
        self.ht_im20 = nn.Parameter(
            torch.zeros(1, self.hidden_im2_dim, device=self.device)
        )
        self.h_plant0 = nn.Parameter(
            torch.zeros(1, self.plant.state_dim, device=self.device),
            requires_grad=False,
        )  # control from nullity

        assert self.plant.dt == self.operator.dt
        self.dt = self.plant.dt

    def freeze_controller(self):
        for module in [self.W_xh, self.W_hh, self.out]:
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = False
        
    def freeze_im1(self): 
        for param in self.im1.parameters():
            param.requires_grad = False
    
    def freeze_im2(self):
        for param in self.im2.parameters():
            param.requires_grad = False
    
    def freeze_plant(self):
        for param in self.plant.parameters():
            param.requires_grad = False
    
    def freeze_operator(self):
        for param in self.operator.parameters():
            param.requires_grad = False

    def freeze_rep_net(self):        
        for param in self.rep_net.parameters():
            param.requires_grad = False
    
    def freeze_readoutnet(self):
        for param in self.readoutnet.parameters():
            param.requires_grad = False
    
    def unfreeze_readoutnet(self):
        for param in self.readoutnet.parameters():
            param.requires_grad = True
            
    def unfreeze_rep_net(self):
        for param in self.rep_net.parameters():
            param.requires_grad = True

    def unfreeze_operator(self):
        for param in self.operator.parameters():
            param.requires_grad = True
    
    def unfreeze_plant(self):
        for param in self.plant.parameters():
            param.requires_grad = True
        
    def unfreeze_im1(self):
        for param in self.im1.parameters():
            param.requires_grad = True

    def unfreeze_im2(self):
        for param in self.im2.parameters():
            param.requires_grad = True

    def unfreeze_controller(self):
        for module in [self.W_xh, self.W_hh, self.out]:
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = True
            
    
    def float(self):
        self.to("cuda")
        self.rep_net = self.rep_net.float()
        self.plant = self.plant.float()
        self.operator = self.operator.float()
        self.im1 = self.im1.float()
        self.im2 = self.im2.float()
        self.readoutnet = self.readoutnet.float()
        self.W_hh = self.W_hh.float() if self.W_hh is not None else None
        self.W_xh = self.W_xh.float()
        self.out = self.out.float()
        self.h.data = self.h.data.float()
        self.ht0.data = self.ht0.data.float()
        self.ht_im10.data = self.ht_im10.data.float()
        self.ht_im20.data = self.ht_im20.data.float()
        self.h_plant0.data = self.h_plant0.data.float()
        return self

    def construct_graph(self):
        if self.recurrent_control:
            self.graph = self.graph_recurrent
        else:
            self.graph = self.graph_feedforward

    def graph_recurrent(self, et, ht):
        # et is the external input
        x_xh = self.W_xh(et)
        x_hh = self.W_hh(ht)
        h = self.activation(x_hh + x_xh)
        # out = self.output_p(self.h) - self.output_m(self.h)
        out = self.out(h)
        return h, out

    def graph_feedforward(self, et, ht):
        h = self.activation(self.W_xh(et))
        out = self.out(h)
        return ht, out

    def forward(
        self,
        h0,
        ref,
        control_period: Optional[int] = None,
        control_threshold: Optional[float] = None,
        period_upperlimit: int = 100,
        return_control_trajectory=False,
    ):
        """Forward loop. Controlled for a number of steps or until a certain threshold is reached."""
        assert control_period is not None or control_threshold is not None
        assert control_period is None or control_threshold is None
        chunk_length = 4096  # to avoid memory overflow, we divide inputs into tiny segments when it's too big
        error_trajectory = None
        x_segment = []
        target_segment = []
        time = 0
        if h0.shape[0] > chunk_length:
            left = h0.shape[0]
            segment_start = 0
            while left > 0:
                x_segment.append(
                    h0[segment_start : (segment_start + min(chunk_length, left))]
                )
                target_segment.append(
                    ref[segment_start : (segment_start + min(chunk_length, left))]
                )
                left -= chunk_length
                segment_start += chunk_length
            del h0, ref
            torch.cuda.empty_cache()

            for i in range(len(x_segment)):
                x_seg = x_segment.pop(0)
                target_seg = target_segment.pop(0)
                if return_control_trajectory:
                    x_, control_trajectory_, error_trajectory_ = self.forward(
                        x_seg,
                        target_seg,
                        h0,
                        control_period,
                        control_threshold,
                        period_upperlimit,
                        return_control_trajectory,
                    )
                    x, control_trajectory, error_trajectory = (
                        (x_, control_trajectory_, error_trajectory_)
                        if error_trajectory is None
                        else (
                            torch.cat([x, x_], dim=0),
                            torch.cat([control_trajectory, control_trajectory_], dim=0),
                            torch.cat([error_trajectory, error_trajectory_], dim=0),
                        )
                    )
                    print(x.shape)
                else:
                    x_, error_trajectory_ = self.forward(
                        x_seg,
                        target_seg,
                        h0,
                        control_period,
                        control_threshold,
                        period_upperlimit,
                        return_control_trajectory,
                    )
                    x, error_trajectory = (
                        (x_, error_trajectory_)
                        if error_trajectory is None
                        else (
                            torch.cat([x, x_], dim=0),
                            torch.cat([error_trajectory, error_trajectory_], dim=0),
                        )
                    )
                    print(x.shape)

            if return_control_trajectory:
                return x, control_trajectory, error_trajectory, ht_trajectory
            return x, error_trajectory

        original_device = self.device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        hi = h0.to(device)  # Note, this is the hidden state of the rep net!
        ht = self.ht0.to(device)
        ht_im1 = self.ht_im10.to(device)
        ht_im2 = self.ht_im20.to(device)
        h_plant = self.h_plant0.to(device)

        pos0 = self.readoutnet(h0[..., -self.rep_net.rnn.rep_units :])
        pos0 = pos0 / pos0.norm(p=2, dim=-1, keepdim=True)
        operator_s = pos0
        ref = ref.to(device)  # let's say we have the same ref throughout?

        if control_period is None:
            control_period = period_upperlimit

        error_trajectory = torch.zeros(hi.shape[0], control_period, device=device)
        action_recovery_error_trajectory = torch.zeros(
            hi.shape[0], control_period, device=device
        )
        action_correction_trajectory = torch.zeros(
            hi.shape[0], control_period, device=device
        )
        control_trajectory = torch.zeros(
            hi.shape[0], control_period, self.output_size, device=device
        )
        out_trajectory = torch.zeros(
            hi.shape[0], control_period, self.action_size, device=device
        )
        ht_trajectory = torch.zeros(
            hi.shape[0], control_period, self.recurrent_control_dim, device=device
        )
        hi_trajectory = torch.zeros(
            hi.shape[0], control_period, self.state_size, device=device
        )
        ht_im1_trajectory = torch.zeros(
            hi.shape[0], control_period, self.hidden_im1_dim, device=device
        )
        ht_im2_trajectory = torch.zeros(
            hi.shape[0], control_period, self.hidden_im2_dim, device=device
        )
        action_trajectory = torch.zeros(
            hi.shape[0], control_period, self.action_size, device=device
        )
        # NOTE: out_trajectory is the true series of actions to the plant; action_trajectory is the inferred action from the IMs
        hi_trajectory[:, -1, :] = hi

        for time in range(control_period):
            if (
                isinstance(control_threshold, float)
                and (hi - ref).norm(p=2).mean() < control_threshold
            ):
                # cut the several trajs short!
                error_trajectory = error_trajectory[:, :time]
                action_recovery_error_trajectory = action_recovery_error_trajectory[
                    :, :time
                ]
                action_correction_trajectory = action_correction_trajectory[:, :time]
                control_trajectory = control_trajectory[:, :time]
                ht_trajectory = ht_trajectory[:, :time]
                ht_im1_trajectory = ht_im1_trajectory[:, :time]
                ht_im2_trajectory = ht_im2_trajectory[:, :time]
                action_trajectory = action_trajectory[:, :time]
                out_trajectory = out_trajectory[:, :time]
                hi_trajectory = hi_trajectory[:, :time]
                break
            self.operator.set_state(operator_s)
            (
                hi_trajectory,
                control_out,
                out,
                action_inferred,
                action_correction,
                ht,
                h_plant,
                ht_im1,
                ht_im2,
                control_error,
                action_recovery_error,
                operator_s,
            ) = self.step_controller(
                hi_trajectory, ref, ht, h_plant, ht_im1, ht_im2, action_trajectory, time
            )

            error_trajectory[:, time] = control_error
            action_recovery_error_trajectory[:, time] = action_recovery_error
            control_trajectory[:, time] = control_out
            ht_trajectory[:, time] = ht
            ht_im1_trajectory[:, time] = ht_im1
            ht_im2_trajectory[:, time] = ht_im2
            action_trajectory[:, time] = action_inferred + action_correction
            action_correction_trajectory[:, time] = action_correction.norm(
                p=2, dim=-1
            )  # this is the norm of the action correction
            out_trajectory[:, time] = out

            if time > period_upperlimit:
                print(f"Control period exceeded upper bound with residual error")
                break

        self.to(original_device)
        if return_control_trajectory:
            return (
                hi_trajectory,
                control_trajectory,
                error_trajectory,
                ht_trajectory,
                ht_im1_trajectory,
                ht_im2_trajectory,
                action_trajectory,
                action_recovery_error_trajectory,
                action_correction_trajectory,
                out_trajectory,
            )
        return hi_trajectory, error_trajectory

    def forward_full(
        self,
        h0,
        ref,
        control_period: Optional[int] = None,
        control_threshold: Optional[float] = None,
        period_upperlimit: int = 100,
    ):
        """Forward loop. Controlled for a number of steps or until a certain threshold is reached."""
        assert control_period is not None or control_threshold is not None
        assert control_period is None or control_threshold is None
        chunk_length = 4096  # to avoid memory overflow, we divide inputs into tiny segments when it's too big
        error_trajectory = None
        x_segment = []
        target_segment = []
        time = 0
        if h0.shape[0] > chunk_length:
            left = h0.shape[0]
            segment_start = 0
            while left > 0:
                x_segment.append(
                    h0[segment_start : (segment_start + min(chunk_length, left))]
                )
                target_segment.append(
                    ref[segment_start : (segment_start + min(chunk_length, left))]
                )
                left -= chunk_length
                segment_start += chunk_length
            del h0, ref
            torch.cuda.empty_cache()

            for i in range(len(x_segment)):
                x_seg = x_segment.pop(0)
                target_seg = target_segment.pop(0)
                returned = self.forward_full(
                    x_seg,
                    target_seg,
                    h0,
                    control_period,
                    control_threshold,
                    period_upperlimit,
                )

            return returned

        pos0 = self.readoutnet(h0[..., -self.rep_net.rnn.rep_units :])
        pos0 = pos0 / pos0.norm(p=2, dim=-1).unsqueeze(-1)
        operator_s = pos0
        original_device = self.device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        hi = h0.to(device)  # Note, this is the hidden state of the rep net!
        ht = self.ht0.to(device)
        ht_im1 = self.ht_im10.to(device)
        ht_im2 = self.ht_im20.to(device)
        h_plant = self.h_plant0.to(device)

        ref = ref.to(device)  # let's say we have the same ref throughout?

        if control_period is None:
            control_period = period_upperlimit

        error_trajectory = torch.zeros(hi.shape[0], control_period, device=device)
        action_recovery_error_trajectory = torch.zeros(
            hi.shape[0], control_period, device=device
        )
        action_correction_trajectory = torch.zeros(
            hi.shape[0], control_period, device=device
        )
        control_trajectory = torch.zeros(
            hi.shape[0], control_period, self.output_size, device=device
        )
        out_trajectory = torch.zeros(
            hi.shape[0], control_period, self.action_size, device=device
        )
        ht_trajectory = torch.zeros(
            hi.shape[0], control_period, self.recurrent_control_dim, device=device
        )
        hi_trajectory = torch.zeros(
            hi.shape[0], control_period, self.state_size, device=device
        )
        ht_im1_trajectory = torch.zeros(
            hi.shape[0], control_period, self.hidden_im1_dim, device=device
        )
        ht_im2_trajectory = torch.zeros(
            hi.shape[0], control_period, self.hidden_im2_dim, device=device
        )
        action_inferred_trajectory = torch.zeros(
            hi.shape[0], control_period, self.action_size, device=device
        )
        action_c_trajectory = torch.zeros(
            hi.shape[0], control_period, self.action_size, device=device
        )

        # NOTE: out_trajectory is the true series of actions to the plant; action_trajectory is the inferred action from the IMs
        hi_trajectory[:, -1, :] = hi

        for time in range(control_period):
            if (
                isinstance(control_threshold, float)
                and (hi - ref).norm(p=2).mean() < control_threshold
            ):
                # cut the several trajs short!
                error_trajectory = error_trajectory[:, :time]
                action_recovery_error_trajectory = action_recovery_error_trajectory[
                    :, :time
                ]
                action_correction_trajectory = action_correction_trajectory[:, :time]
                control_trajectory = control_trajectory[:, :time]
                ht_trajectory = ht_trajectory[:, :time]
                ht_im1_trajectory = ht_im1_trajectory[:, :time]
                ht_im2_trajectory = ht_im2_trajectory[:, :time]
                action_inferred_trajectory = action_inferred_trajectory[:, :time]
                action_c_trajectory = action_c_trajectory[:, :time]
                out_trajectory = out_trajectory[:, :time]
                hi_trajectory = hi_trajectory[:, :time]
                break
            self.operator.set_state(operator_s)
            (
                hi_trajectory,
                control_out,
                out,
                action_inferred,
                action_correction,
                ht,
                h_plant,
                ht_im1,
                ht_im2,
                control_error,
                action_recovery_error,
                operator_s,
            ) = self.step_controller(
                hi_trajectory,
                ref,
                ht,
                h_plant,
                ht_im1,
                ht_im2,
                action_c_trajectory + action_inferred_trajectory,
                time,
            )

            error_trajectory[:, time] = control_error
            action_recovery_error_trajectory[:, time] = action_recovery_error
            control_trajectory[:, time] = control_out
            ht_trajectory[:, time] = ht
            ht_im1_trajectory[:, time] = ht_im1
            ht_im2_trajectory[:, time] = ht_im2
            action_inferred_trajectory[:, time] = action_inferred
            action_c_trajectory[:, time] = action_correction
            action_correction_trajectory[:, time] = action_correction.norm(
                p=2, dim=-1
            )  # this is the norm of the action correction
            out_trajectory[:, time] = out

            if time > period_upperlimit:
                print(f"Control period exceeded upper bound with residual error")
                break

        self.to(original_device)
        return (
            hi_trajectory,
            control_trajectory,
            error_trajectory,
            ht_trajectory,
            ht_im1_trajectory,
            ht_im2_trajectory,
            self.operator.s,
            action_inferred_trajectory,
            action_c_trajectory,
            action_recovery_error_trajectory,
            action_correction_trajectory,
            out_trajectory,
        )

    def step_controller(
        self, hi, ref, ht, h_plant, ht_im1, ht_im2, action_memory, time
    ):
        """Modified step_controller to properly index time-delayed information."""
        et = torch.cat([hi[:, time - 1, :], ref], dim=-1).float()

        ht, control_out = self.graph(et, ht)
        h_plant, out = self.plant(control_out, h_plant, self.dt)
        state_feedback, out, operator_s = *self.operator(out), self.operator.s
        # TODO: check with Guillaume on what exactly shall we consider as efference copies
        # we can definitely use control_out;
        # h_plant or out may also be used? or a lowD projection of them?

        sensorium_im1 = torch.cat([state_feedback, control_out], dim=-1).float()
        # print(f"sensorium_im1 shape: {sensorium_im1.shape}, sensorium_im1 type: {sensorium_im1.dtype}")
        # print(f"ht_im1 shape: {ht_im1.shape}, ht_im1 type: {ht_im1.dtype}")

        ht_im1, action_inferred = self.im1(sensorium_im1, ht_im1)

        action_memory[:, time] += action_inferred

        # if time >= self.operator.delay + 1:
        if time >= self.operator.delay:
            sensorium_im2 = torch.cat(
                [
                    state_feedback[:, 0 : self.operator.position_size],
                    hi[:, time - 1 - self.operator.delay],
                ],
                dim=-1,
            ).float()
            pseudo_action = self.im2(sensorium_im2, ht_im2)[1]
            pseudo_action_norm = pseudo_action.norm(p=2, dim=-1)
            pseudo_action_denormed = pseudo_action / pseudo_action_norm.unsqueeze(-1)

            action_mem = action_memory[:, time - self.operator.delay]
            action_mem_norm = action_mem.clone().norm(p=2, dim=-1)
            action_mem_denormed = action_mem.clone() /action_mem_norm.clone().unsqueeze(-1)
            # NOTE: we may also add some memory here? following TEM conventions maybe - let's see if it works without, first.
            # if it works without, then this is similar to stating that all necessary memories are encoded in the CAN
            # action_recovery_error = (
            #     pseudo_action - action_mem
            # ).norm(p=2, dim=-1) - torch.bmm(pseudo_action_denormed.unsqueeze(1) , action_mem_denormed.unsqueeze(-1)).reshape(-1)
            action_recovery_error = -torch.bmm(pseudo_action_denormed.unsqueeze(1), action_mem_denormed.unsqueeze(-1)).reshape(-1) + F.mse_loss(pseudo_action_norm , action_mem_norm, reduce=None)
        else:
            action_recovery_error = torch.zeros_like(action_memory[:, 0, 0])

        hi[:, time] = self.step_CAN(hi[:, time - 1, :].float(), action_inferred.float())
        if time >= self.operator.delay:
            sensorium_im2 = torch.cat(
                [
                    state_feedback[:, 0 : self.operator.position_size],
                    hi[:, time - self.operator.delay],
                ],
                dim=-1,
            ).float()
            ht_im2, action_correction = self.im2(sensorium_im2, ht_im2)
        else:  # we don't have enough data to infer the action correction
            action_correction = torch.zeros_like(action_inferred)
        hi[:, time] = self.step_CAN(hi[:, time, :].float(), action_correction.float())

        action_memory[:, time] = action_memory[:, time] + action_correction

        control_error = (ref - hi[:, time, -ref.shape[-1] :]).norm(p=2, dim=-1)
        return (
            hi,
            control_out,
            out,
            action_inferred,
            action_correction,
            ht,
            h_plant,
            ht_im1,
            ht_im2,
            control_error,
            action_recovery_error,
            operator_s,
        )

    def _step_controller_obsolete(
        self, hi, ref, ht, h_plant, ht_im1, ht_im2, action_memory
    ):
        """
        hi: shape(batch_size, t, input_units * input_size + rep_units)
        ref: shape(batch_size, ref_size)
        ht: shape(batch_size, recurrent_control_dim)
        h_plant: shape(batch_size, hidden_plant_dim)
        ht_im1: shape(batch_size, hidden_im1_dim)
        ht_im2: shape(batch_size, hidden_im2_dim)
        action_memory: shape(batch_size, t, action_size) NOTE: integrated inside for simplicity

        Returns:
        hi: shape(batch_size, t + 1, repnet_hidden_units), current state after update
        out: shape(batch_size, output_size), output of the controller


        """
        # print((ref - hi[:, -ref.shape[-1]:]).norm(p = 2, dim = -1))
        et = torch.cat([hi[:, -1, :], ref], dim=-1).float()

        ht, control_out = self.graph(et, ht)
        h_plant, out = self.plant(control_out, h_plant, self.dt)  # plant steps one step
        state_feedback = self.operator(
            out, self.dt
        )  # operator state is kept in the operator
        ht_im1, action_inferred = self.im1(
            state_feedback, ht_im1
        )  # include ht_imx for potential recurrence in these predictors
        action_memory_ = torch.cat([action_memory, action_inferred.unsqueeze(1)], dim=1)

        _, pseudo_action = (
            self.im2(
                state_feedback[0 : self.operator.position_size],
                hi[:, -1 - self.operator.delay],
                ht_im2,
            )
            if hi.shape[1] - 1 - self.operator.delay >= 0
            else (None, torch.nan)
        )
        action_recovery_error = (
            pseudo_action - action_memory_[:, -1 - self.operator.delay]
        ).norm(
            p=2, dim=-1
        )  # will be nan if we haven't observed enough (delay points into time <= 0)
        # NOTE: we may also add some memory here? following TEM conventions maybe - let's see if it works without, first.
        # if it works without, then this is similar to stating that all necessary memories are encoded in the CAN

        hi_ = self.step_CAN(hi[:, -1, :], action_inferred)
        hi_ = torch.cat([hi, hi_.unsqueeze(1)], dim=1)
        ht_im2, action_correction = self.im2(
            state_feedback[0 : self.operator.position_size],
            hi[:, -1 - self.operator.delay],
            ht_im2,
        )
        hi__ = self.step_CAN(hi_, action_correction)
        hi = torch.cat([hi, hi__.unsqueeze(1)], dim=1)

        control_error = (ref - hi[:, -1, -ref.shape[-1] :]).norm(p=2, dim=-1)

        # print(error)
        return (
            hi,
            control_out,
            out,
            action_inferred,
            action_correction,
            ht,
            h_plant,
            ht_im1,
            ht_im2,
            control_error,
            action_recovery_error,
        )

    def step_CAN(self, hi, velocity_input):
        """
        hi_input: shape(batch_size, input_units * input_size)
        hi_rep: shape(batch_size, rep_units)
        velocity_input: shape(batch_size, input_size)

        Returns:
        hi: shape(batch_size, rep_units), current state after update
        """
        hi = self.rep_net.rnn.step(hi, velocity_input)
        return hi

class PredControlNetCAN_2(nn.Module):  # (torch.jit.ScriptModule):
    def __init__(
        self,
        rep_net,  # these all should be nn.Modules with forward methods
        plant,
        operator,  # path-integrates in the environment
        IM1,  # internal model 1; in this case only one internal model takes care of everything
        predictor,  # predictive network for alignment purpose only. 
        readoutnet,
        output_size=None,
        rep_units_only=False,  # read from representation units only
        recurrent_control=False,  # use simple recurrence here (RNN)
        activation="relu",
        recurrent_control_dim=0,
        hidden_control_dim=0,
        device="cuda",
        use_rep = False, 
        use_feedback = None,
    ):
        super(PredControlNetCAN_2, self).__init__()
        self.rep_net = rep_net
        for param in self.rep_net.parameters():
            param.requires_grad = False

        self.rep_units_only = rep_units_only
        if self.rep_units_only:
            self.ref_size = self.rep_net.rnn.rep_units
        else:
            self.ref_size = self.rep_net.rnn.hidden_size
        self.state_size = (
            self.rep_net.rnn.hidden_size
        )  # note the possibility of the hidden size being different from the state size
        # self.plant = self._init_plant(plant_specs)
        self.plant = plant.to(device)
        self.readoutnet = readoutnet.to(device)

        if output_size is None:
            self.output_size = self.plant.state_dim
        else:
            self.output_size = output_size

        self.action_size = self.rep_net.rnn.input_size

        self.operator = operator.to(device)
        self.im1, self.hidden_im1_dim = IM1.to(device), IM1.recurrent_size
        self.use_rep = use_rep
        self.use_feedback = use_feedback if use_feedback is not None else use_rep
        self.pred, self.hidden_pred_dim = predictor.to(device), getattr(predictor, "recurrent_size", 0)

        self.recurrent_control = recurrent_control
        self.recurrent_control_dim = recurrent_control_dim

        self.device = device
        self.activation = activation
        try:
            self.activation = getattr(nn, self.activation)()
        except:
            try:
                self.activation = getattr(F, self.activation)()
            except:
                self.activation = globals().get(self.activation)()
        self.hidden_size = (
            recurrent_control_dim
            if recurrent_control_dim != 0
            else self.rep_net.rnn.input_size
        )
        self.hidden_control_dim = hidden_control_dim

        # TODO: fix these
        # self.output_p = nn.Sequential(
        #     nn.Linear(self.recurrent_control_dim, self.output_size, bias=False),
        #     self.activation)
        # self.output_m = nn.Sequential(
        #     nn.Linear(self.recurrent_control_dim, self.output_size, bias = False),
        #     self.activation)

        if self.hidden_control_dim == 0:
            self.W_hh = (
                nn.Linear(self.recurrent_control_dim, self.recurrent_control_dim)
                if self.recurrent_control
                else None
            )
            self.W_xh = nn.Linear(self.state_size + self.ref_size, self.hidden_size)
            assert self.hidden_size == self.output_size, self.hidden_size == self.recurrent_control_dim
            self.out = nn.Identity()
        
        else: 
            assert self.recurrent_control_dim == self.hidden_control_dim, "Recurrent control dimension must match the hidden control dimension when using a hidden control layer."
            self.W_hh = nn.Sequential(
                nn.Linear(self.recurrent_control_dim, self.hidden_control_dim),
            ) if self.recurrent_control else None

            self.W_xh = nn.Sequential(
                nn.Linear(self.state_size + self.ref_size, self.hidden_control_dim),
            )

            self.out = nn.Linear(self.hidden_control_dim, self.output_size)

        self.construct_graph()
        self.ht0 = nn.Parameter(
            torch.zeros(1, self.recurrent_control_dim, device=self.device)
        )
        self.ht_im10 = nn.Parameter(
            torch.zeros(1, self.hidden_im1_dim, device=self.device)
        )
        self.ht_pred0 = nn.Parameter(
            torch.zeros(1, self.hidden_pred_dim, device=self.device)
        )
        self.h_plant0 = nn.Parameter(
            torch.zeros(1, self.plant.state_dim, device=self.device),
            requires_grad=False,
        )  # control from nullity
        self.disturbance=False

        assert self.plant.dt == self.operator.dt
        self.dt = self.plant.dt
        self.mask_im = False
        self.controller_activity_mask = False

    def freeze_controller(self):
        for module in [self.W_xh, self.W_hh, self.out]:
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = False
            self.ht0.requires_grad = False
        
    def freeze_im1(self): 
        for param in self.im1.parameters():
            param.requires_grad = False
        self.ht_im10.requires_grad = False
    
    def freeze_pred(self):
        for param in self.pred.parameters():
            param.requires_grad = False
        self.ht_pred0.requires_grad = False
    
    def freeze_plant(self):
        for param in self.plant.parameters():
            param.requires_grad = False
    
    def freeze_operator(self):
        for param in self.operator.parameters():
            param.requires_grad = False

    def freeze_rep_net(self):        
        for param in self.rep_net.parameters():
            param.requires_grad = False
    
    def freeze_readoutnet(self):
        for param in self.readoutnet.parameters():
            param.requires_grad = False
    
    def unfreeze_readoutnet(self):
        for param in self.readoutnet.parameters():
            param.requires_grad = True
            
    def unfreeze_rep_net(self):
        for param in self.rep_net.parameters():
            param.requires_grad = True

    def unfreeze_operator(self):
        for param in self.operator.parameters():
            param.requires_grad = True
    
    def unfreeze_plant(self):
        for param in self.plant.parameters():
            param.requires_grad = True
        
    def unfreeze_im1(self):
        for param in self.im1.parameters():
            param.requires_grad = True
        self.ht_im10.requires_grad = True

    def unfreeze_pred(self):
        for param in self.pred.parameters():
            param.requires_grad = True
        self.ht_pred0.requires_grad = True

    def efference_only(self):
        self.im1.silence_sensory = True
        self.im1.silence_can = True
        self.im1.silence_efference2 = True
        for param in self.im1.input_streams["sensory"].parameters():
            param.requires_grad = False
        for param in self.im1.input_streams["can"].parameters():
            param.requires_grad = False
        for param in self.im1.input_streams["efference2"].parameters():
            param.requires_grad = False
    
    def full_feedback(self, freeze_efference = True):
        self.im1.silence_sensory = False
        self.im1.silence_can = False
        self.im1.silence_efference2 = False
        for param in self.im1.input_streams["sensory"].parameters():
            param.requires_grad = True
        for param in self.im1.input_streams["can"].parameters():
            param.requires_grad = True
        for param in self.im1.input_streams["efference2"].parameters():
            param.requires_grad = True
        if freeze_efference:
            for param in self.im1.input_streams["efference1"].parameters():
                param.requires_grad = False

    def reinitialize_controller(self):
        if self.hidden_control_dim == 0:
            self.W_hh = (
                nn.Linear(self.recurrent_control_dim, self.recurrent_control_dim)
                if self.recurrent_control
                else None
            )
            self.W_xh = nn.Linear(self.state_size + self.ref_size, self.hidden_size)
            assert self.hidden_size == self.output_size, self.hidden_size == self.recurrent_control_dim
            self.out = nn.Identity()
        
        else: 
            self.W_hh = nn.Sequential(
                nn.Linear(self.recurrent_control_dim, self.hidden_control_dim),
            ) if self.recurrent_control else None

            self.W_xh = nn.Sequential(
                nn.Linear(self.state_size + self.ref_size, self.hidden_control_dim),
            )

            self.out = nn.Linear(self.hidden_control_dim, self.output_size)

        self.construct_graph()
        self.ht0 = nn.Parameter(
            torch.zeros(1, self.recurrent_control_dim, device=self.device)
        )

    def unfreeze_controller(self):
        for module in [self.W_xh, self.W_hh, self.out]:
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = True
        self.ht0.requires_grad = True
    
    def disactivate_controller(self, scale=1):
        """Replace the controller output by a random noise, of scale defined"""
        self.mask_controller = True
        self.mask_scale = scale

    def disactivate_im1(self): 
        self.mask_im = True
    
    def activate_im1(self):
        self.mask_im = False
    
    def activate_controller(self):
        self.mask_controller = False

    def introduce_disturbance(self, scale=1, index=None):
        self.disturbance = True
        self.index = index
        self.disturbance_size = scale

    def remove_disturbance(self):
        self.disturbance = False
    
    def float(self):
        self.to("cuda")
        self.rep_net = self.rep_net.float()
        self.plant = self.plant.float()
        self.operator = self.operator.float()
        self.im1 = self.im1.float()
        self.pred = self.pred.float()
        self.readoutnet = self.readoutnet.float()
        self.W_hh = self.W_hh.float() if self.W_hh is not None else None
        self.W_xh = self.W_xh.float()
        self.out = self.out.float()
        self.ht0.data = self.ht0.data.float()
        self.ht_im10.data = self.ht_im10.data.float()
        self.ht_pred0.data = self.ht_pred0.data.float()
        self.h_plant0.data = self.h_plant0.data.float()
        return self

    def construct_graph(self):
        if self.recurrent_control:
            self.graph = self.graph_recurrent
        else:
            self.graph = self.graph_feedforward

    def graph_recurrent(self, et, ht):
        # et is the external input
        x_xh = self.W_xh(et)
        x_hh = self.W_hh(ht)
        h = self.activation(x_hh + x_xh)
        out = self.activation(self.out(h))
        return h, out

    def graph_feedforward(self, et, ht):
        h = self.activation(self.W_xh(et))
        out = self.activation(self.out(h))
        return ht, out

    def forward(
        self,
        h0,
        ref,
        control_period: Optional[int] = None,
        control_threshold: Optional[float] = None,
        period_upperlimit: int = 100,
        return_control_trajectory=False,
        use_s0 = False, 
        s0 = None,
        observe = 0, 
    ):
        """Forward loop. Controlled for a number of steps or until a certain threshold is reached."""
        assert control_period is not None or control_threshold is not None
        assert control_period is None or control_threshold is None
        chunk_length = 4096  # to avoid memory overflow, we divide inputs into tiny segments when it's too big
        error_trajectory = None
        x_segment = []
        target_segment = []
        time = 0
        if h0.shape[0] > chunk_length:
            left = h0.shape[0]
            segment_start = 0
            while left > 0:
                x_segment.append(
                    h0[segment_start : (segment_start + min(chunk_length, left))]
                )
                target_segment.append(
                    ref[segment_start : (segment_start + min(chunk_length, left))]
                )
                left -= chunk_length
                segment_start += chunk_length
            del h0, ref
            torch.cuda.empty_cache()
            
            for i in range(len(x_segment)):
                x_seg = x_segment.pop(0)
                target_seg = target_segment.pop(0)
                if return_control_trajectory:
                    x_, control_trajectory_, error_trajectory_ = self.forward(
                        x_seg,
                        target_seg,
                        h0,
                        control_period,
                        control_threshold,
                        period_upperlimit,
                        return_control_trajectory,
                    )
                    x, control_trajectory, error_trajectory = (
                        (x_, control_trajectory_, error_trajectory_)
                        if error_trajectory is None
                        else (
                            torch.cat([x, x_], dim=0),
                            torch.cat([control_trajectory, control_trajectory_], dim=0),
                            torch.cat([error_trajectory, error_trajectory_], dim=0),
                        )
                    )
                    print(x.shape)
                else:
                    x_, error_trajectory_ = self.forward(
                        x_seg,
                        target_seg,
                        h0,
                        control_period,
                        control_threshold,
                        period_upperlimit,
                        return_control_trajectory,
                    )
                    x, error_trajectory = (
                        (x_, error_trajectory_)
                        if error_trajectory is None
                        else (
                            torch.cat([x, x_], dim=0),
                            torch.cat([error_trajectory, error_trajectory_], dim=0),
                        )
                    )
                    print(x.shape)

            if return_control_trajectory:
                return x, control_trajectory, error_trajectory, ht_trajectory
            return x, error_trajectory

        original_device = self.device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        hi = h0.to(device)  # Note, this is the hidden state of the rep net!
        ht = self.ht0.to(device)
        ht_im1 = self.ht_im10.to(device)
        ht_pred = self.ht_pred0.to(device)
        h_plant = self.h_plant0.to(device)

        if use_s0:
            assert s0 is not None
            pos0 = s0
        else: 
            pos0 = self.readoutnet(h0[..., -self.rep_net.rnn.rep_units :])
        
        pos0 = pos0 / pos0.norm(p=2, dim=-1, keepdim=True)
        operator_s = pos0

        ref = ref.to(device)  # let's say we have the same ref throughout?

        if control_period is None:
            control_period = period_upperlimit

        if self.disturbance:
            disturbance_index = self.index if self.index is not None else np.random.randint(1, control_period)
            disturbance = torch.randn((hi.shape[0], self.action_size), device=self.device) * self.disturbance_size

        error_trajectory = torch.zeros(hi.shape[0], control_period, device=device)
        state_prediction_error_trajectory = torch.zeros(
            hi.shape[0], control_period, device=device
        )
        
        control_trajectory = torch.zeros(
            hi.shape[0], control_period, self.output_size, device=device
        )
        out_trajectory = torch.zeros(
            hi.shape[0], control_period, self.action_size, device=device
        )
        ht_trajectory = torch.zeros(
            hi.shape[0], control_period, self.recurrent_control_dim, device=device
        )
        hi_trajectory = torch.zeros(
            hi.shape[0], control_period, self.state_size, device=device
        )
        ht_im1_trajectory = torch.zeros(
            hi.shape[0], control_period, self.hidden_im1_dim, device=device
        )
        ht_pred_trajectory = torch.zeros(
            hi.shape[0], control_period, self.hidden_pred_dim, device=device
        )
        action_trajectory = torch.zeros(
            hi.shape[0], control_period, self.operator.position_size, device=device
        )
        state_prediction_trajectory = torch.zeros(
            hi.shape[0], control_period, self.operator.position_size, device=device
        )
        # NOTE: out_trajectory is the true series of actions to the plant; action_trajectory is the inferred action from the IMs
        hi_trajectory[:, -1, :] = hi
        self.operator.set_state(operator_s)

        while observe > 0:
            hi_trajectory, ht_im1 = self.observe(hi_trajectory, ht_im1, self.operator.s)
            observe -= 1


        for time in range(control_period):
            if (
                isinstance(control_threshold, float)
                and (hi - ref).norm(p=2).mean() < control_threshold
            ):
                # cut the several trajs short!
                error_trajectory = error_trajectory[:, :time]
                state_prediction_error_trajectory = state_prediction_error_trajectory[
                    :, :time
                    ]
                state_prediction_trajectory = state_prediction_trajectory[:, :time]
                control_trajectory = control_trajectory[:, :time]
                ht_trajectory = ht_trajectory[:, :time]
                ht_im1_trajectory = ht_im1_trajectory[:, :time]
                ht_pred_trajectory = ht_pred_trajectory[:, :time]
                action_trajectory = action_trajectory[:, :time]
                out_trajectory = out_trajectory[:, :time]
                hi_trajectory = hi_trajectory[:, :time]
                break
                
            self.operator.set_state(operator_s)
            if self.disturbance and time == disturbance_index:
                # integrate the *current* operator_s (B×dim), not self.operator.s
                new_s = self.operator.integrate_joint_velocities(disturbance, operator_s)

                # zero out the velocity channels WITHOUT in-place on new_s
                pos = new_s[..., :self.operator.position_size]
                vel = torch.zeros_like(new_s[..., self.operator.position_size:])
                new_s = torch.cat([pos, vel], dim=-1)

                # update the operator’s buffer in one go
                self.operator.set_state(new_s)

            # if self.disturbance and time == disturbance_index:
            #     s = self.operator.integrate_joint_velocities(disturbance, self.operator.s)
            #     s[..., self.operator.position_size : ] = torch.zeros_like(s[..., self.operator.position_size : ])
            #     self.operator.s[:, -1, :] = s

            if time == 0:
                hi_trajectory,ht_im1 = self.observe(hi_trajectory, ht_im1, self.operator.s) # to adapt to the current operator state (with zero velocity) so that the initial CAN state can be aligned.
            (
                hi_trajectory,
                control_out,
                out,
                action_inferred,
                s_predicted,
                ht,
                h_plant,
                ht_im1,
                ht_pred,
                control_error,
                sensory_prediction_error,
                operator_s,
            ) = self.step_controller(
                hi_trajectory, ref, ht, h_plant, ht_im1, ht_pred, time
            )

            error_trajectory[:, time] = control_error
            state_prediction_error_trajectory[:, time] = sensory_prediction_error
            control_trajectory[:, time] = control_out
            ht_trajectory[:, time] = ht
            ht_im1_trajectory[:, time] = ht_im1
            ht_pred_trajectory[:, time] = ht_pred
            action_trajectory[:, time] = action_inferred 
            state_prediction_trajectory[:, time] = s_predicted

            out_trajectory[:, time] = out

            if time > period_upperlimit:
                print(f"Control period exceeded upper bound with residual error")
                break

        self.to(original_device)
        if return_control_trajectory:
            return (
                hi_trajectory,
                control_trajectory,
                error_trajectory,
                ht_trajectory,
                ht_im1_trajectory,
                ht_pred_trajectory,
                action_trajectory,
                state_prediction_error_trajectory,
                state_prediction_trajectory,
                out_trajectory,
            )
        return hi_trajectory, error_trajectory

    def forward_full(
        self,
        h0,
        ref,
        control_period: Optional[int] = None,
        control_threshold: Optional[float] = None,
        period_upperlimit: int = 1000,
        inspect=False,
        use_s0 = False, 
        s0 = None,
        observe = 0,
    ):
        """Forward loop. Controlled for a number of steps or until a certain threshold is reached."""
        assert control_period is not None or control_threshold is not None
        assert control_period is None or control_threshold is None
        self._last_sensorium_im1 = []
        chunk_length = 4096  # to avoid memory overflow, we divide inputs into tiny segments when it's too big
        error_trajectory = None
        x_segment = []
        target_segment = []
        time = 0
        if h0.shape[0] > chunk_length:
            left = h0.shape[0]
            segment_start = 0
            while left > 0:
                x_segment.append(
                    h0[segment_start : (segment_start + min(chunk_length, left))]
                )
                target_segment.append(
                    ref[segment_start : (segment_start + min(chunk_length, left))]
                )
                left -= chunk_length
                segment_start += chunk_length
            del h0, ref
            torch.cuda.empty_cache()

            for i in range(len(x_segment)):
                x_seg = x_segment.pop(0)
                target_seg = target_segment.pop(0)
                returned = self.forward_full(
                    x_seg,
                    target_seg,
                    h0,
                    control_period,
                    control_threshold,
                    period_upperlimit,
                )

            return returned

        if use_s0:
            assert s0 is not None
            pos0 = s0
        else: 
            pos0 = self.readoutnet(h0[..., -self.rep_net.rnn.rep_units :])

        if ref.dim() == 2: 
            ref = ref.unsqueeze(1)
        _, N_targets, d = ref.shape
        total_T = control_period * N_targets
        ref_train = ref.unsqueeze(2).expand(-1, -1, control_period, -1).contiguous().view(-1, total_T, d)

        pos0 = pos0 / pos0.norm(p=2, dim=-1).unsqueeze(-1)
        operator_s = pos0
        original_device = self.device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        hi = h0.to(device)  # Note, this is the hidden state of the rep net!


        ht = self.ht0.to(device)
        ht_im1 = self.ht_im10.to(device)
        ht_pred = self.ht_pred0.to(device)
        h_plant = self.h_plant0.to(device)

        ref_train = ref_train.to(device)  # let's say we have the same ref throughout?

        if self.disturbance:
            disturbance_index = self.index if self.index is not None else np.random.randint(1, control_period)
            disturbance = torch.randn((hi.shape[0], self.action_size), device=self.device) * self.disturbance_size

        if control_period is None:
            control_period = period_upperlimit

        error_trajectory = torch.zeros(hi.shape[0], total_T, device=device)
        state_prediction_error_trajectory = torch.zeros(
            hi.shape[0], total_T, device=device
        )

        control_trajectory = torch.zeros(
            hi.shape[0], total_T, self.output_size, device=device
        )
        out_trajectory = torch.zeros(
            hi.shape[0], total_T, self.action_size, device=device
        )
        ht_trajectory = torch.zeros(
            hi.shape[0], total_T, self.recurrent_control_dim, device=device
        )
        hi_trajectory = torch.zeros(
            hi.shape[0], total_T, self.state_size, device=device
        )
        ht_im1_trajectory = torch.zeros(
            hi.shape[0], total_T, self.hidden_im1_dim, device=device
        )
        ht_pred_trajectory = torch.zeros(
            hi.shape[0], total_T, self.hidden_pred_dim, device=device
        )
        action_inferred_trajectory = torch.zeros(
            hi.shape[0], total_T, self.action_size, device=device
        )
        action_contribution_afference_trajectory = torch.zeros(
            hi.shape[0], total_T, self.action_size, device=device
        )
        action_contribution_efference_trajectory = torch.zeros(
            hi.shape[0], total_T, self.action_size, device=device
        )
        state_prediction_trajectory = torch.zeros(
            hi.shape[0], total_T, self.operator.position_size, device=device
        )


        # NOTE: out_trajectory is the true series of actions to the plant; action_trajectory is the inferred action from the IMs
        hi_trajectory[:, -1, :] = hi

        self.operator.set_state(operator_s)

        while observe > 0:
            hi_trajectory, ht_im1 = self.observe(hi_trajectory, ht_im1, self.operator.s)
            observe -= 1

        for time in range(total_T):
            if (
                isinstance(control_threshold, float)
                and (hi - ref_train[:, time, :]).norm(p=2).mean() < control_threshold
            ):
                # cut the several trajs short!
                error_trajectory = error_trajectory[:, :time]
                state_prediction_error_trajectory = state_prediction_error_trajectory[
                    :, :time
                ]
                control_trajectory = control_trajectory[:, :time]
                ht_trajectory = ht_trajectory[:, :time]
                ht_im1_trajectory = ht_im1_trajectory[:, :time]
                ht_pred_trajectory = ht_pred_trajectory[:, :time]
                action_inferred_trajectory = action_inferred_trajectory[:, :time]
                if inspect:
                    action_contribution_afference_trajectory = action_contribution_afference_trajectory[:, :time]
                    action_contribution_efference_trajectory = action_contribution_efference_trajectory[:, :time]
                state_prediction_trajectory = state_prediction_trajectory[:, :time]
                out_trajectory = out_trajectory[:, :time]
                hi_trajectory = hi_trajectory[:, :time]
                break
            self.operator.set_state(operator_s)

            if self.disturbance and time == disturbance_index:
                s = self.operator.integrate_joint_velocities(disturbance, self.operator.s)
                s[..., self.operator.position_size : ] = torch.zeros_like(s[..., self.operator.position_size : ])
                self.operator.s[:, -1, :] = s
                out_trajectory[:, time-1, :] += disturbance
                # recalculate error: 
                state_prediction_error_trajectory[:, time-1] = (self.operator.s[:, -1, :self.operator.position_size] - state_prediction_trajectory[:, time-1]).norm(p=2, dim=-1)
                
            if time == 0:
                hi_trajectory,ht_im1 = self.observe(hi_trajectory, ht_im1, self.operator.s) # to adapt to the current operator state (with zero velocity) so that the initial CAN state can be aligned.

            if not inspect:
                (
                    hi_trajectory,
                    control_out,
                    out,
                    action_inferred,
                    s_predicted,
                    ht,
                    h_plant,
                    ht_im1,
                    ht_pred,
                    control_error,
                    sensory_prediction_error,
                    operator_s,
                ) = self.step_controller(
                    hi_trajectory,
                    ref_train[:, time, :],
                    ht,
                    h_plant,
                    ht_im1,
                    ht_pred,
                    time,
                ) 
            else:
                (
                    hi_trajectory,
                    control_out,
                    out,
                    action_inferred,
                    s_predicted,
                    ht,
                    h_plant,
                    ht_im1,
                    ht_pred,
                    control_error,
                    sensory_prediction_error,
                    operator_s,
                    action_contribution_afference, 
                    action_contribution_efference,
                ) = self.step_controller_inspect(
                    hi_trajectory,
                    ref_train[:, time, :],
                    ht,
                    h_plant,
                    ht_im1,
                    ht_pred,
                    time,
                )

            error_trajectory[:, time] = control_error
            state_prediction_error_trajectory[:, time] = sensory_prediction_error
            control_trajectory[:, time] = control_out
            ht_trajectory[:, time] = ht
            ht_im1_trajectory[:, time] = ht_im1
            ht_pred_trajectory[:, time] = ht_pred
            action_inferred_trajectory[:, time] = action_inferred
            if inspect:
                action_contribution_afference_trajectory[:, time] = action_contribution_afference
                action_contribution_efference_trajectory[:, time] = action_contribution_efference
            state_prediction_trajectory[:, time] = s_predicted
            out_trajectory[:, time] = out

            if time > period_upperlimit:
                print(f"Control period exceeded upper bound with residual error")
                break

        self.to(original_device)
        if inspect:
            return (
                hi_trajectory,
                control_trajectory,
                error_trajectory,
                ht_trajectory,
                ht_im1_trajectory,
                ht_pred_trajectory,
                self.operator.s,
                action_inferred_trajectory,
                state_prediction_error_trajectory,
                state_prediction_trajectory,
                out_trajectory,
                action_contribution_afference_trajectory,
                action_contribution_efference_trajectory,
            )
        else:
            return (
                hi_trajectory,
                control_trajectory,
                error_trajectory,
                ht_trajectory,
                ht_im1_trajectory,
                ht_pred_trajectory,
                self.operator.s,
                action_inferred_trajectory,
                state_prediction_error_trajectory,
                state_prediction_trajectory,
                out_trajectory,
            )

    
    def step_controller(
        self, hi, ref, ht, h_plant, ht_im1, ht_pred, time, 
    ):
        """Modified step_controller to properly index time-delayed information."""
        et = torch.cat([hi[:, time - 1, :], ref], dim=-1).float()

        if not self.mask_controller: 
            ht, control_out = self.graph(et, ht)
        else: 
            ht, control_out = self.graph(et, ht)
            control_out = self.activation(torch.randn_like(control_out) * self.mask_scale)
        
        if self.controller_activity_mask: 
            control_out = control_out * self.tuning_masker_for_trial # each would be of B, N shape
        
        h_plant, plant_out = self.plant(control_out, h_plant, self.dt)
        # print(f"out shape: {out.shape}")
        state_feedback, out = self.operator(plant_out)
        operator_s = self.operator.s

        if not self.use_rep: 
            if not self.use_feedback:
                sensorium_im1 = control_out.float()
            else:
                # sensorium_im1 = torch.cat([state_feedback, control_out], dim=-1).float()
                sensorium_im1 = torch.cat([state_feedback, plant_out], dim = -1).float()
        else: 
            # sensorium_im1 = torch.cat([hi[:, time - 1, -self.rep_net.rnn.rep_units:], state_feedback, control_out], dim=-1).float()
            sensorium_im1 = torch.cat([hi[:, time - 1, -self.rep_net.rnn.rep_units:], state_feedback, control_out, plant_out], dim=-1).float()
        # print(f"sensorium_im1 shape: {sensorium_im1.shape}, sensorium_im1 type: {sensorium_im1.dtype}")
        # print(f"ht_im1 shape: {ht_im1.shape}, ht_im1 type: {ht_im1.dtype}")
        try:
            self._last_sensorium_im1.append(sensorium_im1.clone().detach())
        except: 
            assert True

        ht_im1, action_inferred = self.im1(sensorium_im1, ht_im1)
        if self.mask_im: 
            action_inferred = out

        hi[:, time] = self.step_CAN(hi[:, time - 1, :].float(), action_inferred.float())
        sensorium_pred = torch.cat(
            [
                hi[:, time],
            ],
            dim=-1,
        ).float()

        try: 
            ht_pred, s_predicted = self.pred(sensorium_pred, ht_pred)
        except TypeError:
            s_predicted = self.pred(sensorium_pred[..., -self.rep_net.rnn.rep_units :])
            ht_pred = torch.zeros_like(ht_pred)
        # print(f"state feedback {state_feedback}")
        # print(f"operator_s {operator_s[..., -1, :]}")
        # print(f"predicted s {s_predicted}")

        # print(f"out {out}")       
        # print(f'action inferred {action_inferred}')
        
        sensory_prediction_error = (self.operator.s[:, -1, :self.operator.position_size]- s_predicted).norm(p=2, dim=-1) 

        # print(f"state feedback = {state_feedback}")
        # print(f"state predicted = {s_predicted}")

        control_error = (ref - hi[:, time, -ref.shape[-1] :]).norm(p=2, dim=-1) 
        
        # ref_predicted = self.readoutnet(ref)
        # control_error = (ref_predicted - state_feedback[:, :self.operator.position_size]).norm(p = 2, dim = -1) ** 2
        return (
            hi,
            control_out,
            out,
            action_inferred,
            s_predicted,
            ht,
            h_plant,
            ht_im1,
            ht_pred,
            control_error,
            sensory_prediction_error,
            operator_s,
        )

    def step_CAN(self, hi, velocity_input):
        """
        hi_input: shape(batch_size, input_units * input_size)
        hi_rep: shape(batch_size, rep_units)
        velocity_input: shape(batch_size, input_size)

        Returns:
        hi: shape(batch_size, rep_units), current state after update
        """
        hi = self.rep_net.rnn.step(hi, velocity_input)
        return hi

    def observe(self, hi, ht_im1, state_feedback):
        """
        No control out, just observe the state and use IM. 
        """

        control_out = torch.zeros((hi.shape[0], self.output_size), device=self.device)
    
        if not self.use_rep: 
            if not self.use_feedback: 
                sensorium_im1 = control_out.float()
            else:
                sensorium_im1 = torch.cat([state_feedback, control_out], dim=-1).float()
        else: 
            sensorium_im1 = torch.cat([hi[:, - 1, -self.rep_net.rnn.rep_units:], state_feedback, control_out, control_out], dim=-1).float()

        ht_im1, action_inferred = self.im1(sensorium_im1, ht_im1)

        hi[:, -1] = self.step_CAN(hi[:, -1, :].float(), action_inferred.float())
        return hi, ht_im1

    def step_controller_inspect(
        self, hi, ref, ht, h_plant, ht_im1, ht_pred, time
    ):
        """Modified step_controller to properly index time-delayed information."""
        et = torch.cat([hi[:, time - 1, :], ref], dim=-1).float()

        if not self.mask_controller: 
            ht, control_out = self.graph(et, ht)
        else: 
            ht, control_out = self.graph(et, ht)
            control_out = torch.randn_like(control_out) * self.mask_scale

        
        h_plant, out = self.plant(control_out, h_plant, self.dt)
        state_feedback, out, operator_s = *self.operator(out), self.operator.s


        if not self.use_rep: 
            sensorium_im1 = torch.cat([state_feedback, control_out], dim=-1).float()
        else: 
            sensorium_im1 = torch.cat([hi[:, time - 1, -self.rep_net.rnn.rep_units:], state_feedback, control_out], dim=-1).float()
        # print(f"sensorium_im1 shape: {sensorium_im1.shape}, sensorium_im1 type: {sensorium_im1.dtype}")
        # print(f"ht_im1 shape: {ht_im1.shape}, ht_im1 type: {ht_im1.dtype}")


        ht_im1 = ht_im1
        action_contribution_afference =  sensorium_im1[..., :-self.output_size] @ (self.im1.W_io.weight.T[:-self.output_size, :])
        action_contribution_efference =  sensorium_im1[..., -self.output_size:] @ (self.im1.W_io.weight.T[-self.output_size:, :])
        action_inferred = self.im1.activation(action_contribution_afference + action_contribution_efference + self.im1.W_io.bias)
        # ht_im1, action_inferred = self.im1(sensorium_im1, ht_im1)

        hi[:, time] = self.step_CAN(hi[:, time - 1, :].float(), action_inferred.float())
        sensorium_pred = torch.cat(
            [
                hi[:, time],
            ],
            dim=-1,
        ).float()

        try: 
            ht_pred, s_predicted = self.pred(sensorium_pred, ht_pred)
        except TypeError:
            s_predicted = self.pred(sensorium_pred[..., -self.rep_net.rnn.rep_units :])
            ht_pred = torch.zeros_like(ht_pred)
        # print(f"state feedback {state_feedback}")
        # print(f"operator_s {operator_s[..., -1, :]}")
        # print(f"predicted s {s_predicted}")

        # print(f"out {out}")       
        # print(f'action inferred {action_inferred}')
        
        sensory_prediction_error = (self.operator.s[:, -1, :self.operator.position_size]- s_predicted).norm(p=2, dim=-1)

        # print(f"state feedback = {state_feedback}")
        # print(f"state predicted = {s_predicted}")

        control_error = (ref - hi[:, time, -ref.shape[-1] :]).norm(p=2, dim=-1)
        return (
            hi,
            control_out,
            out,
            action_inferred,
            s_predicted,
            ht,
            h_plant,
            ht_im1,
            ht_pred,
            control_error,
            sensory_prediction_error,
            operator_s,
            action_contribution_afference,
            action_contribution_efference,
        )

    def forward_return_hidden(
        self,
        h0,
        ref,
        control_period: Optional[int] = None,
        control_threshold: Optional[float] = None,
        period_upperlimit: int = 1000,
        inspect=False,
        use_s0 = False, 
        s0 = None,
        observe = 0,
        include: Union[str, Dict[str, List[str]]] = "everything",
    ):
        """
        Run the full controller and return concatenated network activities according to `include`.
        `include` can be a dict with keys in ['im1','pred','controller','rep_net']
         and values subsets of ['hidden','output'] (for all except 'rep_net') or ['input','rep'] for 'rep_net'.
        Shorthand strings:
          - 'everything': all modules all their signals
          - 'hidden': all modules hidden only + rep_net both
          - 'output': all modules output only + rep_net both
          - 'hidden+rep': hidden of im1,pred,controller and rep only
          - 'output+rep': output of im1,pred,controller and rep only
          - 'everything+rep': hidden+output for im1,pred,controller and rep only
        Returns:
          - network_activities: Tensor of shape (T, batch, total_dim)
          - self.output_log: dict mapping module->subsignal->[start,end)
        """
        # --- parse `include` shorthand ---
        all_keys = ['im1', 'pred', 'controller', 'rep_net']
        base = {
            'im1': ['hidden','output'],
            'pred': ['hidden','output'],
            'controller': ['hidden','output'],
            'rep_net': ['input','rep']
        }
        if isinstance(include, str):
            s = include.lower()
            if s == 'everything':
                inc = base.copy()
            elif s == 'hidden':
                inc = {m: (['hidden'] if m!='rep_net' else ['input','rep']) for m in all_keys}
            elif s == 'output':
                inc = {m: (['output'] if m!='rep_net' else ['input','rep']) for m in all_keys}
            elif s == 'hidden+rep':
                inc = {m: (['hidden'] if m!='rep_net' else ['rep']) for m in all_keys}
            elif s == 'output+rep':
                inc = {m: (['output'] if m!='rep_net' else ['rep']) for m in all_keys}
            elif s == 'everything+rep':
                inc = {m: (base[m] if m!='rep_net' else ['rep']) for m in all_keys}
            else:
                raise ValueError(f"Unknown include shorthand: {include}")
        else:
            # assume dict, filter only allowed modules
            inc = {m: include[m] for m in all_keys if m in include}

        # prepare storage
        self.output_log = {}
        buffers = {m: {sig: [] for sig in inc.get(m, [])} for m in inc}

        # run original forward without slicing to get trajectories
        returned = self.forward_full(
            h0, ref, control_period, control_threshold, period_upperlimit, inspect, use_s0=use_s0, s0=s0, observe=observe
        )
        hi_traj = returned[0]
        control_traj = returned[1]
        B, T, _ = hi_traj.shape

        _, N_targets, d = ref.shape
        total_T = control_period * N_targets
        ref_train = ref.unsqueeze(2).expand(-1, -1, control_period, -1).contiguous().view(-1, total_T, d)

        # iterate timepoints to collect activities

        ht = self.ht0.expand(B, -1)
        # print(inc)
        for t in range(T):
            # rep_net: hi_traj[t]
            if 'rep_net' in inc:
                hid = hi_traj[:, t]  # (B, state_size)
                inp, rep = hid[:, :self.state_size-self.rep_net.rnn.rep_units], hid[:, -self.rep_net.rnn.rep_units:]
                if 'input' in inc['rep_net']:
                    buffers['rep_net']['input'].append(inp)
                if 'rep' in inc['rep_net']:
                    buffers['rep_net']['rep'].append(rep)

            # controller: hidden=ht, output=control_traj[t]
            if 'controller' in inc:
                if 'hidden' in inc['controller']:
                    buffers['controller']['hidden'].append(ht)
                    # print(f"ht shape: {ht.shape}")
                if 'output' in inc['controller']:
                    buffers['controller']['output'].append(control_traj[:, t])
                    # print(f"control_traj shape: {control_traj[:, t].shape}")
            if 'im1' in inc:
                s = self._last_sensorium_im1[t]

                if hasattr(self.im1, "W_hidden") and hasattr(self.im1, "W_output"):
                    # Single-layer MLP (legacy)
                    h_hidden = self.im1.W_hidden(s)
                    h_out = self.im1.W_output(self.im1.activation(h_hidden))

                else:
                    # Multi-layer MLP (InternalModelMLP)
                    h, h_out = self.im1(s)  # h = unused recurrent state (None), h_out = output
                    # For hidden, extract last hidden layer activation
                    hidden = 0
                    for key, lin in self.im1.input_streams.items():
                        start, end = self.im1.partitions[key]
                        hidden += lin(s[:, start:end])
                    hidden = self.im1.activation(hidden)
                    for layer in self.im1.hidden_layers:
                        hidden = self.im1.activation(layer(hidden))
                    h_hidden = hidden

                if 'hidden' in inc['im1'] and h_hidden is not None:
                    buffers['im1']['hidden'].append(h_hidden)
                    # print(f"h_hidden shape: {h_hidden.shape}")
                if 'output' in inc['im1']:
                    buffers['im1']['output'].append(h_out)
                    # print(f"h_out shape: {h_out.shape}")
            # pred: hidden and output
            if 'pred' in inc:
                p_in = hi_traj[:, t]
                h_h = self.pred.W_hidden(p_in)
                h_o = self.pred.W_output(self.pred.activation(h_h))
                if 'hidden' in inc['pred']:
                    buffers['pred']['hidden'].append(h_h)
                if 'output' in inc['pred']:
                    buffers['pred']['output'].append(h_o)

            # step controller state update for next ht (reuse graph)
            et = torch.cat([hi_traj[:, t], ref_train[:, t, :]], dim=-1)
            ht, _ = self.graph(et, ht)

        # stack and concatenate
        concat_list = []
        idx = 0
        for m in all_keys:
            if m not in buffers: continue
            self.output_log[m] = {}
            for sig in buffers[m]:
                arr = torch.stack(buffers[m][sig], dim=0)  # (T, B, dim)
                D = arr.size(-1)
                concat_list.append(arr)
                self.output_log[m][sig] = [idx, idx+D]
                idx += D
        network_activities = torch.cat(concat_list, dim=-1).permute(1, 0, 2)
        return returned, network_activities
    
    
    # def dynamic_subsampling(self, q0, q1, tuning, threshold = np.pi): 
    #     """
    #     dynamically subsample neurons based on a tuning map. 
    #     only select neurons whose q1 - q0 is within a certain range. 

    #     Args: 
    #         q0: initial state of the neurons (B, N)
    #         q1: final state of the neurons (B, N)
    #         tuning: a map of neuron indices to their tuning values (N, 2)
    #     """
    #     # diff = q1 - q0 # (B, N)
    #     # tuning_diff = (tuning[:, 1] - tuning[:, 0]).unsqueeze(0)
    #     tuning_masker = torch.zeros([q0.shape[0], tuning.shape[0]]).long().to(q0.device) # one-hot encoding of eligibility map
    #     for i in range(q0.shape[0]):
    #         wrapped_tuning = (tuning - q0[i]) % (2 * np.pi) 
    #         diff = wrapped_tuning[:, 1] - wrapped_tuning[:, 0]
    #         tuning_masker[i, :] = (diff.abs() < threshold).long()
            
    #     self.tuning_masker_for_trial = tuning_masker
            

    def dynamic_subsampling(self, q0, q1, tuning, threshold=np.pi):
        """
        Vectorized dynamic subsampling based on tuning map and input shifts.

        Args:
            q0: torch.Tensor (B, N), initial neural state
            q1: torch.Tensor (B, N), final neural state (unused in current logic)
            tuning: torch.Tensor (N, 2), tuning map [start, end] per neuron
            threshold: float, max allowed angular tuning span for inclusion
        """

        q1_expanded = q1.unsqueeze(2) if q1.ndim == 2 else q1   # ensure shape (B, N, 1)
        q0_expanded = q0.unsqueeze(2)                           # shape (B, N, 1)

        qc = qc_calc(q1_expanded, q0_expanded) + np.pi          # shape (B, N, 1)
        # qc is now the “new zero” around which we wrap.

        # ----------------------------------------------------
        # 2) Extract each neuron’s θ₀, θ₁ from `tuning`, then subtract qc and wrap into [−π,+π).
        #
        #    tuning has shape (N,2), where tuning[:,0]=θ₀ and tuning[:,1]=θ₁ for each neuron.
        #    We want t0 = wrap( θ₀ − center ), t1 = wrap( θ₁ − center ), with center = qc.
        #
        #    - tuning0: shape (N,)
        #    - tuning1: shape (N,)
        #    We broadcast these to shape (B, N, 1) by doing `[None,:,None]`.
        tuning0 = tuning[:, 0]                    # (N,)
        tuning1 = tuning[:, 1]                    # (N,)

        # Shape (1, N, 1)  minus  (B, N, 1)  → broadcast to (B, N, 1)
        t0 = wrap(tuning0[None, :, None] - qc)    # (B, N, 1)
        t1 = wrap(tuning1[None, :, None] - qc)    # (B, N, 1)

        # ----------------------------------------------------
        # 3) Form s = t0 + t1,  d = t0 - t1. Both are shape (B, N, 1).
        s = t0 + t1       # shape (B, N, 1)
        d = t0 - t1       # shape (B, N, 1)

        # 4) The “middle square” condition is |s| ≤ π  and  |d| ≤ π.
        mask = (s.abs() <= torch.pi) & (d.abs() <= torch.pi)  # shape (B, N, 1), boolean

        # 5) Convert to long and squeeze off the last dimension → shape (B, N)
        tuning_masker = 1-mask.squeeze(-1).long()

        self.tuning_masker_for_trial = tuning_masker
        # return tuning_masker, qc

    def dynamic_sampling_on(self): 
        """
        Turn on the mask for the controller and IMs.
        """
        self.controller_activity_mask = True
        print("Subsampling is ON")
    
    def dynamic_sampling_off(self):
        self.controller_activity_mask = False
        print("Subsampling is OFF")

def wrap(x): 
    return(x + np.pi) % (2 * np.pi) - np.pi

def qc_calc(q1, q2): 
    S       = ((q1 + q2 + np.pi) % (2*np.pi) - np.pi) /2   # ∈ [0,2π) 
    S_tilde = S - np.pi                            # ∈ [−π,+π)

    # Compute the “wrapped difference” between S_tilde and q1:
    diff_wrapped = wrap(S_tilde - q1)           # ∈ [−π,+π)
    mask         = (diff_wrapped.abs() < (np.pi / 2))  # boolean tensor (B,1)

    # Select: if mask[i] is True → S_tilde[i], else → S[i]
    q0 = torch.where(mask, S_tilde, S)
    return q0 