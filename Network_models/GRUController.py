"""
This module implements two types of GRU controller
The typing is based on the computational graph; 

The first one implements the control problem in one step (albeit can be layered)
- u_t, h_t = f(x_goal, h_{t - 1}, m_{t - 1}, s_{t - 1}) where m and s are the muscle efference and sensory feedback

unlike the two-stepped:
- u_t = f(x_goal, h_{t - 1}): u_t | x_goal, h_{t - 1} \perp m_{t - 1}, s_{t - 1}
- h_{t - 1} = f(h_{t - 2}, m_{t - 1}, s_{t - 1}): h_t | h_{t - 1} \perp x_goal, u_t

- m_t, s_t = f_nature(u_t) 
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from typing import Mapping, Optional
from utils.path_settings import *
from copy import deepcopy
from scipy.linalg import expm
from .Activations import *
import numpy as np

import time

class GRUControllerAlltogether(nn.Module): 
    def __init__(
        self,
        plant,
        operator, 
        hidden_readout_size: int, # hidden size for the readout layer; if 0 then just linear
        hidden_sizes: list,    # list of hidden sizes for each GRU layer, e.g. [64] for one layer, or [64, 32] for two layers
        output_size: int,      # dimension of u_t (control command)
        action_size: int,      # for plant output; you may use same as output_size or separate
        observe_output: bool = True,  # the output layer reads from the last hidden layer
        observe_x_goal: bool = True, # whether to observe the x_goal
        observe_initial_condition: bool = False, # whether to observe the initial condition
        device: str = "cuda"   # device to run on
    ):
        """arguments will specify various things, but mostly to setup the multi-layered GRU structure"""
        # specify the network architecture, activations & c
        # specify the observation behavior : 
            # do we implement the observation as a part of the training problem? 
            # NOTE: this may in fact be another inductive bias we put
            # because if we be like "hey there is a free period for you to prepare your hidden state to represent your initial condition, 
            # that will probably mean different things for if we provide target information or not. 
            # the most general will be if we hand out target + initial condition
            # in that case we can fish for the double-ring representation
            # but maybe it is simpler to just represent a single ring of the residual movement? Though that may be difficult to integrate with sensory input

        # pytorch GRUs are built in tanh

        super(GRUControllerAlltogether, self).__init__()
        self.plant = plant
        self.operator = operator
        # silence the grads for the plant and operator
        for param in self.plant.parameters():
            param.requires_grad = False
        for param in self.operator.parameters():
            param.requires_grad = False

        self.device = device
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        self.output_size = output_size
        self.action_size = action_size
        self.observe_output = observe_output

        self.input_size = self.operator.state_size + self.operator.position_size + self.plant.dim + self.output_size
        # Create a ModuleList of GRU cells.
        # For the first layer, the input dimension is input_size.
        # For subsequent layers, the input dimension is the hidden size of the previous layer.
        self.gru_cells = nn.ModuleList()
        for layer in range(self.num_layers):
            in_dim = self.input_size + sum(hidden_sizes) - hidden_sizes[layer]
            # in_dim = self.input_size if layer == 0 else hidden_sizes[layer - 1] + self.input_size
            self.gru_cells.append(nn.GRUCell(in_dim, hidden_sizes[layer]))

        # Create trainable initial hidden states for each layer.
        # Each is of shape (1, hidden_size) and later will be expanded for the batch.
        self.h0 = nn.ParameterList()
        for h_dim in hidden_sizes:
            init_state = nn.Parameter(torch.zeros(1, h_dim))
            self.h0.append(init_state)

        # Compute total hidden size when concatenated.
        self.hidden_size_total = sum(hidden_sizes)
        readout_input_size = self.hidden_size_total if not observe_output else hidden_sizes[-1]

        # Readout MLP: takes the concatenated hidden states and produces u_t.
        # (You may change the hidden layer structure of the MLP as needed.)
        if hidden_readout_size > 0:
            self.readout = nn.Sequential(
                nn.Linear(readout_input_size, hidden_readout_size),
                nn.ReLU(),
                nn.Linear(hidden_readout_size, self.output_size),
            )
        else:
            self.readout = nn.Sequential(
                nn.Linear(readout_input_size, self.output_size),
            )
        self.hidden_readout_size = hidden_readout_size
        self.h_plant0 = nn.Parameter(
            torch.zeros(1, self.plant.state_dim, device=self.device),
            requires_grad=False,
        )  

        self.disturbance=False
        self.dt = self.plant.dt
        self.disturbance = False
        self.index = None
        self.disturbance_size = 0.0
        self.observe_x_goal = observe_x_goal
        self.observe_initial_condition = observe_initial_condition

        # Construct the graph function pointer based on number of layers.
        self.layers = self.num_layers
        self.construct_graph()

    def construct_graph(self):
        # If we have more than one layer, use the multi-layer forward function.
        if self.layers >= 1:
            self.graph = self.graph_layers
        else:
            self.graph = self.graph_single

    def graph_single(self, et, ht):
        """
        Implements a one-step GRU update for a single-layer GRU.
          - et: input tensor of shape (batch_size, input_size)
          - ht: previous hidden state (batch_size, hidden_size)
        Returns:
          - new hidden state, and the same new hidden state (for consistency with graph_layers)
        """
        # Using the single GRU cell (first and only layer)
        new_ht = self.gru_cells[0](et, ht)
        # For single-layer, the concatenated hidden state is just new_ht.
        return new_ht, new_ht

    def graph_layers(self, et, ht):
        """
        Implements a one-step forward pass through the stacked GRU.
        Args:
          - et: input tensor of shape (batch_size, input_size)
          - ht: concatenated hidden state from all layers,
                    shape (batch_size, hidden_size_total)
                    (we assume that ht is a concatenation of each layer's state)
        Returns:
          - h_new_concat: concatenated new hidden states from all layers (batch_size, hidden_size_total)
          - u_t: control command produced as an MLP readout of h_new_concat
        """
        # Unpack previous hidden states for each layer from the concatenated vector.
        # Here we assume that ht was concatenated in order of layers.
        h_prev_list = []
        new_hidden_states = []
        start_idx = 0
        for h_dim in self.hidden_sizes:
            h_prev_list.append(ht[:, start_idx:start_idx+h_dim].clone())
            start_idx += h_dim

        # Layer 0: update using input et.
        h_in = torch.cat([et] + h_prev_list[1:], dim=-1)
        h_out = self.gru_cells[0](h_in, h_prev_list[0])
        # new_hidden_states.append(h_out)
        new_hidden_states.append(h_out)
        # For subsequent layers, use the hidden state from the previous layer as input.
        for layer in range(1, self.num_layers):
            h_in = torch.cat([et] + new_hidden_states[:layer] + h_prev_list[layer + 1:], dim=-1)
            h_out = self.gru_cells[layer](h_in, h_prev_list[layer])
            new_hidden_states.append(h_out)

        # Concatenate all hidden states along the feature dimension.
        h_new_concat = torch.cat(new_hidden_states, dim=1)
        # Use the MLP readout to compute the control command u_t.
        u_t = self.readout(h_new_concat) if not self.observe_output else self.readout(new_hidden_states[-1])
        return h_new_concat, u_t

    def forward(
        self, 
        x_0: torch.Tensor,
        ref: torch.Tensor,
        control_period: Optional[int] = None,
        period_upperlimit: int = 100,
        return_control_trajectory=False,
        observe = 0, 
        train = True
    ):
        """Forward loop. Controlled for a number of steps or until a certain threshold is reached."""
        assert control_period is not None
        chunk_length = 4096  # to avoid memory overflow, we divide inputs into tiny segments when it's too big
        error_trajectory = None
        x_segment = []
        target_segment = []
        time = 0
        if x_0.shape[0] > chunk_length:
            left = x_0.shape[0]
            segment_start = 0
            while left > 0:
                x_segment.append(
                    x_0[segment_start : (segment_start + min(chunk_length, left))]
                )
                target_segment.append(
                    ref[segment_start : (segment_start + min(chunk_length, left))]
                )
                left -= chunk_length
                segment_start += chunk_length
            del x_0, ref
            torch.cuda.empty_cache()
            
            for i in range(len(x_segment)):
                x_seg = x_segment.pop(0)
                target_seg = target_segment.pop(0)
                if return_control_trajectory:
                    h_trajectory_, control_trajectory_, error_trajectory_, out_trajectory_ = self.forward(
                        x_seg,
                        target_seg,
                        control_period,
                        period_upperlimit,
                        return_control_trajectory,
                    )
                    h_trajectory, control_trajectory, error_trajectory, out_trajectory = (
                        (h_trajectory_, control_trajectory_, error_trajectory_, out_trajectory_)
                        if error_trajectory is None
                        else (
                            torch.cat([h_trajectory, h_trajectory_], dim=0),
                            torch.cat([control_trajectory, control_trajectory_], dim=0),
                            torch.cat([error_trajectory, error_trajectory_], dim=0),
                            torch.cat([out_trajectory, out_trajectory_], dim=0),
                        )
                    )
                    print(h_trajectory.shape)
                else:
                    h_trajectory_, error_trajectory_ = self.forward(
                        x_seg,
                        target_seg,
                        control_period,
                        period_upperlimit,
                        return_control_trajectory,
                    )
                    h_trajectory, error_trajectory = (
                        (h_trajectory_, error_trajectory_)
                        if error_trajectory is None
                        else (
                            torch.cat([h_trajectory, h_trajectory_], dim=0),
                            torch.cat([error_trajectory, error_trajectory_], dim=0),
                        )
                    )
                    print(h_trajectory.shape)

            if return_control_trajectory:
                return h_trajectory, control_trajectory, error_trajectory, out_trajectory
            return h_trajectory, error_trajectory
            
        original_device = self.device
        B = x_0.shape[0]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

        # set up hidden states
        pos0 = x_0 / x_0.norm(p=2, dim=-1, keepdim=True)
        operator_s = pos0

        ref = ref.to(device)  # let's say we have the same ref throughout?

        if ref.dim() == 2:
            ref = ref.unsqueeze(1)         # now (B, 1, d)
        B2, N_targets, d = ref.shape
        assert B2 == B, "batch‐size mismatch"

        # --- 3) build a (B, total_T, d) ref‐train by repeating each target control_period times ---
        total_T = N_targets * control_period
        # (B, N_targets, 1, d) -> (B, N_targets, control_period, d)
        ref_rep = ref.unsqueeze(2).expand(-1, -1, control_period, -1)
        # -> (B, N_targets*control_period, d)
        ref_train = ref_rep.contiguous().view(B, total_T, d)

        # --- 4) pre-allocate trajectories ---
        device = x_0.device
        h_trajectory = torch.zeros(B, total_T, self.hidden_size_total, device=device)
        error_trajectory = torch.zeros(B, total_T, device=device)
        if return_control_trajectory:
            control_trajectory = torch.zeros(B, total_T, self.output_size, device=device)
            out_trajectory     = torch.zeros(B, total_T, self.action_size, device=device)

        if self.disturbance:
            disturbance_index = self.index if self.index is not None else np.random.randint(1, control_period)
            disturbance = torch.randn((x_0.shape[0], self.action_size), device=self.device) * self.disturbance_size
        # NOTE: out_trajectory is the true series of actions to the plant; 
        self.operator.set_state(operator_s)
        h_init = []
        for init_param in self.h0:
            # init_param has shape (1, hidden_dim); expand to (batch_size, hidden_dim)
            h_init.append(init_param.expand(x_0.shape[0], -1))
        ht = torch.cat(h_init, dim=1)  # shape: (batch_size, hidden_size_total)
        h_trajectory[:, -1, :] = ht

       
        state_feedback = self.operator.s
        h_plant = self.h_plant0.to(device).expand(x_0.shape[0], -1)
        plant_out = torch.zeros((x_0.shape[0], self.plant.dim), device = self.device)
        control_out = torch.zeros((x_0.shape[0], self.output_size), device = self.device)

        while observe > 0:
            # TODO implement observe
            observe -= 1
            if self.observe_x_goal: 
                observe_goal = ref_train[:, 0]
            elif self.observe_initial_condition:
                observe_goal = x_0
            else: 
                observe_goal = torch.zeros_like(x_0)
            h_trajectory[:, -1, :] = self.observe(h_trajectory[:, -1, :], state_feedback, observe_goal, plant_out, control_out)
 
        for time in range(total_T):
            self.operator.set_state(operator_s)
            if train:
                if self.disturbance and time == disturbance_index:
                    # integrate the *current* operator_s (B×dim), not self.operator.s
                    new_s = self.operator.integrate_joint_velocities(disturbance, operator_s)

                    # zero out the velocity channels WITHOUT in-place on new_s
                    pos = new_s[..., :self.operator.position_size]
                    vel = torch.zeros_like(new_s[..., self.operator.position_size:])
                    new_s = torch.cat([pos, vel], dim=-1)
                    #cat the new_s with the previous states

                    # update the operator’s buffer in one go
                    self.operator.set_state(new_s)
            else:
                if self.disturbance and time == disturbance_index:
                    s = self.operator.integrate_joint_velocities(disturbance, self.operator.s)
                    s[..., self.operator.position_size : ] = torch.zeros_like(s[..., self.operator.position_size : ])
                    self.operator.s[:, -1, :] = s
                    out_trajectory[:, time-1, :] += disturbance
                    # recalculate error: 
            (
                h_trajectory,
                control_out,
                out,
                control_error,
                operator_s,
                h_plant,
                plant_out, 
                state_feedback, 
            ) = self.step_controller(
                h_trajectory, ref_train[:, time, :], h_plant, control_out, plant_out, state_feedback, time
            )
            # print(ref_train[0, time, :])

            error_trajectory[:, time] = control_error
            control_trajectory[:, time] = control_out
            out_trajectory[:, time] = out

            if time > period_upperlimit:
                print(f"Control period exceeded upper bound with residual error")
                break

        self.to(original_device)
        if return_control_trajectory:
            return (
                h_trajectory,
                control_trajectory,
                error_trajectory,
                out_trajectory,
            )
        return h_trajectory, error_trajectory

    def step_controller(
        self, h_trajectory, ref, h_plant, control_out, plant_out, state_feedback, time
    ): 
        et = torch.cat([ref, control_out, plant_out, state_feedback], dim = -1).float()
        ht, control_out = self.graph(et, ht = h_trajectory[:, time - 1, :])
        h_trajectory[:, time] = ht
        h_plant, plant_out = self.plant(control_out, h_plant, self.dt)
        state_feedback, out = self.operator(plant_out)
        operator_s = self.operator.s
        control_error = (ref - operator_s[:, -1, :self.operator.position_size]).norm(p = 2, dim = -1)

        return (
                h_trajectory,
                control_out,
                out,
                control_error,
                operator_s,
                h_plant,
                plant_out, 
                state_feedback, 
            )
        
    def observe(self, ht, operator_s, ref, zero_plant_out, zero_control_out):
        # update the hidden state, with no control output. 
        # can choose the ref to be the actual x_goal or the initial condition
        et = torch.cat([ref, zero_plant_out, zero_control_out, operator_s], dim = -1).float()
        ht, _ = self.graph(et, ht = ht)
        return ht

    def introduce_disturbance(self, scale=1, index=None):
        self.disturbance = True
        self.index = index
        self.disturbance_size = scale
    
    def remove_disturbance(self):
        self.disturbance = False


    def freeze_plant(self):
        for param in self.plant.parameters():
            param.requires_grad = False
    
    def freeze_operator(self):
        for param in self.operator.parameters():
            param.requires_grad = False

    def unfreeze_operator(self):
        for param in self.operator.parameters():
            param.requires_grad = True
    
    def unfreeze_plant(self):
        for param in self.plant.parameters():
            param.requires_grad = True
 
    def freeze_controller(self):
        for param in self.gru_cells.parameters():
            param.requires_grad = False
        for param in self.h0.parameters():
            param.requires_grad = False
        for param in self.readout.parameters():
            param.requires_grad = False
        
    def unfreeze_controller(self):
        for param in self.gru_cells.parameters():
            param.requires_grad = True
        for param in self.h0.parameters():
            param.requires_grad = True
        for param in self.readout.parameters():
            param.requires_grad = True

class GRUControllerStepwise(nn.Module): 
    def __init__(
            self, 
    ):
        raise NotImplementedError

# Obsolete forward for earlier

    # def forward(
    #     self, 
    #     x_0: torch.Tensor,
    #     ref: torch.Tensor,
    #     control_period: int,
    #     period_upperlimit: int = 100,
    #     return_control_trajectory: bool = False,
    #     observe: int = 0,
    # ):
    #     """
    #     x_0: (B, d) or (B, N_targets, d)
    #     ref: same, but will be unsqueezed to (B, N_targets, d) if necessary
    #     control_period: number of timesteps *per* target
    #     """
    #     assert control_period is not None
    #     chunk_length = 4096  # to avoid memory overflow, we divide inputs into tiny segments when it's too big
    #     error_trajectory = None
    #     x_segment = []
    #     target_segment = []
    #     time = 0
    #     if x_0.shape[0] > chunk_length:
    #         left = x_0.shape[0]
    #         segment_start = 0
    #         while left > 0:
    #             x_segment.append(
    #                 x_0[segment_start : (segment_start + min(chunk_length, left))]
    #             )
    #             target_segment.append(
    #                 ref[segment_start : (segment_start + min(chunk_length, left))]
    #             )
    #             left -= chunk_length
    #             segment_start += chunk_length
    #         del x_0, ref
    #         torch.cuda.empty_cache()
            
    #         for i in range(len(x_segment)):
    #             x_seg = x_segment.pop(0)
    #             target_seg = target_segment.pop(0)
    #             if return_control_trajectory:
    #                 h_trajectory_, control_trajectory_, error_trajectory_, out_trajectory_ = self.forward(
    #                     x_seg,
    #                     target_seg,
    #                     control_period,
    #                     period_upperlimit,
    #                     return_control_trajectory,
    #                 )
    #                 h_trajectory, control_trajectory, error_trajectory, out_trajectory = (
    #                     (h_trajectory_, control_trajectory_, error_trajectory_, out_trajectory_)
    #                     if error_trajectory is None
    #                     else (
    #                         torch.cat([h_trajectory, h_trajectory_], dim=0),
    #                         torch.cat([control_trajectory, control_trajectory_], dim=0),
    #                         torch.cat([error_trajectory, error_trajectory_], dim=0),
    #                         torch.cat([out_trajectory, out_trajectory_], dim=0),
    #                     )
    #                 )
    #                 print(h_trajectory.shape)
    #             else:
    #                 h_trajectory_, error_trajectory_ = self.forward(
    #                     x_seg,
    #                     target_seg,
    #                     control_period,
    #                     period_upperlimit,
    #                     return_control_trajectory,
    #                 )
    #                 h_trajectory, error_trajectory = (
    #                     (h_trajectory_, error_trajectory_)
    #                     if error_trajectory is None
    #                     else (
    #                         torch.cat([h_trajectory, h_trajectory_], dim=0),
    #                         torch.cat([error_trajectory, error_trajectory_], dim=0),
    #                     )
    #                 )
    #                 print(h_trajectory.shape)

    #         if return_control_trajectory:
    #             return h_trajectory, control_trajectory, error_trajectory, out_trajectory
    #         return h_trajectory, error_trajectory
            
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     self.to(device)


    #     B = x_0.shape[0]

    #     # --- 1) normalize x_0, build operator initial state ---
    #     pos0 = x_0 / x_0.norm(p=2, dim=-1, keepdim=True)

    #     self.operator.set_state(pos0)      # only once, at t=0
    #     operator_s = self.operator.s

    #     # --- 2) massage ref into shape (B, N_targets, d) ---
    #     if ref.dim() == 2:
    #         ref = ref.unsqueeze(1)         # now (B, 1, d)
    #     B2, N_targets, d = ref.shape
    #     assert B2 == B, "batch‐size mismatch"

    #     # --- 3) build a (B, total_T, d) ref‐train by repeating each target control_period times ---
    #     total_T = N_targets * control_period
    #     # (B, N_targets, 1, d) -> (B, N_targets, control_period, d)
    #     ref_rep = ref.unsqueeze(2).expand(-1, -1, control_period, -1)
    #     # -> (B, N_targets*control_period, d)
    #     ref_train = ref_rep.contiguous().view(B, total_T, d)

    #     # --- 4) pre-allocate trajectories ---
    #     device = x_0.device
    #     h_trajectory = torch.zeros(B, total_T, self.hidden_size_total, device=device)
    #     error_trajectory = torch.zeros(B, total_T, device=device)
    #     if return_control_trajectory:
    #         control_trajectory = torch.zeros(B, total_T, self.output_size, device=device)
    #         out_trajectory     = torch.zeros(B, total_T, self.action_size, device=device)

    #     # initialize your concatenated hidden‐state h0 (as before):
    #     h_init = [p.expand(B, -1) for p in self.h0]
    #     ht     = torch.cat(h_init, dim=1)        # (B, hidden_size_total)
    #     h_trajectory[:, -1, :] = ht

    #     # plant and feedback states:
    #     h_plant      = self.h_plant0.to(device).expand(B, -1)
    #     plant_out    = torch.zeros(B, self.plant.dim, device=device)
    #     state_feedback = self.operator.s        # after that initial set_state

    #     # (optional) observation warm-up (unchanged):
    #     while observe > 0:
    #         # your existing observe logic here,
    #         # writing into h_trajectory[:, 0, :]
    #         observe -= 1
    #         if self.observe_x_goal: 
    #             observe_goal = ref[:, 0]
    #         elif self.observe_initial_condition:
    #             observe_goal = x_0
    #         else: 
    #             observe_goal = torch.zeros_like(x_0)
    #         h_trajectory[:, -1, :] = self.observe(h_trajectory[:, -1, :], state_feedback, observe_goal, plant_out)
 
    #     # --- 5) main loop over ALL timesteps ---
 
    #     for time in range(control_period):
    #         self.operator.set_state(operator_s)

    #         if self.disturbance and time == disturbance_index:
    #             s = self.operator.integrate_joint_velocities(disturbance, self.operator.s)
    #             s[..., self.operator.position_size : ] = torch.zeros_like(s[..., self.operator.position_size : ])
    #             self.operator.s[:, -1, :] = s

    #         (
    #             h_trajectory,
    #             control_out,
    #             out,
    #             control_error,
    #             operator_s,
    #             h_plant,
    #             plant_out, 
    #             state_feedback, 
    #         ) = self.step_controller(
    #             h_trajectory, ref_train[:, t, :], h_plant, plant_out, state_feedback, time
    #         )

    #         error_trajectory[:, time] = control_error
    #         control_trajectory[:, time] = control_out
    #         out_trajectory[:, time] = out

    #         if time > period_upperlimit:
    #             print(f"Control period exceeded upper bound with residual error")
    #             break


    #     for t in range(1, total_T):

    #         self.operator.set_state(operator_s)
    #         cur_ref = ref_train[:, t, :]         # (B, d)

    #         # 5a) step controller
    #         ht, control_out = self.graph(
    #             torch.cat([cur_ref, plant_out, state_feedback], dim=-1), 
    #             ht
    #         )
    #         h_trajectory[:, t, :] = ht

    #         # 5b) plant + operator as before
    #         h_plant, plant_out     = self.plant(control_out, h_plant, self.dt)
    #         state_feedback, out    = self.operator(plant_out)

    #         # 5c) error
    #         error_trajectory[:, t] = (cur_ref - state_feedback[:, -self.operator.position_size:]).norm(dim=-1)

    #         # 5d) book-keeping for optional returns
    #         if return_control_trajectory:
    #             control_trajectory[:, t, :] = control_out
    #             out_trajectory[:, t, :]     = out

    #         # safety
    #         if t > period_upperlimit * N_targets:
    #             print("Exceeded upper limit; cutting off")
    #             break


    #     # --- 6) return exactly as before ---
    #     if return_control_trajectory:
    #         return h_trajectory, control_trajectory, error_trajectory, out_trajectory
    #     else:
    #         return h_trajectory, error_trajectory
