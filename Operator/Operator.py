"""
This module is involved with creating an operator that interacts with motor outputs. 

incoming messages may come in as a motor activation vector.

operator readout matrix parses it through the "environment"
1. reads the control signals from the large vector thing (let's say we consult the controllability gramian a bit to optimize this)
- for example, reads out joint velocities from the motor activation matrix.

- 
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from typing import Mapping, Optional
from utils.path_settings import * 
from copy import deepcopy
from scipy.linalg import expm
from Network_models.Activations import * 
from Rotations import *

import time

class Operator(nn.Module):
    """This module implements the thing we operate on
    - we always assume that the first few elements of the sensory feedback is the position
        - and position is special! the thing is to align with the CAN for the corrective loss
        - we always define the position_size to confine this  
    """
    def __init__(self, configs):
        super(Operator, self).__init__()
        self.configs = configs
        self.device = configs.get('device', "cpu")
        self.state_size = configs.get('state_size', None)
        self.action_size = configs.get('action_size', None)
        self.dt = configs.get('dt', 0.01)
        self.s0 = None
        self.s = None
        self.enforce_joint_limits = configs.get('enforce_joint_limits', False)
        self.enforce_muscle_saturation = configs.get('enforce_muscle_saturation', False)
        self.enforce_velocity_limits = configs.get('enforce_velocity_limits', False)
        self.actuation_noise = configs.get('actuation_noise', 0.0)
        self.feedback_noise = configs.get('feedback_noise', 0.0)
        
        # Define readout matrix for extracting meaningful signals from motor activations
        # self.readout_matrix = nn.Linear(self.action_size, self.state_size, bias=False)
        
        # Define muscle activation constraints
        self.activation_min = configs.get('activation_min', 0.0)
        self.activation_max = configs.get('activation_max', 1.0)
        
        # Define joint limit constraints
        self.joint_min = configs.get('joint_min', -1.0)
        self.joint_max = configs.get('joint_max', 1.0)
    
    def forward(self, action, s0=None):
        """Processes incoming action to update state
        Args:
            action (torch.Tensor(batch_size, seq_len, action_dim)): Motor activation vector
            s0 (torch.Tensor(state_dim)): Initial state
        """
        
        action = action.to(self.device)
        if s0 is not None:
            self.s = s0
        elif self.s is not None:
            pass
        else: 
            self.s = self.s0 
        if self.s is None:
            raise ValueError("Initial state not set.")
        
        if self.s.dim() == 1:
            self.s = self.s.expand(action.shape[0], -1) # expand initial state to batch size
            self.s = self.s.unsqueeze(1).to(self.device)
        elif self.s.dim() == 2:
            self.s = self.s.unsqueeze(1).to(self.device)
        if action.dim() == 2: 
            return self.forward_single(action)

        feedback_list = []
        out_list = []
        for i in range(action.shape[1]):
            s, out = self.step(action[:, i], self.s)
            self.s = torch.cat([self.s, s.unsqueeze(1)], dim = 1)

            feedback_list.append(self.feedback(self.s))
            out_list.append(out)

        return torch.stack(feedback_list, dim=1) , torch.stack(out_list, dim=1)
    
    def forward_single(self, action):
        action = action.to(self.device)
        if self.s is None:
            raise ValueError("Initial state not set.")
        s, out = self.step(action, self.s)
        self.s = torch.cat([self.s, s.unsqueeze(1)], dim = 1)
        return self.feedback(self.s), out
        

    def step(self, action, s=None):
        """Applies constraints and updates state."""
        if s is None:
            s = self.s if self.s is not None else self.s0
        
        # Enforce hard constraint on muscle activation
        # NOTE: enforce before OR after the readout matrix 
        if self.enforce_muscle_saturation:
            action = self._enforce_muscle_saturation(action)
        
        # Readout joint velocities from activation

        joint_velocities = self.readout_matrix(action) * 0.1
        if self.enforce_velocity_limits:
            joint_velocities = self._enforce_velocity_limits(joint_velocities)

        joint_velocities = joint_velocities + torch.randn_like(joint_velocities).to(joint_velocities.device) * self.actuation_noise
        
        new_state = self.integrate_joint_velocities(joint_velocities, s)
        # Enforce joint limits
        if self.enforce_joint_limits:
            new_state = self._enforce_joint_limits(new_state)
        return new_state, joint_velocities
    
    def feedback(self, s=None):
        """Feedback mechanism for the operator."""
        return self.s[:, -1] + torch.randn_like(self.s[:, -1]) * self.feedback_noise
    
    def reset(self):
        self.s = self.s0
    
    def set_state(self, s):
        if s.shape[-1] < self.state_size:
            append_zeros = torch.zeros(*s.shape[:-1], self.state_size - s.shape[-1]).to(s.device)
            s = torch.cat([s, append_zeros], dim = -1)
        self.s = s
    
    def set_null_states(self, batch):
        """Sets the state to zeros."""
        assert self.s0 is not None, "Null state not defined."
        self.s = self.s0.expand(batch, -1).to(self.device).unsqueeze(1)
    
    def _enforce_muscle_saturation(self, action):
        """Ensures activations stay within biological limits."""
        return torch.tanh(action)
    
    def _enforce_velocity_limits(self, velocities):
        """Ensures velocities remain within feasible limits."""
        return torch.tanh(velocities) 

    
    def _enforce_joint_limits(self, state):
        """Ensures state remains within feasible joint limits."""
        return torch.clamp(state, min=self.joint_min, max=self.joint_max)


class Joint(Operator): 
    """This module is the simplest joint operator.
    control by velocity or acceleration; 
    feedback by position, velocity, acceleration
    position is special! this modality is to be aligned with the CAN for the corrective loss
    """
    def __init__(self, configs):
        super(Joint, self).__init__(configs)

        self.state_size = configs.get('state_size', 10) # 4D position, 3D velocity, 3D acceleration
        self.action_size = configs.get('action_size', 3)
        self.plant_state_size = configs.get('plant_state_size', 128)
        self.readout_matrix = nn.Linear(self.plant_state_size, self.action_size, bias=False).to(self.device)
        self.action_modality = configs.get('action_modality', 'velocity')
        if self.state_size == 10: 
            self.s0 = torch.cat([torch.zeros(self.state_size)])
            self.s0[0] = 1.0 # real part of the quaternion to 1

        else: 
            raise ValueError("Invalid state size. Currently only supports 3x accel + 3x vel + 4x pos")
        self.delay = configs.get('delay', 0)
        self.feedback_modalities = configs.get('feedback_modalities', ['velocity', 'acceleration'])
        self.position_size = 4

        
    def feedback(self, s=None):
        """
        Takes the s trajectory to return the feedback at the current time step
        """
        if s is None:
            s = self.s if self.s is not None else self.s0

        returned = []
        returned.append(s[:, max(0, s.shape[1]-1-self.delay), 0:4]) # position
        return_dim = [[4, 7], [7, 10]]
        for i, modality in enumerate(['velocity', 'acceleration']):
            if modality in self.feedback_modalities:
                returned.append(s[:, -1, return_dim[i][0]:return_dim[i][1]])
            
        returned = torch.cat(returned, dim = -1) 
        return returned + torch.randn_like(returned) * self.feedback_noise

    def integrate_joint_velocities(self, joint_velocities, s=None): 
        if s is None:
            s = self.s if self.s is not None else self.s0

        if self.action_modality == 'velocity':
            s = self.integrate_joint_velocities_velocity(joint_velocities, s)
        elif self.action_modality == 'acceleration':
            s = self.integrate_joint_velocities_acceleration(joint_velocities, s)
        else: 
            raise ValueError("Invalid action modality. Currently only supports 'velocity' or 'acceleration'")

        return s

    def integrate_joint_velocities_velocity(self, joint_velocities, s=None):
        if s is None:
            s = self.s if self.s is not None else self.s0

        pos = s[:, -1, 0:4]
        qt = exp_quat(joint_velocities * self.dt)
        pos_new = q_mult(qt.double(), pos.double(), scalar_first=True).float()
        pos_new = pos_new/ pos_new.norm(dim = -1, keepdim = True)
        vel = joint_velocities
        accel = (vel - s[:, -1, 4:7]) / self.dt
        return torch.cat([pos_new, vel, accel], dim = -1)
    
    def integrate_joint_velocities_acceleration(self, joint_velocities, s=None):
        if s is None:
            s = self.s if self.s is not None else self.s0

        pos = s[:, -1, 0:4]
        vel = s[:, -1, 4:7]
        accel = joint_velocities
        vel_new = vel + accel * self.dt
        qt = exp_quat(vel_new * self.dt)
        pos_new = q_mult(qt.double(), pos.double(), scalar_first=True).float()
        pos_new = pos_new/ pos_new.norm(dim = -1, keepdim = True)
        return torch.cat([pos_new, vel_new, accel], dim = -1)


class Ring(Operator):
    """This module implements a ring operator
    - Control via angular velocity or angular acceleration
    - Feedback includes position (x, y), angular velocity, and angular acceleration
    """
    def __init__(self, configs):
        super(Ring, self).__init__(configs)
        
        self.state_size = configs.get('state_size', 4)  # (x, y, angular velocity, angular acceleration)
        self.action_size = configs.get('action_size', 1)  # Control is angular acceleration
        self.plant_state_size = configs.get('plant_state_size', 128)
        self.readout_matrix = nn.Linear(self.plant_state_size, self.action_size, bias=False).to(self.device)
        self.action_modality = configs.get('action_modality', 'acceleration')
        self.delay = configs.get('delay', 0)
        
        # Initial state: x=1, y=0 (theta=0), zero velocity, zero acceleration
        self.s0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.s = self.s0.unsqueeze(0)
        
        self.feedback_modalities = configs.get('feedback_modalities', ['velocity', 'acceleration'])
        self.position_size = 2  # x, y
    
    def feedback(self, s=None):
        """Extracts feedback for the current state."""
        if s is None:
            s = self.s if self.s is not None else self.s0.unsqueeze(0)
    
        returned_index = s.shape[1] - 1 - self.delay
        if returned_index < 0:
            returned = [torch.zeros_like(s[:, 0, :2])]
            if 'velocity' in self.feedback_modalities:
                returned.append(torch.zeros_like(s[:, 0, 2:3]))
            if 'acceleration' in self.feedback_modalities:
                returned.append(torch.zeros_like(s[:, 0, 3:4]))
        else: 
            returned = [s[:, returned_index, :2]]  # x, y position
            if 'velocity' in self.feedback_modalities:
                returned.append(s[:, returned_index, 2:3])  # angular velocity
            if 'acceleration' in self.feedback_modalities:
                returned.append(s[:, returned_index, 3:4])  # angular acceleration
        
        returned = torch.cat(returned, dim=-1)
        # print(returned)
        return returned + torch.randn_like(returned) * self.feedback_noise
    
    def integrate_joint_velocities(self, joint_velocities, s=None):
        if s is None:
            s = self.s if self.s is not None else self.s0.unsqueeze(0)
        
        if self.action_modality == 'velocity':
            return self.integrate_velocity(joint_velocities, s)
        elif self.action_modality == 'acceleration':
            return self.integrate_acceleration(joint_velocities, s)
        else:
            raise ValueError("Invalid action modality. Supports 'velocity' or 'acceleration'.")
    
    def integrate_velocity(self, joint_velocities, s=None):
        """Updates position using angular velocity."""
        if s is None:
            s = self.s if self.s is not None else self.s0.unsqueeze(0)
        
        x, y = s[:, -1, 0], s[:, -1, 1]
        theta = torch.atan2(y, x)
        
        angular_velocity = joint_velocities.squeeze()
        theta_new = theta + angular_velocity * self.dt
        
        x_new, y_new = torch.cos(theta_new), torch.sin(theta_new)
        accel = (angular_velocity - s[:, -1, 2]) / self.dt
        
        return torch.stack([x_new, y_new, angular_velocity, accel], dim=-1)
    
    def integrate_acceleration(self, joint_velocities, s=None):
        """Updates position using angular acceleration."""
        if s is None:
            s = self.s if self.s is not None else self.s0.unsqueeze(0)
        
        x, y = s[:, -1, 0], s[:, -1, 1]
        theta = torch.atan2(y, x)
        
        angular_velocity = s[:, -1, 2] + joint_velocities.squeeze() * self.dt
        theta_new = theta + angular_velocity * self.dt
        
        x_new, y_new = torch.cos(theta_new), torch.sin(theta_new)
        accel = joint_velocities.squeeze()
        
        return torch.stack([x_new, y_new, angular_velocity, accel], dim=-1)


class Wheel(Operator):
    """
    A 1D wheel operator that extends the `Operator` base class, 
    but stores position as x,y (like a wheel handle on a 2D plane).
    
    State dimension: (x, y, angular_velocity, angular_acceleration)
    Action dimension: typically 1 (torque or angular acceleration).
    """

    def __init__(self, configs):
        super(Wheel, self).__init__(configs)

        # We want the state: x, y, angular_vel, angular_acc
        self.state_size = configs.get('state_size', 4)
        self.action_size = configs.get('action_size', 1)
        self.plant_state_size = configs.get('plant_state_size', 128)
        self.readout_matrix = nn.Linear(self.plant_state_size, self.action_size, bias=False).to(self.device)

        # angle update mode: either 'velocity' or 'acceleration'
        self.action_modality = configs.get('action_modality', 'acceleration')
        self.delay = configs.get('delay', 0)

        # Initial state: x=1, y=0 => angle=0, velocity=0, acceleration=0
        self.s0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self.s = self.s0.unsqueeze(0)

        self.feedback_modalities = configs.get('feedback_modalities', ['velocity', 'acceleration'])
        self.position_size = 2  # x,y

    def feedback(self, s=None):
        """
        Return the current or delayed state variables as feedback:
            - x, y
            - [velocity]
            - [acceleration]
        """
        if s is None:
            s = self.s if self.s is not None else self.s0.unsqueeze(0)

        returned_index = s.shape[1] - 1 - self.delay
        if returned_index < 0:
            # no previous states yet -> return zeros
            batch_size = s.shape[0]
            zero_feedback = []
            zero_feedback.append(torch.zeros(batch_size, 2, device=self.device))
            if 'velocity' in self.feedback_modalities:
                zero_feedback.append(torch.zeros(batch_size, 1, device=self.device))
            if 'acceleration' in self.feedback_modalities:
                zero_feedback.append(torch.zeros(batch_size, 1, device=self.device))
            returned = torch.cat(zero_feedback, dim=-1)
        else:
            # x,y
            returned_list = [s[:, returned_index, :2]]
            # velocity => index 2
            if 'velocity' in self.feedback_modalities:
                returned_list.append(s[:, returned_index, 2:3])
            # acceleration => index 3
            if 'acceleration' in self.feedback_modalities:
                returned_list.append(s[:, returned_index, 3:4])
            returned = torch.cat(returned_list, dim=-1)

        # Add noise, if configured
        return returned + torch.randn_like(returned) * self.feedback_noise

    def integrate_joint_velocities(self, torque_or_acc, s=None):
        """
        Given torque or angular acceleration, 
        compute new (x, y, angular_velocity, angular_acceleration).
        """
        if s is None:
            s = self.s if self.s is not None else self.s0.unsqueeze(0)

        if self.action_modality == 'velocity':
            return self.integrate_velocity(torque_or_acc, s)
        elif self.action_modality == 'acceleration':
            return self.integrate_acceleration(torque_or_acc, s)
        else:
            raise ValueError("Invalid action modality: choose 'velocity' or 'acceleration'")

    def integrate_velocity(self, torque_or_vel, s=None):
        """
        Interprets incoming action as angular velocity => update angle in (x, y).
        Then derive acceleration from the difference in velocity.
        """
        if s is None:
            s = self.s if self.s is not None else self.s0.unsqueeze(0)

        prev_x, prev_y = s[:, -1, 0] + 1e-6, s[:, -1, 1]
        prev_vel = s[:, -1, 2]
        if (prev_x**2 + prev_y**2).min() == 0:
            print("Zero-norm state detected!", prev_x, prev_y) 
        # compute old angle
        theta = torch.atan2(prev_y, prev_x)
        angular_velocity = torque_or_vel.reshape(-1)

        theta_new = theta + angular_velocity * self.dt

        # convert angle to x,y
        x_new = torch.cos(theta_new)
        y_new = torch.sin(theta_new)

        # derive acceleration
        acceleration = (angular_velocity - prev_vel) / self.dt

        return torch.stack([x_new, y_new, angular_velocity, acceleration], dim=-1)

    def integrate_acceleration(self, torque_or_acc, s=None):
        """
        Interprets incoming action as angular acceleration => update velocity and angle in x,y.
        """
        if s is None:
            s = self.s if self.s is not None else self.s0.unsqueeze(0)

        prev_x, prev_y = s[:, -1, 0], s[:, -1, 1]
        prev_vel = s[:, -1, 2]

        # old angle
        theta = torch.atan2(prev_y + 1e-6, prev_x + 1e-6)

        # update velocity
        gamma = self.configs.get("velocity_decay", 0.1)  # default decay

        # update velocity with decay
        # print(torque_or_acc)
        angular_velocity = (1 - gamma * self.dt) * prev_vel + torque_or_acc.squeeze() * self.dt
        theta_new = theta + angular_velocity * self.dt

        x_new = torch.cos(theta_new)
        y_new = torch.sin(theta_new)
        acceleration = torque_or_acc.reshape(-1)

        return torch.stack([x_new, y_new, angular_velocity, acceleration], dim=-1)

class Fly(Operator):
    """
    This is simple fly walking model. Briefly, we assert there is structure in in the input population of a concatenated (lLAL and rLAL), 
    each moderating forward velocity in the left and right legs. We convert these forward velocities into angular state space. 
    
    State dimension: (x, y, angular_velocity, angular_acceleration)
    Action dimension: typically 1 (torque or angular acceleration).
    """

    def __init__(self, configs):
        super(Fly, self).__init__(configs)

        # We want the state: x, y, angular_vel, angular_acc
        self.state_size = configs.get('state_size', 4)
        self.action_size = configs.get('action_size', 1)
        self.plant_state_size = configs.get('plant_state_size', 128) # 2 x LAL size
        self.readout_matrix = nn.Linear(self.plant_state_size, self.action_size, bias=False).to(self.device)
        readout = abs(torch.randn(self.plant_state_size//2, self.action_size)) / np.sqrt(self.plant_state_size)
        # copy this into the weights
        self.readout_matrix.weight.data = torch.cat([readout, -readout], dim = 0).T.to(self.device)
        self.width = configs.get("width", 1) # width of the fly

        # angle update mode: either 'velocity' or 'acceleration'
        self.action_modality = configs.get('action_modality', 'acceleration')
        assert self.action_modality in ['velocity'] , "Invalid action modality. Currently only supports 'velocity'"

        self.delay = configs.get('delay', 0)

        # Initial state: x=1, y=0 => angle=0, velocity=0, acceleration=0
        self.s0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self.s = self.s0.unsqueeze(0)

        self.feedback_modalities = configs.get('feedback_modalities', ['velocity', 'acceleration'])
        self.position_size = 2  # x,y

    def feedback(self, s=None):
        """
        Return the current or delayed state variables as feedback:
            - x, y
            - [velocity]
            - [acceleration]
        """
        if s is None:
            s = self.s if self.s is not None else self.s0.unsqueeze(0)

        returned_index = s.shape[1] - 1 - self.delay
        if returned_index < 0:
            # no previous states yet -> return zeros
            batch_size = s.shape[0]
            zero_feedback = []
            zero_feedback.append(torch.zeros(batch_size, 2, device=self.device))
            if 'velocity' in self.feedback_modalities:
                zero_feedback.append(torch.zeros(batch_size, 1, device=self.device))
            if 'acceleration' in self.feedback_modalities:
                zero_feedback.append(torch.zeros(batch_size, 1, device=self.device))
            returned = torch.cat(zero_feedback, dim=-1)
        else:
            # x,y
            returned_list = [s[:, returned_index, :2]]
            # velocity => index 2
            if 'velocity' in self.feedback_modalities:
                returned_list.append(s[:, returned_index, 2:3])
            # acceleration => index 3
            if 'acceleration' in self.feedback_modalities:
                returned_list.append(s[:, returned_index, 3:4])
            returned = torch.cat(returned_list, dim=-1)

        # Add noise, if configured
        return returned + torch.randn_like(returned) * self.feedback_noise

    def integrate_joint_velocities(self, torque_or_acc, s=None):
        """
        Given torque or angular acceleration, 
        compute new (x, y, angular_velocity, angular_acceleration).
        """
        if s is None:
            s = self.s if self.s is not None else self.s0.unsqueeze(0)

        if self.action_modality == 'velocity':
            return self.integrate_velocity(torque_or_acc, s)
        elif self.action_modality == 'acceleration':
            return self.integrate_acceleration(torque_or_acc, s)
        else:
            raise ValueError("Invalid action modality: choose 'velocity' or 'acceleration'")

    def integrate_velocity(self, velocity_diff, s=None):
        """
        Interprets incoming action as velocity difference => update angle in (x, y).
        Then derive acceleration from the difference in velocity.
        """
        if s is None:
            s = self.s if self.s is not None else self.s0.unsqueeze(0)

        prev_x, prev_y = s[:, -1, 0], s[:, -1, 1]
        prev_vel = s[:, -1, 2]

        # compute old angle
        theta = torch.atan2(prev_y, prev_x)
        # print(velocity_diff*self.dt / self.width)
        shift = torch.clamp(velocity_diff.reshape(-1) * self.dt / self.width, -1 + 1e-6, 1 - 1e-6)
        
        angular_displacement = torch.arcsin(shift)

        theta_new = theta + angular_displacement

        # convert angle to x,y
        x_new = torch.cos(theta_new)
        y_new = torch.sin(theta_new)

        # derive acceleration
        acceleration = (angular_displacement / self.dt - prev_vel) 

        return torch.stack([x_new, y_new, angular_displacement / self.dt, acceleration], dim=-1)

    def integrate_acceleration(self, torque_or_acc, s=None):
        raise NotImplementedError("Integrate acceleration not implemented for Fly operator.")