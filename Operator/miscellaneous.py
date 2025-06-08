import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MusclePlant(nn.Module):
    """
    A simplified muscle interface:
      - Takes neural inputs (e.g., from a neural network or controller).
      - Produces muscle activation using first-order dynamics.
      - Maps muscle activation to torque (or force for a 1D handle).
    """

    def __init__(
        self,
        dt=0.01,
        alpha=10.0,        # rate of muscle activation dynamics
        max_activation=1.0,
        K_torque=5.0,      # maps activation to torque
        muscle_damping=0.1, # minor damping term to illustrate 'biological joint' effect
        muscle_number=1,
    ):
        """
        Args:
            dt: time step for forward() calls
            alpha: speed of activation (large => faster approach to desired activation)
            max_activation: clamp muscle activation between [0, max_activation]
            K_torque: scalar turning muscle activation into torque
            muscle_damping: small damping factor to reduce torque at high muscle velocities
        """
        super().__init__()
        self.dt = dt
        self.alpha = alpha
        self.max_activation = max_activation
        self.K_torque = K_torque
        self.muscle_damping = muscle_damping

        # Store muscle activation state. 
        # If you have multiple muscles, you can store more dimensions here.
        self.activation = torch.zeros(muscle_number)

        # Optionally track muscle velocity if you want more complex dynamics, e.g., Hill-based model.
        self.muscle_vel = torch.zeros(muscle_number)
        self.state_dim = muscle_number

    def forward(self, neural_input, activation=None, dt=None):
        """
        Args:
            neural_input (Tensor): neural command (scalar or shape [..., 1])
            activation (Tensor): optional override of the current muscle activation
            dt (float): optional override of time step
        Returns:
            next_activation (Tensor)
            torque (Tensor): muscle torque output
        """
        update_self_activation = False
        if activation is None:
            activation = self.activation
            update_self_activation = True
        if dt is None:
            dt = self.dt
        
        # Convert neural command -> target activation (clamped)
        desired_activation = torch.nn.functional.sigmoid(neural_input) * self.max_activation

        # Activation dynamics: da/dt = alpha * (desired - a)
        d_activation = self.alpha * (desired_activation - activation)
        next_activation = activation + d_activation * dt

        # Compute torque. Introduce a small damping factor that depends on current activation velocity.
        # This is a toy example of 'biological joint' friction or muscle damping.
        activation_velocity = d_activation
        damping_factor = 1.0 - self.muscle_damping * activation_velocity
        torque = self.K_torque * next_activation * damping_factor

        if update_self_activation:
            self.activation = next_activation.detach()

        return next_activation, torque

class DecayPlant(nn.Module): 
    def __init__(self, dim, dim_input=None, dt=0.01, constant = 0.9, K_torque=1): 
        super(DecayPlant, self).__init__()
        self.dt = dt
        self.A = nn.Linear(dim, dim, False)
        self.dim_input = dim_input if dim_input is not None else dim
        self.B = nn.Linear(dim_input, dim, False) 
        with torch.no_grad():
            # self.A.weight.copy_(torch.diag(-torch.rand(dim,)).cuda())
            self.A.weight.copy_(-constant * torch.eye(dim).cuda())
            # self.B.weight.copy_(torch.rand(dim, self.dim_input).cuda())
        for param in self.parameters(): 
            param.requires_grad = False
        self.dim = dim
        self.state_dim = dim
        self.K_torque = K_torque

        self.h = torch.zeros(1, dim).cuda()
    def forward(self, u, h=None, dt=None): 
        update_self_h = False
        if h is None: 
            h = self.h
            update_self_h = True
        if dt is None:
            dt = self.dt
        h_new = h + self.A(h) * dt + self.B(u) * dt * self.K_torque # Euler discretization - fine for now!
        if update_self_h: 
            self.h = h_new.detach()
        return h_new, h_new

class SingleLayerLM(nn.Module): 
    def __init__(self, input_size, output_size, recurrent_size=0, activation="tanh", zero_initialization=False): 
        super(SingleLayerLM, self).__init__()
        
        self.recurrent_size = recurrent_size
        self.input_size = input_size
        self.output_size = output_size
        self.activation = getattr(torch, activation) if activation != "linear" else lambda x: x

        if recurrent_size > 0:
            self.W_ih = nn.Linear(input_size, recurrent_size, True)  # Input-to-hidden
            self.W_hh = nn.Linear(recurrent_size, recurrent_size, True)  # Hidden-to-hidden
            self.W_ho = nn.Linear(recurrent_size, output_size, True)  # Hidden-to-output
            self.h0 = nn.Parameter(torch.zeros(1, recurrent_size))  # Initial hidden state
        else:
            self.W_ih = None
            self.W_hh = None
            self.W_ho = None
            self.h0 = None
        self.W_io = nn.Linear(input_size, output_size, True)  # Input-to-output direct mapping
        if zero_initialization:
            with torch.no_grad():
                zero_like = torch.zeros_like(self.W_io.weight)
                print(zero_like.shape)
                zero_like[0, 2] = 1.
                self.W_io.weight.copy_(zero_like)
                self.W_io.bias.copy_(torch.zeros_like(self.W_io.bias))
                if recurrent_size > 0:
                    self.W_ih.weight.copy_(torch.zeros_like(self.W_ih.weight))
                    self.W_hh.weight.copy_(torch.zeros_like(self.W_hh.weight))
                    self.W_ho.weight.copy_(torch.zeros_like(self.W_ho.weight))
                    self.W_ih.bias.copy_(torch.zeros_like(self.W_ih.bias))
                    self.W_hh.bias.copy_(torch.zeros_like(self.W_hh.bias))
                    self.W_ho.bias.copy_(torch.zeros_like(self.W_ho.bias))
            

    def forward(self, x, h=None):
        if self.recurrent_size > 0:
            if h is None:
                h = self.h0.expand(x.shape[0], -1) 
            h = self.activation(self.W_ih(x) + self.W_hh(h))  
            y = self.W_io(x) + self.W_ho(h)  # Output depends on both input and hidden state
        else:
            y = self.W_io(x)  # Simple feedforward case
        y = self.activation(y)
        
        return h, y

class MLP(nn.Module): 
    def __init__(self, input_size, hidden_size, output_size, recurrent_size=0, activation="relu", output_activation="tanh", zero_initialization=False, normalize=False): 
        super(MLP, self).__init__()
        
        self.recurrent_size = recurrent_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = getattr(torch, activation) if activation != "linear" else lambda x: x
        self.output_activation = getattr(torch, output_activation) if output_activation != "linear" else lambda x: x
        self.normalize=normalize

        bias = True

        if recurrent_size > 0:
            # Recurrence at the hidden layer level
            self.W_ih = nn.Linear(input_size, recurrent_size, bias)  # Input-to-hidden
            self.W_hh = nn.Linear(recurrent_size, recurrent_size, bias)  # Hidden-to-hidden recurrence
            self.W_rh = nn.Linear(recurrent_size, hidden_size, bias)  # Recurrent hidden-to-hidden layer
            self.h0 = nn.Parameter(torch.zeros(1, recurrent_size))  # Initial hidden state
        else:
            self.W_ih = None
            self.W_hh = None
            self.W_rh = None
            self.h0 = None
        
        # Standard MLP layself.opself.operator.s[:, -2, :self.operator.position_size]erator.s[self.operator.s[:, -2, :self.operator.position_size]:, -2, :self.operator.position_size]ers
        self.W_hidden = nn.Linear(input_size, hidden_size, bias)  # Input-to-hidden direct mapping
        self.W_output = nn.Linear(hidden_size, output_size, bias)  # Hidden-to-output mapping
        
        if recurrent_size > 0:
            self.W_ro = nn.Linear(recurrent_size, output_size, bias)  # Additional recurrent-to-output mapping

        if zero_initialization:
            with torch.no_grad():
                self.W_hidden.weight.copy_(torch.zeros_like(self.W_hidden.weight))
                self.W_hidden.bias.copy_(torch.zeros_like(self.W_hidden.bias))
                self.W_output.weight.copy_(torch.zeros_like(self.W_output.weight))
                self.W_output.bias.copy_(torch.zeros_like(self.W_output.bias))
                
                if recurrent_size > 0:
                    self.W_ih.weight.copy_(torch.zeros_like(self.W_ih.weight))
                    self.W_hh.weight.copy_(torch.zeros_like(self.W_hh.weight))
                    self.W_rh.weight.copy_(torch.zeros_like(self.W_rh.weight))
                    self.W_ro.weight.copy_(torch.zeros_like(self.W_ro.weight))
                    self.W_ih.bias.copy_(torch.zeros_like(self.W_ih.bias))
                    self.W_hh.bias.copy_(torch.zeros_like(self.W_hh.bias))
                    self.W_rh.bias.copy_(torch.zeros_like(self.W_rh.bias))
                    self.W_ro.bias.copy_(torch.zeros_like(self.W_ro.bias))

    def forward(self, x, h=None):
        if self.recurrent_size > 0:
            if h is None:
                h = self.h0.expand(x.shape[0], -1)  
            h = self.activation(self.W_ih(x) + self.W_hh(h))  # Recurrent state update
            hidden = self.activation(self.W_hidden(x) + self.W_rh(h))  # Combine recurrent state with input
            y = self.W_output(hidden) + self.W_ro(h)  # Output depends on both hidden state and recurrent state
        else:
            hidden = self.activation(self.W_hidden(x))  # Standard feedforward hidden layer
            y = self.W_output(hidden)  # Standard feedforward output layer

        y = self.output_activation(y)
        if self.normalize:
            y = y/torch.norm(y, dim=-1, keepdim=True)
        return h, y

class InternalModelMLP(nn.Module):
    def __init__(
        self,
        input_sizes,  # dict: {'efference': d1, 'sensory': d2, 'can': d3}
        hidden_size,  # int or list of ints for multilayer
        output_size,
        recurrent_size=0,
        activation="relu",
        output_activation="tanh",
        zero_initialization=False,
        normalize=False,
    ):
        super().__init__()
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]

        self.input_sizes = input_sizes
        self.hidden_sizes = hidden_size
        self.output_size = output_size
        self.recurrent_size = recurrent_size
        self.activation = getattr(torch, activation) if activation != "linear" else lambda x: x
        self.output_activation = getattr(torch, output_activation) if output_activation != "linear" else lambda x: x
        self.normalize = normalize

        self.input_streams = nn.ModuleDict()
        for key, dim in input_sizes.items():
            self.input_streams[key] = nn.Linear(dim, hidden_size[0])
            

        # Hidden layers (deep feedforward MLP)
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_size) - 1):
            self.hidden_layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))

        self.W_output = nn.Linear(hidden_size[-1], output_size)

        if recurrent_size > 0:
            raise NotImplementedError("Recurrent connections not implemented in this version.")
            self.W_hh = nn.Linear(recurrent_size, recurrent_size)
            self.W_rh = nn.Linear(recurrent_size, hidden_size[0])
            self.W_ro = nn.Linear(recurrent_size, output_size)
            self.h0 = nn.Parameter(torch.zeros(1, recurrent_size))

        if zero_initialization:
            for lin in self.input_streams.values():
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
            for lin in self.hidden_layers:
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
            nn.init.zeros_(self.W_output.weight)
            nn.init.zeros_(self.W_output.bias)

        self.silence_sensory = False
        self.silence_can = False
        self.silence_efference1 = False
        self.silence_efference2 = True
        self.partitions = {
            "can": [0, self.input_sizes["can"]],
            "sensory": [self.input_sizes["can"], self.input_sizes["can"] + self.input_sizes["sensory"]],
            "efference1": [
                self.input_sizes["can"] + self.input_sizes["sensory"],
                self.input_sizes["can"] + self.input_sizes["sensory"] + self.input_sizes["efference1"]
            ],
            "efference2": [
                self.input_sizes["can"] + self.input_sizes["sensory"] + self.input_sizes["efference1"],
                self.input_sizes["can"] + self.input_sizes["sensory"] + self.input_sizes["efference1"] + self.input_sizes["efference2"]
            ],
        }

    def forward(self, x, h=None):
        hidden = 0
        for key, lin in self.input_streams.items():
            add = True
            match key:
                case 'efference1':
                    if self.silence_efference1:
                        add = False
                case 'efference2':
                    if self.silence_efference2:
                        add = False
                case 'sensory':
                    if self.silence_sensory:
                        add = False
                case 'can':
                    if self.silence_can:
                        add = False
            if add:
                start, end = self.partitions[key]
                hidden = hidden + lin(x[..., start:end])

        hidden = self.activation(hidden)
        for layer in self.hidden_layers:
            hidden = self.activation(layer(hidden))

        y = self.W_output(hidden)
        y = self.output_activation(y)

        if self.normalize:
            y = y / torch.norm(y, dim=-1, keepdim=True)

        return h, y

class LAL(nn.Module):
    def __init__(
        self,
        controller_dim,
        n_lal_neurons=64,
        use_dale=True,
        excitation_ratio=0.8,
        dale_rectifier='softplus',  # choose from ['exp', 'abs', 'softplus']
        init_scale=0.1,
        dt =0.01,
        hand_designed = False,
    ):
        """
        Args:
            controller_dim: number of controller output neurons
            n_lal_neurons: number of neurons of two LAL hemispheres together
            use_dale: whether to apply Dale's law
            excitation_ratio: fraction of excitatory controller neurons
            dale_rectifier: function used to enforce positivity ('exp', 'abs', 'softplus')
            init_scale: stddev for initialization of raw weight parameters
        """
        super().__init__()
        self.state_dim = n_lal_neurons
        self.dim = self.state_dim
        self.controller_dim = controller_dim
        self.use_dale = use_dale
        self.excitation_ratio = excitation_ratio
        self.dale_rectifier = dale_rectifier.lower()
        self.dt = dt


        assert self.dale_rectifier in ['exp', 'abs', 'softplus'], "Invalid dale_rectifier"

        # Raw unconstrained weights
        self.W_par = nn.Parameter(init_scale * np.sqrt(32) / np.sqrt(self.controller_dim) * torch.randn(n_lal_neurons, controller_dim // 2))
        self.perm = torch.cat([
            torch.linspace(n_lal_neurons // 2, n_lal_neurons - 1, n_lal_neurons // 2, dtype=torch.long),
            torch.linspace(0, n_lal_neurons // 2 - 1, n_lal_neurons // 2, dtype=torch.long)
            ], dim=0)
        

        # if hand_designed: 
        #     self.W_param.data = torch.cat

        if use_dale:
            # Dale sign mask: shape (1, controller_dim), +1 for E, -1 for I
            signs = torch.ones(controller_dim // 2, dtype=torch.float32)
            n_exc = int(excitation_ratio * controller_dim // 2)
            signs[n_exc:] = -1
            signs = torch.cat([signs, signs], dim=0)
            self.register_buffer("dale_signs", signs.view(1, -1))

    def apply_dale_constraint(self, W_param):
        if self.dale_rectifier == 'exp':
            W_pos = torch.exp(W_param)
        elif self.dale_rectifier == 'abs':
            W_pos = torch.abs(W_param)
        elif self.dale_rectifier == 'softplus':
            W_pos = F.softplus(W_param)
        else:
            raise ValueError("Unknown rectifier: " + self.dale_rectifier)
        return W_pos * self.dale_signs

    def forward(self, controller_output, h = None, dt = None):
        """
        Args:
            controller_output: shape [..., controller_dim]
            h and dt are not used in this class, but are included for compatibility with other modules
        Returns:
            lal_activity: shape [..., n_lal_neurons]
        """

        self.W_param = torch.cat([self.W_par, self.W_par[self.perm]], dim = 1)
        if self.use_dale:
            W = self.apply_dale_constraint(self.W_param)
        else:
            W = self.W_param

        lal_activity = F.relu(F.linear(F.relu(controller_output), W))
        return lal_activity, lal_activity
    
    def to(self, *args, **kwargs):
        """
        Override to ensure that the parameters are moved correctly.
        """
        super().to(*args, **kwargs)
        self.W_param = self.W_param.to(*args, **kwargs)
        return self