"""This module contains the implementation of the PartitionedNet and PartitionedRNN classes.

PartitionedNet:
    A neural network model that consists of a partitioned recurrent neural network (RNN) followed by a linear readout layer.
    It takes input configurations as input and initializes the network accordingly.
    The network type can be specified as "PartitionedRNN" or any other supported type.
    The forward method performs the forward pass of the network, returning the hidden states and the output predictions.
    The to method is overridden to move the network to the specified device.
PartitionedRNN:
    A partitioned RNN model that can be used as a building block in the PartitionedNet.
    It takes input configurations as input and initializes the RNN accordingly.
    The forward method performs the forward pass of the RNN, returning the hidden states.
Note: The PartitionedNet class uses the PartitionedRNN class as its RNN component.
"""

from torch import nn
import torch
import torch.nn.functional as F
from .Activations import * 


class PartitionedNet(nn.Module):
    """This class implements my custom PartitionedNet model. Different partitions of the network interact
    differently with input generators.

    Args:
        configs (dict): A dictionary containing the configuration parameters for the PartitionedNet.
            - hidden_size (int, optional): The size of the hidden state in the RNN. Defaults to 10.
            - input_size (int, optional): The size of the input to the RNN. Defaults to 1.
            - loss_params (dict, optional): A dictionary containing the parameters for the loss function.
                - reg_loss_weight (float, optional): The weight for the regularization loss. Defaults to 0.0.
                - pred_loss_weight (float, optional): The weight for the prediction loss. Defaults to 1.0.
            - network_type (str, optional): The type of the network architecture. Defaults to "PartitionedRNN".
    Attributes:
        rnn (nn.Module): The recurrent neural network module used in the PartitionedNet.
        readout (nn.Linear): The linear layer used for the final readout of the hidden state.
        z (None): A placeholder attribute.
    Methods:
        forward(x, truncate_output=True):
            Performs a forward pass through the PartitionedNet.
            Args:
                x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, input_size).
                truncate_output (bool, optional): Whether to truncate the output or return the full sequence.
                    Defaults to True.
            Returns:
                torch.Tensor: The output tensor of shape (batch_size, sequence_length, input_size) if truncate_output is False,
                    otherwise the output tensor of shape (batch_size, input_size).
        to(device):
            Moves the PartitionedNet to the specified device.
            Args:
                device (torch.device): The device to move the PartitionedNet to.
            Returns:
                PartitionedNet: The PartitionedNet instance after moving it to the specified device.
    """

    def __init__(self, configs):
        super(PartitionedNet, self).__init__()
        network_type = configs.get("network_type", "PartitionedRNN")
        self.rnn = globals().get(network_type)(configs)

    def forward(self, x, truncate_output=True):
        h = self.rnn(x.float())
        return h[:, :, -self.rnn.rep_units :] - self.rnn.c.reshape(1, 1, -1) if truncate_output else h

    def to(self, device):
        self.rnn = self.rnn.to(device).float()
        return super().to(device)

class FlexibleNet(nn.Module):
    def __init__(self, configs):
        super(FlexibleNet, self).__init__()
        network_type = configs.get("network_type", "FlexibleRNN")
        self.rnn = globals().get(network_type)(configs)

    def forward(self, x, truncate_output=True):
        if truncate_output == False:
            print("truncate output doesn't matter!")
        h = self.rnn(x)

        return h[:, :, :] - self.rnn.c.reshape(1, 1, -1) if truncate_output else h

    def to(self, device):
        self.rnn = self.rnn.to(device)
        return super().to(device)



class PartitionedRNN(nn.Module):
    def __init__(self, kwargs) -> None:
        super(PartitionedRNN, self).__init__()
        self.rep_units = kwargs.get("rep_units", 256)
        self.input_units = kwargs.get("input_units", self.rep_units)
        self.input_size = kwargs.get("input_size", 1)
        self.input_connected = kwargs.get("input_connected", False)
        self.hihr_identity = kwargs.get("hihr_identity", False)
        print(f"input_connected: {self.input_connected}")

        # Define the total_hidden_size
        self.hidden_size = self.rep_units + self.input_units * self.input_size

        # Define weight matrices for each partition
        self.W_hrhr = nn.Parameter(
            torch.randn(self.rep_units, self.rep_units) * 10 / self.rep_units
        )  # Rep-to-Rep
        # self.W_hihr = nn.ParameterList([nn.Parameter(torch.randn(self.input_units, self.rep_units) * 0.1) for _ in range(self.input_size)])  # Input-to-Rep
        # self.W_hrhi = nn.ParameterList([nn.Parameter(torch.randn(self.rep_units, self.input_units) * 0.1) for _ in range(self.input_size)]) # Rep-to-Input
        # self.W_hihi = nn.ParameterList([nn.Parameter(torch.zeros(self.input_units, self.input_units) * 0.1) for _ in range(self.input_size)])  # Input-to-Input; no cross-input-channel communication
        if self.hihr_identity:
            self.W_hihr = nn.Parameter(
                torch.eye(self.input_units, self.rep_units).repeat(self.input_size, 1), requires_grad=False
            )
        else:
            self.W_hihr = nn.Parameter(
                torch.randn(self.input_size * self.input_units, self.rep_units) * 10 / self.rep_units
            )  # Input-to-Rep
        self.W_hrhi = nn.Parameter(
            torch.randn(self.rep_units, self.input_units * self.input_size) * 10 / self.input_units
        )  # Rep-to-Input
        if self.input_connected:
            self.W_hihi = nn.Parameter(
                torch.zeros(
                    self.input_units * self.input_size,
                    self.input_units * self.input_size,
                )
                * 10/ self.input_units
            )  # Input-to-Input;
        else:
            self.W_hihi = nn.Parameter(
                torch.zeros(self.input_size, self.input_units, self.input_units) * 10/ self.input_units #, requires_grad=False
            )  # Input-to-Input; no cross-input-channel communication

        # Biases
        self.bias_input, self.bias_rep = kwargs.get("bias_input", True), kwargs.get("bias_rep", True) 
        if self.bias_input:
            self.b_hi = nn.Parameter(torch.randn(self.input_units * self.input_size) * 10/self.input_units)
        else: 
            self.b_hi = nn.Parameter(torch.zeros(self.input_units * self.input_size), requires_grad=False)
        if self.bias_rep:
            self.b_hr = nn.Parameter(torch.randn(self.rep_units) * 10/self.input_units)
        else:
            self.b_hr = nn.Parameter(torch.zeros(self.rep_units), requires_grad=False)
        
        self.zero_null_input, self.zero_null_rep = kwargs.get("zero_null_input", False), kwargs.get("zero_null_rep", False)
        if self.zero_null_input:
            self.h0_input = nn.Parameter(torch.zeros(self.input_size, self.input_units), requires_grad=False) 
        else: 
            self.h0_input = nn.Parameter(torch.randn(self.input_size, self.input_units) * 10/self.input_units)
        if self.zero_null_rep:
            self.h0_rep = nn.Parameter(torch.zeros(self.rep_units), requires_grad=False)
        else:
            self.h0_rep = nn.Parameter(torch.randn(self.rep_units) * 10/self.rep_units)

        if kwargs.get("c_param", False):
            self.c = nn.Parameter(torch.randn(self.rep_units) * 10/self.rep_units)
        else:
            self.c = nn.Parameter(torch.zeros(self.rep_units), requires_grad=False)

        # Nonlinear activation
        self.nonlinearity = kwargs.get("nonlinearity", "tanh")
        try: 
            self.nonlinearity_fn = getattr(F, self.nonlinearity)
        except: 
            print(f"Using custom activation {self.nonlinearity}")
            self.nonlinearity_fn = globals().get(self.nonlinearity)()

        self.input_N = kwargs.get("input_N", 1)
        print(self.input_N)

    def forward(self, x):

        batch_size, seq_len = x.shape[0], x.shape[1]

        # Initialize hi_input and hi_rep with batch dimension
        hi_input = self.h0_input.unsqueeze(1).expand(
            -1, batch_size, self.input_units
        )  # shape is (input_size, batch_size, input_units)

        try: 
            hi_rep = self.h0_rep_alt.unsqueeze(0).expand(batch_size, -1)
        except:
            hi_rep = self.h0_rep.unsqueeze(0).expand(batch_size, -1)
            hi_rep = self.nonlinearity_fn(hi_rep)  # shape is (batch_size, rep_units)

        h_all = []
        h_all.append(torch.cat((hi_input.permute(1, 2, 0).reshape(batch_size, -1), hi_rep), dim=1).unsqueeze(1))

        for i in range(seq_len):
            verbose = False
            scaling_factor = x[:, i, :]

            # Update hi_input and hi_rep using the partitioned weight matrices
            if verbose: 
                print(hi_input[:, 2744])
                print(hi_rep[2744])
            for _ in range(self.input_N):
                if not self.input_connected:
                    hi_input = self.nonlinearity_fn(
                        torch.bmm(hi_input, self.W_hihi)
                        .permute(1, 2, 0)
                        .reshape(batch_size, -1)
                        + hi_rep @ self.W_hrhi
                        + self.b_hi.unsqueeze(0)
                    )  # shape is (batch_size, input_units* input_size)
                else:
                    term1 = (hi_input.permute(1, 2, 0).reshape(batch_size, -1, self.input_size)
                    * scaling_factor.unsqueeze(1)
                    ).reshape(batch_size, -1) @ self.W_hihi
                    # term1 = hi_input.permute(1, 2, 0).reshape(batch_size, -1) @ self.W_hihi
                    term2 = hi_rep @ self.W_hrhi
                    term3 = self.b_hi.unsqueeze(0)
                    hi_input = self.nonlinearity_fn(
                        term1 + term2 + term3
                    )
                    # hi_input = self.nonlinearity_fn(
                    #     (hi_input.permute(1, 2, 0).reshape(batch_size, -1))
                    #     @ self.W_hihi
                    #     + hi_rep @ self.W_hrhi
                    #     + self.b_hi
                    # )  # shape is (batch_size, input_units* input_size)
                    if verbose:
                        print(f"input is {hi_input[2744]}")
                        print(f"term 1 is {term1[2744]}")
                        print(f"term 2 is {term2[2744]}")
                        print(f"term 3 is {term3}")
                        print(f"their sum is {term1[2744] + term2[2744] + term3}")
                hi_input = hi_input.reshape(
                    batch_size, self.input_units, self.input_size
                ).permute(
                    2, 0, 1
                )  # shape is (input_size, batch_size, input_units)
                
            hi_input = hi_input.permute(1, 2, 0).reshape(
                batch_size, -1
            )  # shape is (batch_size, input_units* input_size)
            # hi_rep = self.nonlinearity_fn(
            #     hi_rep @ self.W_hrhr +
            #     sum([(hi_input[j] @ self.W_hihr[j]) * scaling_factor[j].unsqueeze(1) for j in range(self.input_size)])
            #     + self.b_hr
                # + torch.randn_like(hi_rep) * 0.1
            # )
            hi_rep = self.nonlinearity_fn(
                hi_rep @ self.W_hrhr
                + (
                    hi_input.reshape(batch_size, -1, self.input_size)
                    # hi_input.reshape(batch_size, self.input_size, -1)
                    * scaling_factor.unsqueeze(1)
                ).reshape(batch_size, -1)
                @ self.W_hihr
                + self.b_hr
                # + torch.randn_like(hi_rep) * .1
            )  # shape is (batch_size, rep_units)

            # Concatenate the input and representation parts for the full hidden state at this time step
            h_all.append(torch.cat((hi_input, hi_rep), dim=1).unsqueeze(1))
            hi_input = hi_input.reshape(
                batch_size, self.input_units, self.input_size
            ).permute(
                2, 0, 1
            )  # shape is (input_size, batch_size, input_units)
            # if hi_input.isnan().any():
            #     print(torch.where(hi_input.isnan()))
            #     print(hi_input)
            #     print(hi_rep)
            #     print(scaling_factor)
            #     raise ValueError("NaN in hi_input")
        
        # Concatenate all time steps
        hi = torch.cat(h_all, dim=1)
        return hi
    
    def step(self, hi, scaling_factor, noisy = False):
        # hi_input is of shape (batch_size, input_units*input_size)
        # hi_rep is of shape (batch_size, rep_units)
        hi_input = hi[:, :-self.rep_units]
        hi_rep = hi[:, -self.rep_units:]
        batch_size = hi_input.shape[0]
        hi_input = hi_input.reshape(batch_size, self.input_units, self.input_size).permute(2, 0, 1)
        # Update hi_input and hi_rep using the partitioned weight matrices
        for _ in range(self.input_N):
            if not self.input_connected:
                hi_input = self.nonlinearity_fn(
                    torch.bmm(hi_input, self.W_hihi)
                    .permute(1, 2, 0)
                    .reshape(batch_size, -1)
                    + hi_rep @ self.W_hrhi
                    + self.b_hi.unsqueeze(0)
                )
            else:
                term1 = hi_input.permute(1, 2, 0).reshape(batch_size, -1) @ self.W_hihi
                term2 = hi_rep @ self.W_hrhi
                term3 = self.b_hi.unsqueeze(0)
                hi_input = self.nonlinearity_fn(
                    term1 + term2 + term3
                )
            hi_input = hi_input.reshape(
                batch_size, self.input_units, self.input_size
            ).permute(2, 0, 1)
        hi_input = hi_input.permute(1, 2, 0).reshape(batch_size, -1)
        hi_rep = self.nonlinearity_fn(
            hi_rep @ self.W_hrhr
            + (
                hi_input.reshape(batch_size, -1, self.input_size)
                * scaling_factor.unsqueeze(1)
            ).reshape(batch_size, -1)
            @ self.W_hihr
            + self.b_hr
            # + torch.randn_like(hi_rep) * 0.1 * noisy
        )  # shape is (batch_size, rep_units)
        h = torch.cat((hi_input, hi_rep), dim=1)
        return h

    def get_dW(self, hr): 
        # Give one step of zero input and calculate the change in hidden state
        hr_new = self.nonlinearity_fn(
            hr @ self.W_hrhr + self.b_hr
        )
        return hr_new - hr



class PartitionedCTRNN(nn.Module):
    def __init__(self, kwargs) -> None:
        super(PartitionedCTRNN, self).__init__()
        self.rep_units = kwargs.get("rep_units", 256)
        self.input_units = kwargs.get("input_units", self.rep_units)
        self.input_size = kwargs.get("input_size", 1)
        self.input_connected = kwargs.get("input_connected", False)
        self.dt = kwargs.get("dt", 0.05)  # User-defined time step
        self.tau = nn.Parameter(
            torch.tensor(kwargs.get("tau", 0.05))
        )  # Learnable time constant
        if kwargs.get("freeze_tau", False):
            self.tau.requires_grad = False

        # Define the total_hidden_size
        self.hidden_size = self.rep_units + self.input_units * self.input_size

        # Define weight matrices for each partition
        self.W_hrhr = nn.Parameter(
            torch.randn(self.rep_units, self.rep_units) * 0.1
        )  # Rep-to-Rep
        self.W_hihr = nn.Parameter(
            torch.randn(self.input_size * self.input_units, self.rep_units) * 0.1
        )  # Input-to-Rep
        self.W_hrhi = nn.Parameter(
            torch.randn(self.rep_units, self.input_units * self.input_size) * 0.1
        )  # Rep-to-Input

        if self.input_connected:
            self.W_hihi = nn.Parameter(
                torch.zeros(
                    self.input_units * self.input_size,
                    self.input_units * self.input_size,
                )
                * 0.1
            )
        else:
            self.W_hihi = nn.Parameter(
                torch.zeros(self.input_size, self.input_units, self.input_units) * 0.1
            )

        # Biases
        self.b_hi = nn.Parameter(torch.randn(self.input_units * self.input_size) * 0.1)
        self.b_hr = nn.Parameter(torch.randn(self.rep_units) * 0.1)

        # Nonlinear activation
        self.nonlinearity = kwargs.get("nonlinearity", "tanh")
        try: 
            self.nonlinearity_fn = getattr(F, self.nonlinearity)
        except: 
            self.nonlinearity_fn = globals().get(self.nonlinearity)()

        # Initial hidden states
        self.h0_input = nn.Parameter(
            torch.randn(self.input_size, self.input_units) * 0.1
        )
        self.h0_rep = nn.Parameter(torch.randn(self.rep_units) * 0.1)

        self.input_N = kwargs.get("input_N", 1)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Initialize hi_input and hi_rep with batch dimension
        hi_input = self.h0_input.unsqueeze(1).expand(-1, batch_size, -1)
        hi_rep = self.h0_rep.unsqueeze(0).expand(batch_size, -1)

        h_all = []

        for i in range(seq_len):
            scaling_factor = x[:, i, :]

            # Update hi_input and hi_rep using the partitioned weight matrices
            for _ in range(self.input_N):
                if not self.input_connected:
                    hi_input = hi_input.permute(1, 2, 0).reshape(batch_size, -1)
                    hi_input = (1 - self.dt / self.tau) * hi_input + (
                        self.dt / self.tau
                    ) * (
                        self.nonlinearity_fn(
                            torch.bmm(
                                hi_input.reshape(
                                    batch_size, self.input_units, self.input_size
                                ).permute(2, 0, 1)
                            ),
                            self.W_hihi,
                        )
                        .permute(1, 2, 0)
                        .reshape(batch_size, -1)
                        + hi_rep @ self.W_hrhi
                        + self.b_hi
                    )
                else:
                    hi_input = hi_input.permute(1, 2, 0).reshape(batch_size, -1)
                    hi_input = hi_input * (1 - self.dt / self.tau)
                    +(self.dt / self.tau) * self.nonlinearity_fn(
                        (hi_input @ self.W_hihi + hi_rep @ self.W_hrhi + self.b_hi)
                    )
                hi_input = hi_input.reshape(
                    batch_size, self.input_units, self.input_size
                ).permute(2, 0, 1)

            hi_input = hi_input.permute(1, 2, 0).reshape(batch_size, -1)

            hi_rep = hi_rep + (self.dt / self.tau) * (
                -hi_rep
                + self.nonlinearity_fn(
                    hi_rep @ self.W_hrhr
                    + (
                        hi_input.reshape(batch_size, -1, self.input_size)
                        * scaling_factor.unsqueeze(1)
                    ).reshape(batch_size, -1)
                    @ self.W_hihr
                    + self.b_hr
                    + torch.randn_like(hi_rep) * 0.1
                )
            )

            h_all.append(torch.cat((hi_input, hi_rep), dim=1).unsqueeze(1))
            hi_input = hi_input.reshape(
                batch_size, self.input_units, self.input_size
            ).permute(2, 0, 1)

        hi = torch.cat(h_all, dim=1)
        return hi


class PartitionedRNN_scale_states(nn.Module):
    """
    the module is a bit obsolete. introduces jumps in input states which is not desirable
    """

    def __init__(self, kwargs) -> None:
        super(PartitionedRNN_scale_states, self).__init__()
        self.hidden_size = kwargs.get("hidden_size", 256)
        self.input_size = kwargs.get("input_size", 1)
        self.W_hh = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.1)
        self.b_hh = nn.Parameter(torch.randn(self.hidden_size) * 0.1)

        self.input_units = kwargs.get("input_units", self.hidden_size // 2)
        self.rep_units = self.hidden_size - self.input_units
        self.nonlinearity = kwargs.get("nonlinearity", "tanh")
        self.nonlinearity_fn = getattr(F, self.nonlinearity)
        self.h0 = nn.Parameter(torch.randn(self.hidden_size) * 0.1)
        # THERE IS NO INPUT, inputs are used to scale the communications from the input partition to the representation partition!

    def forward(self, x):
        hi = self.h0.clone()
        hi = hi.unsqueeze(0).repeat(x.shape[0], 1).unsqueeze(1)
        h_all = []
        for i in range(x.shape[1]):
            xi = x[:, i, :]
            scaling_factor = xi.squeeze(-1)
            W = self.W_hh.unsqueeze(0).clone().repeat(x.shape[0], 1, 1)
            W[
                :, 0 : self.input_units, (self.hidden_size - self.rep_units) :
            ] *= scaling_factor.unsqueeze(-1).unsqueeze(-1)
            hi = self.nonlinearity_fn(
                torch.bmm(hi, W).squeeze(1) + self.b_hh
            ).unsqueeze(1)
            h_all.append(hi)
        hi = torch.cat(h_all, dim=1)
        return hi


# we can as well define partitioned GRU, LSTM, etc. RNN is the simplest and it works, so we'll stick with it for now.
#--------------------

class FlexibleRNN(nn.Module):
    def __init__(self, kwargs) -> None:
        super(FlexibleRNN, self).__init__()
        self.rep_units = kwargs.get("rep_units", 256)
        self.input_units = kwargs.get("input_units", self.rep_units)
        self.input_size = kwargs.get("input_size", 1)

        # Define the total_hidden_size
        self.hidden_size = self.rep_units + self.input_units * self.input_size

        self.W_ir = nn.Linear(self.input_size, self.hidden_size ** 2, bias=True).float()

        self.zero_null_rep = kwargs.get("zero_null_rep", False)

        if self.zero_null_rep:
            self.h0_rep = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False).float()
        else:
            self.h0_rep = nn.Parameter(torch.randn(self.hidden_size) * 10/self.hidden_size).float()

        if kwargs.get("c_param", False):
            self.c = nn.Parameter(torch.randn(self.hidden_size) * 10/self.hidden_size).float()
        else:
            self.c = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False).float()

        # Nonlinear activation
        self.nonlinearity = kwargs.get("nonlinearity", "tanh")
        try: 
            self.nonlinearity_fn = getattr(F, self.nonlinearity)
        except: 
            self.nonlinearity_fn = globals().get(self.nonlinearity)()

        self.input_N = kwargs.get("input_N", 1)
        print(self.input_N)

    def forward(self, X):
        index = 0
        batch_size_large = X.shape[0]
        X = X.float()

        hi_all = []
        
        while batch_size_large > 0: 
            batch_size_large -= 128

            batch_size, seq_len = X[index: min(X.shape[0], index + 128)].shape[0], X.shape[1]
            x = X[index:min(X.shape[0], index + 128)]
            index += 128

            # Initialize hi_rep with batch dimension
            hi_rep = self.h0_rep.unsqueeze(0).expand(batch_size, -1)

            h_all = []
            W_ir = self.W_ir(x).reshape(batch_size, seq_len, self.hidden_size, self.hidden_size) # shape is batch_size, seq_len, hidden_size ** 2
            W_ir = W_ir.permute(0, 2, 3, 1).float()
            
            h_all.append(hi_rep.unsqueeze(1))

            for i in range(seq_len):
                # print(hi_rep.shape)
                hi_rep = self.nonlinearity_fn(
                    # torch.einsum("ij, jki->ik", hi_rep, W_ir[:, :, :, i])
                    torch.bmm(hi_rep.unsqueeze(1), W_ir[:, :, :, i]).squeeze(1)
                    # + torch.randn_like(hi_rep) * .1
                )  # shape is (batch_size, rep_units)

                # Concatenate the input and representation parts for the full hidden state at this time step
                h_all.append(hi_rep.unsqueeze(1))

            # Concatenate all time steps
            hi = torch.cat(h_all, dim=1)
            hi_all.append(hi)
        return torch.cat(hi_all, dim = 0)
    
    def step(self, hi, scaling_factor, noisy = False):
        # hi_input is of shape (batch_size, input_units*input_size)
        # hi_rep is of shape (batch_size, rep_units)
        raise NotImplementedError
        hi_input = hi[:, :-self.rep_units]
        hi_rep = hi[:, -self.rep_units:]
        batch_size = hi_input.shape[0]
        hi_input = hi_input.reshape(batch_size, self.input_units, self.input_size).permute(2, 0, 1)
        # Update hi_input and hi_rep using the partitioned weight matrices
        for _ in range(self.input_N):
            if not self.input_connected:
                hi_input = self.nonlinearity_fn(
                    torch.bmm(hi_input, self.W_hihi)
                    .permute(1, 2, 0)
                    .reshape(batch_size, -1)
                    + hi_rep @ self.W_hrhi
                    + self.b_hi.unsqueeze(0)
                )
            else:
                term1 = hi_input.permute(1, 2, 0).reshape(batch_size, -1) @ self.W_hihi
                term2 = hi_rep @ self.W_hrhi
                term3 = self.b_hi.unsqueeze(0)
                hi_input = self.nonlinearity_fn(
                    term1 + term2 + term3
                )
            hi_input = hi_input.reshape(
                batch_size, self.input_units, self.input_size
            ).permute(2, 0, 1)
        hi_input = hi_input.permute(1, 2, 0).reshape(batch_size, -1)
        hi_rep = self.nonlinearity_fn(
            hi_rep @ self.W_hrhr
            + (
                hi_input.reshape(batch_size, -1, self.input_size)
                * scaling_factor.unsqueeze(1)
            ).reshape(batch_size, -1)
            @ self.W_hihr
            + self.b_hr
            # + torch.randn_like(hi_rep) * 0.1 * noisy
        )  # shape is (batch_size, rep_units)
        h = torch.cat((hi_input, hi_rep), dim=1)
        return h

    def get_dW(self, hr): 
        # Give one step of zero input and calculate the change in hidden state
        hr_new = self.nonlinearity_fn(
            hr @ self.W_hrhr + self.b_hr
        )
        return hr_new - hr
    
    def to(self, device):
        self.W_ir = self.W_ir.to(device)
        self.h0_rep = self.h0_rep.to(device)
        self.c = self.c.to(device)
        return super().to(device)


