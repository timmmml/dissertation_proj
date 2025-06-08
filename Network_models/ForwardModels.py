"""This module contains the various classes the predictor/encoder/decoder can take form of.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .RotationVAE import RotationVAE
from torch import jit 

class PCAEncoder(nn.Module):
    def __init__(self, model_params):
        super(PCAEncoder, self).__init__()
        self.input_size = model_params.get("input_size", 64 * 64)
        self.PC_num = model_params.get("PC_num", 400)
        self.encoding_dim = self.PC_num
        assert model_params.get("stored_params_path", False), "stored_params_path must be provided"
        self.stored_params_path = model_params.get("stored_params_path", None)
        if self.stored_params_path is not None: 
            self.load_stored()
        
    def train_fast(self, data):
        """Let's assume data comes in the shape of (n_samples, 1, 64, 64)"""
        # extract PCs from flattened data. 
        flattened = data.view(-1, self.input_size) - data.view(-1, self.input_size).mean(dim=0)
        U, S, V = torch.svd(flattened)
        self.PCs = V[:, :self.PC_num] # Now the shape will be (input_size, PCs)
    
    def load_stored(self):
        self.stored_params = torch.load(self.stored_params_path)
        self.V = self.stored_params.get("V", self.stored_params.get("v_10000", None))
        self.mean = self.stored_params.get("mean", self.stored_params.get("mean_10000", None))
        self.s = self.stored_params.get("s", self.stored_params.get("s_10000", None))
        self.PCs = self.V[:, :self.PC_num]
        
    def forward(self, x):
        """flattens the input and map it to the top few principal components"""
        flattened = x.view(x.shape[0], x.shape[1], -1) - self.mean.unsqueeze(0).unsqueeze(0)
        return flattened @ self.PCs
    
    def decode(self, x):
        """decode the latent representation back to the original space"""
        return (x @ self.PCs.T + self.mean).reshape(x.shape[0], x.shape[1], int(self.input_size ** 0.5), int(self.input_size ** 0.5))


class VAEEncoder(nn.Module): 
    def __init__(self, model_params):
        super(VAEEncoder, self).__init__()
        assert model_params.get("stored_model_path", False), "stored_model_path must be provided"
        self.stored_model_path = model_params.get("stored_model_path", None)
        self.model = torch.load(self.stored_model_path)
        self.model.encoder.eval()
        self.model.decoder.eval()
        self.encoding_dim = self.model.encoder.latent_dimension
        self.xshape = None

    def forward(self, x):
        if x.dim() == 5: 
            self.xshape = x.shape
            x_r = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
        mu, _ = self.model.encode(x_r)
        return mu.view(-1, self.xshape[-4], self.encoding_dim)

    def decode(self, z):
        if z.dim() == 3: 
            z_r = z.view(-1, z.shape[-1])
        return self.model.decode(z_r).view(z.shape[0], z.shape[1], self.xshape[-3], self.xshape[-2], self.xshape[-1])
from typing import Dict, List, Any

class FFModel(nn.Module): 
    def __init__(self, model_params: Dict[str, Any]):
        super(FFModel, self).__init__()
        self.input_size = model_params.get("input_size", 200) # input size is the latent dimension (input an initial state)
        self.hidden_sizes = model_params.get("hidden_sizes", [256, 64]) # slightly overparametrize hidden 1
        self.output_size = model_params.get("output_size", 4) # output can be either PC, VAE/AE latent dimension, or a quaternion. 
        self.activation = model_params.get("activation", "ReLU")

        self.graph = LinearActivation(self.input_size, self.output_size, self.hidden_sizes, activation=self.activation)

    def forward(self, x):
        # """forward pass

        # Args:
        #     x (torch.Tensor): input tensor of shape (batch_size, input_size)
        # """
        output = self.graph(x)
        if self.output_size == 4:
            output = output / output.norm(p = 2, dim=-1, keepdim = True)
        if output.dim() == 3: 
            return output
        else: 
            return output.unsqueeze(1)


class FastRecurrence(nn.Module): 
    def __init__(self, model_params): 
        super(FastRecurrence, self).__init__()
        self.input_size = model_params.get("input_size", 200) # input size is the latent dimension (input an initial state)
        self.output_size = model_params.get("output_size", 4) # output can be either PC, VAE/AE latent dimension, or a quaternion.
        self.input_presentation_time = model_params.get("input_presentation_time", 1)        
        self.silence_time = model_params.get("silence_time", 0) # NOTE: This is when the input is silent, i.e., action period
        self.stepwise = model_params.get("stepwise", 0)
        self.RNN_type = model_params.get("RNN_type", "GRU")
        rnn_specs = model_params.get("RNN_params", {
            "input_size": self.input_size, 
            "hidden_size": self.output_size, 
            "num_layers": 1, 
            "batch_first": True
        })
        self.hidden_size = rnn_specs["hidden_size"]
        rnn_specs["input_size"] = self.input_size
        self.rnn = getattr(nn, self.RNN_type)(**rnn_specs)
        self.activation = model_params.get("activation", "ReLU")
        self.activation_fn = getattr(nn, self.activation)()
        linear_hidden_params = model_params.get("linear_params", {})
        self.flatten_readout = model_params.get("flatten_readout", 0) # if True, the linear layer is applied to temporally flattened output.

        linear_hidden_params['input_size'] = self.hidden_size if not self.flatten_readout else self.hidden_size * (self.input_presentation_time + self.silence_time)
        linear_hidden_params['output_size'] = self.output_size
        self.linear_output = FFModel(linear_hidden_params)
         
    def forward(self, x): 
        """forward pass

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, input_size)

            hx (torch.Tensor, optional): hidden state. Defaults to None.
        """
        # first we repeat and pad the input.
        x = x.unsqueeze(1).repeat(1, self.input_presentation_time, 1)
        if self.silence_time > 0:
            x = torch.cat([x, torch.zeros(x.size(0), self.silence_time, x.size(-1), device=x.device)], dim=1)

        # pass through RNN
        output, _ = self.rnn(x)
        output = self.activation_fn(output)
        if self.stepwise: 
            if self.flatten_readout:
                output = output.flatten(start_dim=1).unsqueeze(1).repeat(1, self.silence_time, 1)
            else: 
                output = output[:, -self.silence_time:, :]
        else:
            if self.flatten_readout:
                output = output.flatten(start_dim=1)
            else: 
                output = output[:, -1, :]
        return self.linear_output(output)


# ----------------- Helper classes -----------------#

class LinearActivation(nn.Module): 
    def __init__(self, input_size: int, output_size: int, hidden_sizes: list, activation: str ="ReLU"):
        super(LinearActivation, self).__init__()
        if isinstance(hidden_sizes, int): 
            hidden_sizes = [hidden_sizes]
        if hidden_sizes is None:
            hidden_sizes = []
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        activation_fn = getattr(nn, activation)
        
        self.graph = nn.Sequential()

        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                self.graph.add_module(f"Linear_{i}", nn.Linear(input_size, hidden_size))
                self.graph.add_module(f"Activation_{i}", activation_fn())
            else:
                self.graph.add_module(f"Linear_{i}", nn.Linear(hidden_sizes[i-1], hidden_size))
                self.graph.add_module(f"Activation_{i}", activation_fn())
        
        if hidden_sizes:  # Only add final layer if hidden_sizes is not empty
            self.graph.add_module(f"Linear_{len(hidden_sizes)}", nn.Linear(hidden_sizes[-1], output_size))
        else:
            self.graph.add_module("Linear_output", nn.Linear(input_size, output_size))

    def forward(self, x):
        return self.graph(x)


# ----------------- Obsolete -----------------#
class FFModel_jit(jit.ScriptModule): 
    """experiment with jit - apparently not much faster and theres an overhead."""
    def __init__(self, model_params):
        super(FFModel_jit, self).__init__()
        self.input_size = model_params.get("input_size", 200) # input size is the latent dimension (input an initial state)
        self.hidden_sizes = model_params.get("hidden_sizes", [256, 64]) # slightly overparametrize hidden 1
        self.output_size = model_params.get("output_size", 4) # output can be either PC, VAE/AE latent dimension, or a quaternion. 
        self.activation = model_params.get("activation", "ReLU")
        self.graph = LinearActivation(self.input_size, self.output_size, self.hidden_sizes, activation=self.activation)

    @jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.graph(x)
        if self.output_size == 4:
            output = output / output.norm(dim=-1, p = 2).unsqueeze(-1)
        return output
