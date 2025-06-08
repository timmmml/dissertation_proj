"""This file implements the RotationVAE model, empowered by Pytorch.

Goal: learn a latent representation of rotated objects by VAE.

Purpose:
    1. familiarise with VAE idea and implementation.
    2. use these representations to ease training a world model in which the model understands consequences of rotations.

Constraint:
    1. Fixed object; no translation or scaling; no lighting differences; white background

With this constraint, the task is expected to be easier and the latent variable should only capture the rotation
- Object identity information is captured in the generative model.
- Toy case would be: encoder maps image to quaternion; decoder maps quaternion to image.
    - reconstruction is bounded by rotation inference. This is an achievable bound: replace decoder with the ground-truth 3D model of the object and rotate!

Of course, the model can learn something else. That's worth investigating - how does this depend on the architecture and the information bottleneck?
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms



# The rotations-related segments are not relevant at the moment. Just in case I need them later

class RotationVAE(nn.Module):
    def __init__(self, config):
        super(RotationVAE, self).__init__()
        print("trying to initialise VAE")
        self.device = "cuda"
        self.config = config
        self.encoder = self.set_up_encoder(config.get('encoder', 'simple'))  # Args can be "pretrained"
        self.decoder = self.set_up_decoder(
            config.get('decoder', 'simple'))  # Args can be something else. Not implemented

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.sample_reparameterize(mean, torch.exp(log_var * 0.5))
        return self.decode(z), mean, log_var

    def encode(self, x):
        if not hasattr(self, 'encoder'):
            raise NotImplementedError
        else:
            return self.encoder(x)

    def decode(self, z):
        if not hasattr(self, 'decoder'):
            raise NotImplementedError
        else:
            return self.decoder(z)

    def sample_reparameterize(self, mu, var):
        """Reparameterization trick: z = mu + var * epsilon, where epsilon ~ N(0, 1)
        """
        epsilon = torch.randn_like(var).to(self.device)
        return mu + var * epsilon

    def to(self, device):
        self.device = device
        return super().to(device)

    def reconstruct(self, x):
        """Reconstruct the input x
        """
        x_hat, mu, log_var = self.forward(x)
        print(f"mu is {mu}")
        print(f"log_var is {log_var}")
        return self.decode(self.sample_reparameterize(mu, torch.exp(0.5 * log_var)))

    def set_up_encoder(self, encoder_type):
        match encoder_type:
            case 'simple':
                return SimpleEncoder(self.config)
            case 'pretrained':
                return PretrainedEncoder(self.config)
            case _:
                raise NotImplementedError

    def set_up_decoder(self, decoder_type):
        match decoder_type:
            case 'simple':
                return SimpleDecoder(self.config)
            case 'FC':
                return FCDecoder(self.config)
            case _:
                raise NotImplementedError

class SimpleEncoder(nn.Module):
    """As implemented in the matlab tutorial"""
    def __init__(self, config):
        super(SimpleEncoder, self).__init__()
        self.input_resolution = config.get('input_resolution', config.get('resolution', 64))
        self.latent_dimension = config.get('latent_dimension', 16)
        self.graph = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, 3, 2, 1),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(self.input_resolution * self.input_resolution, self.latent_dimension * 2)
        )
        # self.graph = nn.Sequential(
        #     nn.Conv2d(1, 64, 3, 1, 0), # 62 x 62
        #     nn.ReLU(), 
        #     nn.Conv2d(64, 32, 3, 2), # 31 x 31
        #     nn.Flatten(), 
        #     nn.Linear(28800, self.latent_dimension * 2)
        # )
        self.FC_mean = nn.Linear(self.latent_dimension*2, self.latent_dimension)
        self.FC_log_var = nn.Linear(self.latent_dimension*2, self.latent_dimension)
        # self.FC_sigma = nn.Sequential(
        #     nn.Linear(self.latent_dimension*2, self.latent_dimension),
        #     nn.Softmax(dim=-1)
        # )

    def forward(self, x):
        x = self.graph(x)
        if hasattr(self, "flatten"):
            x = self.flatten(x)
        mean = self.FC_mean(x)
        if hasattr(self, "FC_sigma"):
            sigma = self.FC_sigma(x) * 16 + 1e-6
            log_var = 2 * torch.log(sigma)
            return mean, log_var
        log_var = self.FC_log_var(x)
        return mean, log_var

class PretrainedEncoder(nn.Module):
    def __init__(self, config):
        super(PretrainedEncoder, self).__init__()
        raise NotImplementedError

class SimpleDecoder(nn.Module):
    def __init__(self, config):
        super(SimpleDecoder, self).__init__()
        self.output_resolution = config.get('output_resolution', config.get('resolution', 64))
        self.latent_dimension = config.get('latent_dimension', 16)
        self.graph = nn.Sequential(
            nn.Sequential(
                nn.Linear(self.latent_dimension, 7 * 7 * 64),
                nn.Unflatten(1, (64, 7, 7))
            ),
            nn.ConvTranspose2d(64, 64, 3, 2), #15 x 15
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2), # 31 x 31
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2), # 64 x 64
            nn.Sigmoid()
        )

    def forward(self, z):
        x_hat = self.graph(z)

        return x_hat

class FCDecoder(nn.Module):
    def __init__(self, config):
        super(FCDecoder, self).__init__()
        self.latent_dimension = config.get('latent_dimension', 16)
        self.output_resolution = config.get('output_resolution', config.get('resolution', 64))
        self.graph = nn.Sequential(
            nn.Linear(self.latent_dimension, self.output_resolution * self.output_resolution),
            nn.Unflatten(1, (1, self.output_resolution, self.output_resolution)),
            # nn.ConvTranspose2d(8, 3, 4, 2, 1),
            nn.Sigmoid()
        )
    def forward(self, z):
        x_hat = self.graph(z)
        return x_hat