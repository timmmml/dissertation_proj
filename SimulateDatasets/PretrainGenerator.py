"""Extends the base DataGenerator class to the model pretraining stage."""

import torch
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import random_split
from . import DataGenerator as d
from .RotationDataset import RotationDataset
from importlib import reload
from scipy.spatial.transform import Rotation as R
reload(d)


class PretrainGenerator(d.DataGenerator):
    def __init__(self, configs):
        super(PretrainGenerator, self).__init__(additional_config=configs)
        self.additional_config = configs
        self.output_noise = configs.get("output_noise", 0) # noise to the output, in theta
        self.mode = configs.get("mode", "quaternion") 
        assert self.mode in ["quaternion_double", "quaternion_+", "quaternion", "matrix"]

        if isinstance(self.output_noise, tuple):
            # In this case we have specified the output noise to the angular displacement and the noise to the axis separately.
            self.output_noise_theta = self.output_noise[0]
            self.output_noise_axis = self.output_noise[1]
        else:
            self.output_noise_theta = self.output_noise
            self.output_noise_axis = self.output_noise

        self.mini_batch_size = self.additional_config.get("mini_batch_size", 128)
        self.train_data = None
        self.val_data = None

    def generate_data(self, n_samples):
        """Generate a dataset of n_samples."""
        seq_len, prep_phase = self.additional_config["seq_len"], self.additional_config["prep_phase"]
        thetas = torch.rand(n_samples, 1, 1) * torch.pi
        axes = torch.rand(n_samples, 1, 3) * 2 - 1
        thetas_label = (thetas + torch.randn_like(thetas) * self.output_noise_theta)
        axes_label = (axes + torch.randn_like(axes) * self.output_noise_axis) * ((thetas_label > torch.pi).float() * -2 + 1)
        thetas_label = thetas_label * (thetas_label < torch.pi) + (2 * torch.pi - thetas_label) * (thetas_label > torch.pi)
        features = torch.cat((torch.cos(thetas/2), torch.sin(thetas/2) * axes / axes.norm(dim = -1).unsqueeze(-1)), dim=-1)
        if self.mode == "quaternion_double":
            features = torch.cat((features, -features), dim=-1)
        elif self.mode == "quaternion_+":
            features = features * (features[:, :, :1] *2 - 1)
        elif self.mode == "quaternion":
            features = features * (torch.rand_like(features[:, :, :1]) * 2 - 1)
        elif self.mode == "matrix":
            features_ = features.view(-1, 4)
            features_ = torch.tensor(R.from_quat(features_.cpu().numpy(), scalar_first=True).as_matrix()).to(features.device)
            features = features_.view(*features.shape[:-1], 9)
        features = features.repeat(1, prep_phase, 1)
        features = torch.cat((features, torch.zeros(features.shape[0], seq_len - prep_phase, features.shape[-1])), dim=-2)
        print(features.shape)
        features = features.float() 

        labels = torch.cat((torch.cos(thetas_label/2), torch.sin(thetas_label/2) * axes_label / axes_label.norm(dim = -1).unsqueeze(-1)), dim=-1)
        return(RotationDataset(features, labels.squeeze(1)))

