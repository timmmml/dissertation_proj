"""This module wraps the specifications of the training process for the agents


"""
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.transform import Rotation as R
import numpy as np
import Rotations as Rot

class Trainer:
    def __init__(self, config = None):
        if config is None:
            config = self._default_config()
        self.config = config

        self.loss_function = self._loss_functions(config['loss_function'])

    def _loss_functions(self, loss_name):
        match loss_name:
            case 'geodesic':
                return GeodesicLoss()

    def _default_config(self):
        """Not implemented yet."""
        raise NotImplementedError

class GeodesicLoss(nn.Module):
    def __init__(self, cls = "quat"):
        super(GeodesicLoss, self).__init__()
        self.cls = cls

    def forward(self, pred, target):
        if self.cls == "quat":
            return 2 * torch.acos(torch.abs(torch.sum(pred * target, dim = -1)))
        if self.cls == "mat": # rotation matrix
            return torch.norm(torch.logm(pred @ target.transpose(-1, -2)), p = 'fro')
        if self.cls == "euler":
            Pred = Rot.exp_quat(pred)
            Target = Rot.exp_quat(target)
            return 2 * torch.acos(torch.abs(torch.sum(Pred * Target, dim = -1)))
        return NotImplementedError(f"Class {cls} is not implemented.")


# class ConfigDict(dict):
#     def __init__(self, *args, **kwargs):
#         super(ConfigDict, self).__init__(*args, **kwargs)
#         self.__dict__ = self
#         if 'distance_loss' not in self:
#             self.distance_loss = 'geodesic'
#         if 'loss_weights' not in self:
#             self.loss_weights = {'distance': 1, 'regularise': 1}
#         if 'learning_rate' not in self:
#             self.learning_rate = 0.001
#         raise NotImplementedError("The class is under construction."
#                                   "For now, refer to README for a list of configs you"
#                                   "must have.")