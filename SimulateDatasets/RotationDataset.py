# import torch
from torch.utils.data import Dataset
# import numpy as np
from scipy.spatial.transform import Rotation as R


class RotationDataset(Dataset):
    """ This module extends pytorch's dataset module to encapsulate rotations

    Args:
        features (tensor(batch_size, seq_len, input_size)): features to infer from
        target (tensor(batch_size, 4)): the appropriate rotation for each sequence (quaternion)

    Attributes:
        features (tensor(batch_size, seq_len, input_size)): features to infer from
        target (tensor(batch_size, 4)): the appropriate rotation for each sequence (quaternion)
        rotation (Rotation): the rotation object, implemented with scipy.spatial.transform.Rotation
            this way it's easier to plot &c.
    """

    def __init__(self, features, target):
        self.data = features
        self.labels = target
        if self.labels.shape[-1] == 4:
            self.rotation = R.from_quat(self.labels.numpy())
        elif self.labels.shape[-2] == 3 and self.labels.shape[-1] == 3:
            self.rotation = R.from_matrix(self.labels.numpy())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]