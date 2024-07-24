"""This module is used to generate datasets in an object-oriented manner.

Here I define the basic functionalities within the base DataGenerator Class.
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import random_split
from .RotationDataset import RotationDataset

class DataGenerator:
    def __init__(self, additional_config):
        self.additional_config = additional_config
        self.mini_batch_size = additional_config.get('mini_batch_size', 128)
        self.train_data = None
        self.val_data = None

    def generate(self, n_samples, val_size = None, record = False):
        if val_size is None:
            val_size = int(n_samples * 0.1)
        train_data = self.generate_data(n_samples)
        val_data = self.generate_data(val_size)
        if record:
            self.train_data = train_data
            self.val_data = val_data
            train_loader = DataLoader(train_data, batch_size=self.mini_batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=self.mini_batch_size, shuffle=True)
            return train_loader, val_loader
        return train_data, val_data

    def replace(self, replacement_rate):
        """Replace a fraction of the data with new data.

        Args:
            replacement_rate (float): the fraction of the data to replace

        Returns:
            train_loader, val_loader (pytorch DataLoader): the new data loaders
        """
        if int(len(self.train_data) * replacement_rate) >= len(self.train_data) or int(len(self.val_data) * replacement_rate) >= len(self.val_data):
            return self.generate(len(self.train_data), len(self.val_data), record = True)
        if self.train_data is None:
            raise ValueError("No data to replace")
        l_t = len(self.train_data)
        l_v = len(self.val_data)
        self.train_data_features = self.train_data.data[int(len(self.train_data) * replacement_rate):]
        self.train_data_labels = self.train_data.labels[int(len(self.train_data) * replacement_rate):]
        self.val_data_features = self.val_data.data[int(len(self.val_data) * replacement_rate):]
        self.val_data_labels = self.val_data.labels[int(len(self.val_data) * replacement_rate):]
        train_2, val_2 = self.generate(l_t - len(self.train_data_labels), l_v - len(self.val_data_labels))
        train_data = torch.cat((self.train_data_features, train_2.data), dim = 0)
        train_labels = torch.cat((self.train_data_labels, train_2.labels), dim = 0)
        self.train_data = RotationDataset(train_data, train_labels)
        val_data = torch.cat((self.val_data_features, val_2.data), dim = 0)
        val_labels = torch.cat((self.val_data_labels, val_2.labels), dim = 0)
        self.val_data = RotationDataset(val_data, val_labels)
        print(f"Replaced {replacement_rate * 100}% of the data, new data size: {len(self.train_data)}, original size: {l_t}")
        train_loader = DataLoader(self.train_data, batch_size=self.mini_batch_size, shuffle=True)
        val_loader = DataLoader(self.val_data, batch_size=self.mini_batch_size, shuffle=True)
        return train_loader, val_loader

    def generate_data(self, n_samples):
        raise NotImplementedError