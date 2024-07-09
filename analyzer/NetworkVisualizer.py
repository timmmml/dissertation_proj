"""This module wraps around a Network and visualizes neural activity, primarily using tensorboard

"""
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE

class NetworkVisualizer:
    def __init__(self, configs):
        self.log_path_overall = configs["log_path"]
        self.configs = configs
        self.model = None

    def load_model(self, model):
        self.model = model
        self.model.eval()

    def visualise_TSNE(self, features, target, n_components=3, random_state=0):
        log_path_TSNE = self.log_path_overall + "/TSNE"
        writer = SummaryWriter(log_path_TSNE)
        with torch.no_grad():
            activations = []


