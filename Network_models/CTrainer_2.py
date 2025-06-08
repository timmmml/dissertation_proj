import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import Rotations as Rot
from copy import deepcopy
from utils import goto_project_root
from torch.optim import lr_scheduler as LR
import SimulateDatasets.PredictorModelDataGenerator as dg
import time

from importlib import reload

reload(dg)


def reinitialise_weights(model):
    for layer in model.children():
        if not hasattr(layer, "frozen") or not layer.frozen:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
            elif (
                isinstance(layer, nn.Sequential)
                or isinstance(layer, nn.ModuleList)
                or isinstance(layer, nn.Module)
            ):
                reinitialise_weights(layer)



class BaseTrainer:
    def __init__(self, config=None):
        if config is None:
            config = self._default_config()
        self.config = config
        self.save_path = config["save_path"]
        self.check_path = config.get("check_path", None)
        self.log_path = config["log_path"]
        
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self.device)
        
        self.optimizer_specs = config["optimizer_specs"]
        self.optimizer = self._optimizer_type(self.optimizer_specs)
        self.lr_init = self.optimizer_specs["optimizer_params"]["lr"]
        
        if config.get("scheduler", False):
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.1, patience=100, min_lr=5e-4, cooldown=50
            )
            print("Using a scheduler")
        else:
            self.scheduler = None
            print("No scheduler")
        
        self.best_val_loss = float("inf")
        self.best_model = None
        self.writer = SummaryWriter(self.log_path)
        
    def refresh(self):
        self.optimizer = self._optimizer_type(self.optimizer_specs)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=100, min_lr=5e-4, cooldown=50
        )
        
    def _optimizer_type(self, optimizer_specs):
        optimizer_name = optimizer_specs["optimizer_name"]
        try:
            OptimizerClass = getattr(optim, optimizer_name)
            return OptimizerClass(self.model.parameters(), **optimizer_specs["optimizer_params"])
        except AttributeError:
            print(f"Optimizer {optimizer_name} not found in torch.optim")
        except Exception as e:
            print(f"Failed to initialize optimizer: {e}")
        return None

class CTrainer_CAN(BaseTrainer):
    def __init__(self, config=None):
        super().__init__(config)
        
        self.model_specs = config["model_specs"]
        self.model = self._model_type(self.model_specs, config=False)
        self.model = self.model.to(self.device)
        for module in self.model.modules():
            module = module.to(self.device)
        self.model.device = self.device
    
    def train(self, **kwargs):
        self.optimizer = self._optimizer_type(self.optimizer_specs)
        epochs = kwargs.get("epochs", 1000)
        batch_size = kwargs.get("batch_size", 128)
        mini_batch_size = kwargs.get("mini_batch_size", 32)
        print(f"batch_size: {batch_size}, mini_batch_size: {mini_batch_size}")
        
        for e in range(epochs):
            self.model.train()
            train_loss = 0
            for i, data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model(data))
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            print(f"Epoch {e}, training loss: {train_loss/len(self.train_loader)}") 