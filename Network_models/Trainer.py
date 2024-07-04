"""This module wraps the specifications of the training process for the agents


"""
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import Rotations as Rot
from utils import goto_project_root


class Trainer:
    def __init__(self, config=None):
        if config is None:
            config = self._default_config()
        self.config = config
        self.save_path = config["save_path"]
        self.check_path = config.get("check_path", None)
        self.log_path = config["log_path"]

        self.distance_loss = self._distance_loss_functions(config["distance_loss"])
        self.regularisation_loss = self._regularisation_loss_functions(
            config["regularisation_loss"]
        )

        self.action_period = config["training_config"]["seq_len"] - config["training_config"]["prep_phase"]

        self.model_specs = config["model_specs"]  # Should be a dictionary
        self.model = self._model_type(self.model_specs)
        self.model.action_period = self.action_period
        self.device = config["device"]
        self.optimizer_specs = config["optimizer_specs"]
        self.optimizer = self._optimizer_type(self.optimizer_specs)
        distance_weight = config["distance_weight"]
        self.loss_fn = CombinedLoss(
            self.distance_loss, self.regularisation_loss, distance_weight
        )
        self.model.to(self.device)

    def refresh(self):
        if self.model.cell_type == "RNN":
            self.model.rnn = nn.RNN(self.model.input_size, self.model.hidden_size, self.model.num_layers, batch_first=True)
        elif self.model.cell_type == "GRU":
            self.model.rnn = nn.GRU(self.model.input_size, self.model.hidden_size, self.model.num_layers, batch_first=True)
        else:
            self.model.rnn = nn.LSTM(self.model.input_size, self.model.hidden_size, self.model.num_layers, batch_first=True)

        self.model.fc = nn.Linear(self.model.hidden_size, self.model.output_size)
        self.model.out = None
        self.optimizer = self._optimizer_type(self.optimizer_specs)
        self.model.to(self.device)

    def train(
        self, train_loader, val_loader, epochs=10, save_interval=10, save_path = None, check_path = None, log_path = None):

        if save_path is None:
            save_path = self.save_path
        if check_path is None:
            check_path = self.check_path
        if log_path is None:
            log_path = self.log_path

        self.writer = SummaryWriter(log_path)

        for e in range(epochs):
            self.model.train()
            for i, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output[:, self.action_period:, :], self.model.pred, target)
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar(
                    "training loss", loss.item(), i + e * len(train_loader)
                )
            if e % save_interval == 0:
                if check_path is not None:
                    self.save_checkpoint(
                        check_path + "\\checkpoint" + str(e) + ".pth", e, loss
                    )

            self.validate(val_loader, e)
        self.save_model(save_path)

    def validate(self, val_loader, epoch):
        self.model.eval()
        val_total_loss = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss = self.loss_fn(output[:, self.action_period:, :], self.model.pred, target)
                val_loss = val_loss.mean()
                val_total_loss += val_loss
            self.writer.add_scalar(
                "validation loss", val_total_loss / len(val_loader), epoch
            )

    def save_checkpoint(self, path, e, loss):
        torch.save(
            {
                "epoch": e,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
            },
            path,
        )

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model.to(self.device)  # Just in case it's not there already

    def save_model(self, path, full = False):
        if full: # Save the full model
            torch.save(self.model, path)
        else:
            torch.save(self.model.state_dict(), path)

    def load_model(self, path, full = False):
        if full:
            self.model = torch.load(path)
        else:
            self.model.load_state_dict(torch.load(path))

        self.model.to(self.device)  # Just in case it's not there already

    def _model_type(self, model_specs):
        model_name = model_specs["model_name"]
        model = None
        try:
            # Dynamically import the module
            module = __import__(f"{model_specs['model_path']}", fromlist=[model_name])
            # Get the model class
            ModelClass = getattr(module, model_name)
            # Instantiate the model with parameters
            model = ModelClass(model_specs["model_params"])
        except Exception as e:
            print(f"Failed to load model: {e}")
        return model

    def _optimizer_type(self, optimizer_specs):
        optimizer_name = optimizer_specs["optimizer_name"]
        optimizer = None

        try:
            # Get the optimizer class from the optim module
            OptimizerClass = getattr(optim, optimizer_name)
            # Instantiate the optimizer with the model parameters and provided optimizer parameters
            optimizer = OptimizerClass(self.model.parameters(), **optimizer_specs["optimizer_params"])
        except AttributeError:
            print(f"Optimizer {optimizer_name} not found in torch.optim")
        except Exception as e:
            print(f"Failed to initialize optimizer: {e}")

        return optimizer

    def _distance_loss_functions(self, loss_name):
        match loss_name:
            case "geodesic":
                return GeodesicLoss()
            case "chordal distance":
                return NotImplementedError
            case "angular distance":
                return NotImplementedError

    def _regularisation_loss_functions(self, loss_name):
        match loss_name:
            case "L2":
                return L2Regularisation()
            case "L1":
                return NotImplementedError
            case "Elastic":
                return NotImplementedError

    def _default_config(self):
        """Not implemented yet. Goal is to handle missing pieces in config."""
        raise NotImplementedError


class CombinedLoss(nn.Module):
    """Implements the combined loss function"""

    def __init__(self, distance_loss, regularisation_loss, distance_weight=1):
        super(CombinedLoss, self).__init__()
        self.distance_loss = distance_loss
        self.regularisation_loss = regularisation_loss
        self.distance_weight = distance_weight

    def forward(self, output, pred, target):
        return self.distance_weight * self.distance_loss(
            pred, target
        ) + (1 - self.distance_weight) * self.regularisation_loss(output)


class RegularisationLoss(nn.Module):
    """Implements a head handler for regularisation loss functions"""

    def __init__(self):
        super(RegularisationLoss, self).__init__()

    def forward(self, pred):
        raise NotImplementedError


class L2Regularisation(RegularisationLoss):
    """Implements the L2 regularisation loss"""

    def __init__(self):
        super(L2Regularisation, self).__init__()

    def forward(self, output):
        loss = torch.norm(output, p=2)
        return loss


class DistanceLoss(nn.Module):
    """Implements a head handler for distance loss functions"""

    def __init__(self):
        super(DistanceLoss, self).__init__()

    def forward(self, pred, target):
        raise NotImplementedError


class GeodesicLoss(DistanceLoss):
    def __init__(self, cls="quat"):
        super(GeodesicLoss, self).__init__()
        self.cls = cls

    def forward(self, pred, target):
        if self.cls == "quat":
            return 2 * torch.acos(torch.clamp(torch.abs(torch.sum(pred * target, dim=-1)), -1, 1))
        if self.cls == "mat":  # rotation matrix
            return torch.norm(torch.logm(pred @ target.transpose(-1, -2)), p="fro")
        if self.cls == "euler":
            Pred = Rot.exp_quat(pred)
            Target = Rot.exp_quat(target)
            return 2 * torch.acos(torch.clamp(torch.abs(torch.sum(Pred * Target, dim=-1)), -1, 1))
        return NotImplementedError(f"Class {self.cls} is not implemented.")


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
