"""This module wraps the specifications of the training process for the agents


"""

import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import Rotations as Rot
from copy import deepcopy
from utils import goto_project_root
from torch.optim import lr_scheduler as LR

def reinitialise_weights(model):
    for layer in model.children():
        if not hasattr(layer, 'frozen') or not layer.frozen:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            elif isinstance(layer, nn.Sequential) or isinstance(layer, nn.ModuleList):
                reinitialise_weights(layer)

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
        self.silence_activity_loss = None if not config.get("silence_activity", False) else self._regularisation_loss_functions(
            config["silence_activity_loss"]
        )
        self.weight_regs_loss = None  # TODO
        self.total_period = config["training_config"].get("seq_len", None)
        self.prep_phase = config["training_config"].get("prep_phase", None)
        self.action_period = self.total_period - self.prep_phase if self.prep_phase is not None else self.total_period
        self.model_specs = config["model_specs"]  # Should be a dictionary
        self.model = self._model_type(self.model_specs)
        self.model.resolution = config['training_config'].get('resolution', None)
        self.model.cam_position = config['training_config'].get('cam_position', 2.7)
        self.model.object_path = config['training_config'].get('object_path', None)
        self.model.action_period = self.action_period
        self.model.total_period = self.total_period
        self.device = config["device"]
        self.optimizer_specs = config["optimizer_specs"]
        self.optimizer = self._optimizer_type(self.optimizer_specs)
        self.lr_init = self.optimizer_specs['optimizer_params']["lr"]
        self.latest_checkpoint = 0
        if config.get("scheduler", False):
            self.scheduler = LR.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=100, min_lr=5e-4, cooldown=50)
            print("Using a scheduler")
        else:
            self.scheduler = None
            print("no scheduler")
        distance_weight = config["distance_weight"]
        output_regs_weight = config.get("output_regs_weight", 1)
        silence_activity_weight = config.get("silence_activity_weight", 0)
        weight_regs_weight = config.get("weight_regs_weight", 0)
        self.loss_fn = CombinedLoss(
            self.distance_loss,
            self.regularisation_loss,
            self.silence_activity_loss,
            self.weight_regs_loss,
            distance_weight=distance_weight,
            output_regs_weight=output_regs_weight,
            silence_activity_weight=silence_activity_weight,
            weight_regs_weight=weight_regs_weight,
            action_period=self.action_period,
            total_period=self.total_period,
        )

        self.model = self.model.to(self.device)
        for module in self.model.modules():
            module = module.to(self.device)
        self.model.device = self.device
        self.model.prep_phase = self.prep_phase

        self.best_val_loss = float("inf")
        self.best_val_loss_split = float("inf")
        self.best_model = None
        self.recorded_distance_loss = 0
        self.recorded_regularisation_loss = 0

    def refresh(self):
        reinitialise_weights(self.model)
        self.model.out = None
        self.optimizer = self._optimizer_type(self.optimizer_specs)
        self.scheduler = LR.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=100, min_lr=5e-4, cooldown=50)
        self.model.to(self.device)

    def train(
        self, train_loader, val_loader, epochs=10, save_interval=10, reset_interval = 200, save_path=None, check_path=None, log_path=None, early_stop = False, weight_regs = 0):

        if save_path is None:
            save_path = self.save_path
        if check_path is None:
            check_path = self.check_path
        if log_path is None:
            log_path = self.log_path

        self.writer = SummaryWriter(log_path)
        self.epochs_no_improve = 0
        self.best_val_loss_split = float("inf")

        for e in range(epochs):
            self.model.train()
            for i, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, self.model.pred, target)
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar("training loss", loss.item(), i + e * len(train_loader))

                # Log additional loss components
                self.writer.add_scalar("training recorded_distance_loss", self.loss_fn.recorded_distance_loss.mean().item(), i + e * len(train_loader))
                self.writer.add_scalar("training final_distance_loss", self.loss_fn.final_distance_loss.mean().item(), i + e * len(train_loader))
                self.writer.add_scalar("training recorded_regularisation_loss", self.loss_fn.recorded_regularisation_loss.mean().item(), i + e * len(train_loader))
                if self.loss_fn.weight_regs_loss is not None:
                    self.writer.add_scalar("training weight_regs_loss", self.loss_fn.weight_regs_loss.mean().item(), i + e * len(train_loader))
                if self.loss_fn.silence_activity_loss is not None:
                    self.writer.add_scalar("training silence_activity_loss", self.loss_fn.recorded_silence_activity_loss.mean().item(), i + e * len(train_loader))

            if (e + 1) % save_interval == 0:
                if check_path is not None:
                    self.save_checkpoint(check_path + "/checkpoint" + str(e) + ".pth", e, loss)
                    self.latest_checkpoint = e

            if (e + 1) % reset_interval == 0 and e/epochs < 0.75:
                if self.scheduler is not None:
                    self.scheduler = LR.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=100, min_lr=5e-4, cooldown=50)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr_init
                self.epochs_no_improve = 0

            if self.scheduler is None:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr_init

            if self.validate(val_loader, e):
                break
        self.save_model(save_path)

    def train_unlimited(self, data_generator, epochs = 10, save_interval=10, reset_interval = 500, save_path=None, check_path=None, log_path=None, early_stop = False, weight_regs = 0):
        """Same as the Train function, except that in this case we have unlimited data.
        train_loader and val_loader will no longer be provided; instead, there is a data generator we can use every epoch.

        Args of interest:
            - data_generator: a function that returns a dataloader with a given number of datapoints

        """

        if save_path is None:
            save_path = self.save_path
        if check_path is None:
            check_path = self.check_path
        if log_path is None:
            log_path = self.log_path

        self.writer = SummaryWriter(log_path)
        self.epochs_no_improve = 0
        self.best_val_loss_split = float("inf")

        batch_size = self.config.get("training_config", {}).get("batch_size", 128)
        mini_batch_size = self.config.get("training_config", {}).get("mini_batch_size", 32)
        val_size = round(self.config.get("training_config", {}).get("val_ratio", 0.1) * batch_size)
        replacement_baseline = self.config.get("training_config", {}).get("replacement_baseline", 0)
        replacement_tolerance = self.config.get("training_config", {}).get("replacement_tolerance", 0.5)
        epochs_to_calculate = self.config.get("training_config", {}).get("epochs_to_calculate", 5)
        refractory_period = epochs_to_calculate * 0.7
        last_resampling = 0
        train_loss_store = torch.tensor([], device = self.device)
        val_loss_store = torch.tensor([], device = self.device)
        temp = torch.arange(epochs_to_calculate, device = self.device, dtype = torch.get_default_dtype())
        temp_mean = temp.mean()
        temp = temp - temp_mean
        temp_SS = temp.norm(p = 2) ** 2
        temp = temp/temp_SS
        train_loss = 0
        val_loss = 0

        for e in range(epochs):
            self.model.train()
            if e == 0:
                train_loader, val_loader = data_generator.generate(n_samples = batch_size, val_size = val_size, record = True)

            elif e - last_resampling > epochs_to_calculate + refractory_period:
                train_loss_store = torch.cat((train_loss_store, torch.tensor([train_loss], device = self.device)))[-epochs_to_calculate:]
                val_loss_store = torch.cat((val_loss_store, torch.tensor([val_loss], device = self.device)))[-epochs_to_calculate:]
                # print(f"Train loss slope: {train_loss_store}")
                # print(f"Val loss slope: {val_loss_store}")
                train_loss_slope =  temp @ (train_loss_store - train_loss_store.mean())
                val_loss_slope = temp @ (val_loss_store - val_loss_store.mean())
                replacement_rate = self.compute_replacement_rate(train_loss_slope, val_loss_slope, baseline = replacement_baseline, tolerance = replacement_tolerance)
                if replacement_rate > 0:
                    train_loader, val_loader = data_generator.replace(replacement_rate) # Replace a proportion of the training set
                    self.train_loader = train_loader
                    self.val_loader = val_loader
                    last_resampling = e
                    train_loss_store = torch.tensor([], device=self.device)
                    val_loss_store = torch.tensor([], device=self.device)
            else:
                train_loss_store = torch.cat((train_loss_store, torch.tensor([train_loss], device = self.device)))
                val_loss_store = torch.cat((val_loss_store, torch.tensor([val_loss], device = self.device)))

            train_loss = 0
            for i, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, self.model.pred, target)
                loss = loss.mean()
                l1_reg = torch.tensor(0., requires_grad=True)
                for param in self.model.parameters():
                    l1_reg = l1_reg + torch.norm(param, 1) 
                loss = loss + weight_regs * l1_reg
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar("training loss", loss.item(), i + e * len(train_loader))

                # Log additional loss components
                self.writer.add_scalar("training recorded_distance_loss", self.loss_fn.recorded_distance_loss.mean().item(), i + e * len(train_loader))
                self.writer.add_scalar("training final_distance_loss", self.loss_fn.final_distance_loss.mean().item(), i + e * len(train_loader))
                self.writer.add_scalar("training recorded_regularisation_loss", self.loss_fn.recorded_regularisation_loss.mean().item(), i + e * len(train_loader))
                if self.loss_fn.weight_regs_loss is not None:
                    self.writer.add_scalar("training weight_regs_loss", self.loss_fn.weight_regs_loss.mean().item(), i + e * len(train_loader))
                if self.loss_fn.silence_activity_loss is not None:
                    self.writer.add_scalar("training silence_activity_loss", self.loss_fn.recorded_silence_activity_loss.mean().item(), i + e * len(train_loader))
                train_loss += loss

            train_loss /= len(train_loader)

            if (e + 1) % save_interval == 0:
                if check_path is not None:
                    self.save_checkpoint(check_path + "/checkpoint" + str(e) + ".pth", e, loss)
                    self.latest_checkpoint = e

            if (e + 1) % reset_interval == 0 and e/epochs < 0.75:
                if self.scheduler is not None:
                    self.scheduler = LR.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=100, min_lr=5e-4, cooldown=50)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr_init
                self.epochs_no_improve = 0

            if self.scheduler is None:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr_init

            if self.validate(val_loader, e, early_stop):
                break
            val_loss = self.val_loss
        self.save_model(save_path)

    def compute_replacement_rate(self, train_loss_slope, val_loss_slope, baseline = 0, tolerance = 0.5):
        """Rationale: if validation loss is improving, we can afford to replace a smaller proportion of the training set.
        Else (if val loss is static/increasing), this is sign of overfitting and we should replace more.

        Args:
            - train_loss_slope: the slope of the training loss curve, sampled across the past 5 validation points
            - val_loss_slope: the slope of the validation loss curve, sampled across the past 5 points

        Returns:
            - replacement_rate: the proportion of the training set to replace
        """
        ratio = val_loss_slope / train_loss_slope
        replacement_rate = min(1, max(baseline, 1 - ratio)) if val_loss_slope > train_loss_slope and ratio < tolerance else 0
        return replacement_rate

    def validate(self, val_loader, epoch, early_stop = False):
        self.model.eval()
        val_total_loss = 0
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, learning rate: {self.optimizer.param_groups[0]['lr']}")
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss = self.loss_fn(output[:, :, :], self.model.pred, target)
                val_loss = val_loss.mean()
                val_total_loss += val_loss

            avg_val_loss = val_total_loss / len(val_loader)

            if self.scheduler is not None:
                self.scheduler.step(avg_val_loss)

            self.writer.add_scalar("validation loss", avg_val_loss, epoch)
            self.val_loss = avg_val_loss

            # Log additional loss components
            self.writer.add_scalar("validation recorded_distance_loss", self.loss_fn.recorded_distance_loss.mean().item(), epoch)
            self.writer.add_scalar("validation final_distance_loss", self.loss_fn.final_distance_loss.mean().item(), epoch)
            self.writer.add_scalar("validation recorded_regularisation_loss", self.loss_fn.recorded_regularisation_loss.mean().item(), epoch)
            if self.loss_fn.weight_regs_loss is not None:
                self.writer.add_scalar("validation weight_regs_loss", self.loss_fn.weight_regs_loss.mean().item(), epoch)
            if self.loss_fn.silence_activity_loss is not None:
                self.writer.add_scalar("validation silence_activity_loss", self.loss_fn.recorded_silence_activity_loss.mean().item(), epoch)

            if avg_val_loss < self.best_val_loss_split:
                self.best_val_loss_split = avg_val_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= 100 and early_stop:
                    print("Early stopping")
                    return(1)
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.best_model = deepcopy(self.model.state_dict())
                self.save_checkpoint(self.check_path + "/best_model.pth", epoch, avg_val_loss)
            return(0)

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

    def save_model(self, path, full=False):
        if full:  # Save the full model
            torch.save(self.model, path)
        else:
            torch.save(self.model.state_dict(), path)

    def load_best_model(self):
        self.model.load_state_dict(self.best_model)

    def load_model(self, path, full=False):
        if full:
            self.model = torch.load(path)
        else:
            self.model.load_state_dict(torch.load(path))

        self.model.to(self.device)  # Just in case it's not there already

    def forward(self, features, labels):
        self.model.eval()
        with torch.no_grad():
            output = self.model(features)
            loss = self.loss_fn(output[:, -self.action_period:, :], self.model.pred, labels)
        return output, loss

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
            case "geodesic_gradual":
                return GeoGradualLoss(self.config["gradual_loss_weighting"], self.config['training_config'])
            case "chordal distance":
                return NotImplementedError
            case "angular distance":
                return NotImplementedError

    def _regularisation_loss_functions(self, loss_name):
        match loss_name:
            case "L2":
                return L2Regularisation()
            case "L2+exp": 
                return ExpL2Regularisation()
            case "L1":
                return NotImplementedError
            case "Elastic":
                return NotImplementedError

class CombinedLoss(nn.Module):
    """Implements the combined loss function"""

    def __init__(self, distance_loss, regularisation_loss, silence_activity_loss=None, weight_regs_loss=None, distance_weight=1, output_regs_weight=1, silence_activity_weight=0, weight_regs_weight=0, action_period=50, total_period=100):
        super(CombinedLoss, self).__init__()
        self.distance_loss = distance_loss
        self.regularisation_loss = regularisation_loss
        self.silence_activity_loss = silence_activity_loss
        self.weight_regs_loss = weight_regs_loss

        self.distance_weight = distance_weight
        self.output_regs_weight = output_regs_weight
        self.silence_activity_weight = silence_activity_weight
        self.weight_regs_weight = weight_regs_weight
        self.action_period = action_period
        self.total_period = total_period

    def forward(self, output, pred, target):
        self.recorded_distance_loss, final_distance_loss = self.distance_loss(pred, target)
        if final_distance_loss is None:
            final_distance_loss = self.recorded_distance_loss  # For the case of instant loss (no longer used)
        self.final_distance_loss = final_distance_loss

        self.recorded_regularisation_loss = self.regularisation_loss(output[:, -self.action_period:, :]) if self.action_period is not None else self.regularisation_loss(output)

        if self.silence_activity_loss is not None:
            self.recorded_silence_activity_loss = self.silence_activity_loss(output[:, :-self.action_period, :]) if self.action_period is not None else self.silence_activity_loss(output)
        else:
            self.recorded_silence_activity_loss = None
        if self.weight_regs_loss is not None:
            self.recorded_weight_regs_loss = self.weight_regs_loss(output)
        else:
            self.recorded_weight_regs_loss = None

        if self.weight_regs_weight == 0 and self.silence_activity_weight == 0:
            total_loss = (self.distance_weight * self.recorded_distance_loss +
                          self.output_regs_weight * self.recorded_regularisation_loss)
        elif self.weight_regs_weight == 0:
            total_loss = (self.distance_weight * self.recorded_distance_loss +
                          self.silence_activity_weight * self.recorded_silence_activity_loss +
                          self.output_regs_weight * self.recorded_regularisation_loss)
        elif self.silence_activity_weight == 0:
            total_loss = (self.distance_weight * self.recorded_distance_loss +
                          self.output_regs_weight * self.recorded_regularisation_loss +
                          self.weight_regs_weight * self.weight_regs_loss)
        else:
            total_loss = (self.distance_weight * self.recorded_distance_loss +
                          self.silence_activity_weight * self.recorded_silence_activity_loss +
                          self.output_regs_weight * self.recorded_regularisation_loss +
                          self.weight_regs_weight * self.weight_regs_loss)

        return total_loss

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
        loss = torch.norm(output, p=2, dim = -1).sum(dim=-1)
        return loss

class ExpL2Regularisation(RegularisationLoss): 
    """Implements the exp + L2 loss"""
    def __init__(self): 
        super(ExpL2Regularisation, self).__init__()

    def forward(self, output): 
        loss = torch.exp(torch.norm(output, p = 2, dim = -1)).sum(dim = -1)
        return loss
class DistanceLoss(nn.Module):
    """Implements a head handler for distance loss functions"""

    def __init__(self):
        super(DistanceLoss, self).__init__()

    def forward(self, pred, target):
        raise NotImplementedError

class GeoGradualLoss(DistanceLoss):
    """Implements the geodesic loss evaluated gradually, over the total rotation prediction at each time step.

    Methods:
        forward(pred, target): computes the geodesic distance between two quaternions at each time step
        - let's stick with quaternions for now.
    """
    def __init__(self, weighting_function, configs):
        super(GeoGradualLoss, self).__init__()
        match weighting_function:
            case "linear":
                self.weighting_function = lambda t_current: (t_current + 1) / (configs["seq_len"] - configs["prep_phase"])
            case "constant+linear":
                self.weighting_function = lambda t_current: 0.5 + (t_current + 1) / (configs["seq_len"] - configs["prep_phase"])
            case _:
                raise NotImplementedError(f"Weighting function {weighting_function} not implemented.")

    def forward(self, pred, target):
        self.weights = torch.tensor([self.weighting_function(i) for i in range(0, pred.shape[1])], device=pred.device).unsqueeze(0)  # (1, seq_len)
        dot_prods = torch.sum(pred * target.unsqueeze(1), dim=-1)  # (batch_size, seq_len)
        loss_train = self.weights * 2 * torch.acos(torch.abs(torch.clamp(dot_prods, -1 + 1e-6, 1 - 1e-6)))
        loss = torch.sum(loss_train, dim=-1)
        return loss, loss_train[..., -1] / self.weights[..., -1]

class GeodesicLoss(DistanceLoss):
    """Implements the geodesic loss function for quaternions

    Methods:
        forward(pred, target): computes the geodesic distance between two quaternions
    """
    def __init__(self, cls="quat"):
        super(GeodesicLoss, self).__init__()
        self.cls = cls

    def forward(self, pred, target):
        if self.cls == "quat":
            loss = 2 * torch.acos(torch.clamp(torch.abs(torch.sum(pred * target, dim=-1)), -1 + 1e-6, 1 - 1e-6))
            return loss, loss
        if self.cls == "mat":  # rotation matrix
            return torch.norm(torch.logm(pred @ target.transpose(-1, -2)), p="fro"), None
        if self.cls == "euler":
            Pred = Rot.exp_quat(pred)
            Target = Rot.exp_quat(target)
            return 2 * torch.acos(torch.clamp(torch.abs(torch.sum(Pred * Target, dim=-1)), -1, 1)), None
        return NotImplementedError(f"Class {self.cls} is not implemented.")