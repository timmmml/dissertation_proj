"""This module wraps the specifications of the training process for the trainer


"""

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


class CTrainer:
    def __init__(self, config=None):
        if config is None:
            config = self._default_config()
        self.config = config
        self.save_path = config["save_path"]
        self.check_path = config.get("check_path", None)
        self.log_path = config["log_path"]

        self.dynamics_model_specs = config.get("dynamics_model_specs", None)
        self.dynamics_model = self._model_type(self.dynamics_model_specs, config=True)

        self.model_specs = config["model_specs"]  # Should be a dictionary
        self.model_specs["model_params"]["W"] = self.dynamics_model.W
        self.model = self._model_type(self.model_specs, config=False)
        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        torch.set_default_device(self.device)
        self.optimizer_specs = config["optimizer_specs"]
        self.optimizer = self._optimizer_type(self.optimizer_specs)
        self.lr_init = self.optimizer_specs["optimizer_params"]["lr"]

        self.latest_checkpoint = 0

        if config.get("scheduler", False):
            self.scheduler = LR.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.1,
                patience=100,
                min_lr=5e-4,
                cooldown=50,
            )
            print("Using a scheduler")
        else:
            self.scheduler = None
            print("no scheduler")

        self.loss_fn = CustomLoss(
            config.get("additional_loss", []), self.dynamics_model.U, writer=None
        )
        self.need_control_trajectory = "control_norm" in config.get(
            "additional_loss", []
        )

        self.model = self.model.to(self.device)
        for module in self.model.modules():
            module = module.to(self.device)
        self.model.device = self.device

        self.best_val_loss = float("inf")
        self.best_val_loss_split = float("inf")
        self.best_model = None
        # For DataGenerator, we use the trainer's home data generator
        # print(data_loader_configs)

        self.control_period = config.get("control_period", 1000)
        self.random_initial_condition = config.get("random_initial_condition", False)
        if "data_loader_configs" in config:
            self.data_gen = self.data_generator(
                lean=False,
                custom_configs=config["data_loader_configs"] | {"target_seq_len": 5},
            )
        self.model.mode = self.data_gen.mesh.default_mode
        self.model.mesh = self.data_gen.mesh
        self.model.dynamics_model = self.dynamics_model
        # make the dynamics model free of gradients:
        for param in self.dynamics_model.parameters():
            param.requires_grad = False

    def refresh(self):
        reinitialise_weights(self.model)
        self.model.out = None
        self.optimizer = self._optimizer_type(self.optimizer_specs)
        self.scheduler = LR.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.1,
            patience=100,
            min_lr=5e-4,
            cooldown=50,
        )
        self.model.to(self.device)

    def train(self, **kwargs):
        self.optimizer = self._optimizer_type(self.optimizer_specs)
        if "data_loader_configs" in kwargs:
            self.data_gen = self.data_generator(
                lean=False,
                custom_configs=kwargs["data_loader_configs"] | {"target_seq_len": 5},
            )
        epochs = kwargs.get("epochs", 1000)
        save_interval = kwargs.get("save_interval", 10)
        reset_interval = kwargs.get("reset_interval", 200)
        save_path = kwargs.get("save_path", self.save_path)
        check_path = kwargs.get("check_path", self.check_path)
        log_path = kwargs.get("log_path", self.log_path)
        early_stop = kwargs.get("early_stop", False)
        verbose = kwargs.get("verbose", True)
        load_indices = kwargs.get("load_index", {"train": 0, "val": 0})
        load_upper = kwargs.get(
            "load_upper", 44
        )  # after this index is reached, we break the training loop.
        save_indices = kwargs.get("store_index", {"train": 0, "val": 0})

        self.writer = SummaryWriter(log_path)
        self.loss_fn.writer = self.writer
        self.epochs_no_improve = 0
        self.best_val_loss_split = float("inf")

        batch_size = kwargs.get("batch_size", 128)
        mini_batch_size = kwargs.get("mini_batch_size", 32)
        print(f"batch_size: {batch_size}, mini_batch_size: {mini_batch_size}")
        self.data_gen.mini_batch_size = mini_batch_size
        self.data_gen.load_index = load_indices
        self.data_gen.store_index = save_indices
        self.data_gen.save_path = kwargs.get("data_save_path", None)
        val_size = round(
            self.config.get("training_config", {}).get("val_ratio", 0.1) * batch_size
        )
        replacement_baseline = kwargs.get("replacement_baseline", 0)
        replacement_tolerance = kwargs.get("replacement_tolerance", 0.5)
        epochs_to_calculate = kwargs.get("epochs_to_calculate", 5)
        refractory_period = epochs_to_calculate * 0.7
        last_resampling = 0
        train_loss_store = torch.tensor([], device=self.device)
        val_loss_store = torch.tensor([], device=self.device)
        temp = torch.arange(
            epochs_to_calculate, device=self.device, dtype=torch.get_default_dtype()
        )
        temp_mean = temp.mean()
        temp = temp - temp_mean
        temp_SS = temp.norm(p=2) ** 2
        temp = temp / temp_SS
        train_loss = 0
        val_loss = 0

        for e in range(epochs):
            self.model.train()
            if e == 0:

                train_loader, val_loader = self.data_gen.generate(
                    n_samples=batch_size, val_size=val_size, record=True
                )
            elif e - last_resampling > epochs_to_calculate + refractory_period:
                train_loss_store = torch.cat(
                    (train_loss_store, torch.tensor([train_loss], device=self.device))
                )[-epochs_to_calculate:]
                val_loss_store = torch.cat(
                    (val_loss_store, torch.tensor([val_loss], device=self.device))
                )[-epochs_to_calculate:]
                train_loss_slope = temp @ (train_loss_store - train_loss_store.mean())
                val_loss_slope = temp @ (val_loss_store - val_loss_store.mean())
                replacement_rate = self.compute_replacement_rate(
                    train_loss_slope,
                    val_loss_slope,
                    baseline=replacement_baseline,
                    tolerance=replacement_tolerance,
                )
                if replacement_rate > 0:
                    if self.data_gen.load_index["train"] > load_upper:
                        self.save_model(save_path, full=True)
                        break
                    train_loader, val_loader = self.data_gen.replace(
                        replacement_rate * 1.0
                    )  # Replace a proportion of the training set
                    self.train_loader = train_loader
                    self.val_loader = val_loader
                    last_resampling = e
                    train_loss_store = torch.tensor([], device=self.device)
                    val_loss_store = torch.tensor([], device=self.device)
            else:
                train_loss_store = torch.cat(
                    (train_loss_store, torch.tensor([train_loss], device=self.device))
                )
                val_loss_store = torch.cat(
                    (val_loss_store, torch.tensor([val_loss], device=self.device))
                )

            train_loss = 0

            for i, data in enumerate(train_loader):
                self.optimizer.zero_grad()
                if self.random_initial_condition:
                    z0 = (
                        self.dynamics_model.sample_initial_condition(len(data[0]))
                        * self.dynamics_model.input_scalar
                    )
                    z0 = z0.to(self.device)
                else:
                    z0 = torch.zeros(data[0].shape, device=self.device)

                if not self.need_control_trajectory:
                    x, error_trajectory = self.model(
                        z0, data[1][:, -1, :], control_period=self.control_period
                    )
                    control_trajectory = None
                else:
                    x, control_trajectory, error_trajectory = self.model(
                        z0,
                        data[1][:, -1, :],
                        control_period=self.control_period,
                        return_control_trajectory=True,
                    )

                # norm_scalar = 1#e/epochs
                norm_scalar = (
                    (0.1 if e / epochs <= 0.2 else e / epochs)
                    if e / epochs <= 0.7
                    else 0.7
                )
                loss = self.loss_fn(
                    x,
                    error_trajectory,
                    control_trajectory=control_trajectory,
                    state_norm_weight=norm_scalar * 0.1,
                    control_norm_weight=norm_scalar * 0.1,
                    state_likelihood_weight=norm_scalar * 0.1,
                )
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()

                self.writer.add_scalar(
                    "training loss", loss.item(), i + e * len(train_loader)
                )
                self.writer.add_scalar(
                    "training final distance to actual z0",
                    (x[:, -1, :] - data[0]).norm(p=2, dim=-1).mean(),
                    i + e * len(train_loader),
                )
                self.writer.add_scalar(
                    "training final output image distance",
                    error_trajectory[:, -1].mean(),
                    i + e * len(train_loader),
                )
                self.loss_fn.log_components(i + e * len(train_loader), "training")
                train_loss += loss

            train_loss /= len(train_loader)

            if (e + 1) % save_interval == 0:
                if check_path is not None:
                    self.save_checkpoint(
                        check_path + "/checkpoint" + str(e) + ".pth", e, loss
                    )
                    self.latest_checkpoint = e

            if (e + 1) % reset_interval == 0 and e / epochs < 0.75:
                if self.scheduler is not None:
                    self.scheduler = LR.ReduceLROnPlateau(
                        self.optimizer,
                        mode="min",
                        factor=0.1,
                        patience=100,
                        min_lr=5e-4,
                        cooldown=50,
                    )
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.lr_init
                self.epochs_no_improve = 0

            if self.scheduler is None:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr_init

            if self.validate(val_loader, e, early_stop):
                break
            val_loss = self.val_loss
            if verbose:
                print(
                    f"Epoch {e}, training loss: {train_loss}, validation loss: {val_loss}"
                )
        self.save_model(save_path, full=True)

    def validate(self, val_loader, epoch, early_stop=False, epochs=1000):
        self.model.eval()
        val_total_loss = 0
        if epoch % 100 == 0:
            print(
                f"Epoch {epoch}, learning rate: {self.optimizer.param_groups[0]['lr']}"
            )
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                if self.random_initial_condition:
                    z0 = (
                        self.dynamics_model.sample_initial_condition(len(data[0]))
                        * self.dynamics_model.input_scalar
                    )
                    z0 = z0.to(self.device)
                else:
                    z0 = torch.zeros(data[0].shape, device=self.device)

                if not self.need_control_trajectory:
                    x, error_trajectory = self.model(
                        z0, data[1][:, -1, :], control_period=self.control_period
                    )
                    control_trajectory = None
                else:
                    x, control_trajectory, error_trajectory = self.model(
                        z0,
                        data[1][:, -1, :],
                        control_period=self.control_period,
                        return_control_trajectory=True,
                    )

                norm_scalar = (
                    (0 if epoch / epochs <= 0.2 else epoch / epochs)
                    if epoch / epochs <= 0.7
                    else 0.7
                )
                val_loss = self.loss_fn(
                    x,
                    error_trajectory,
                    control_trajectory=control_trajectory,
                    state_norm_weight=norm_scalar * 0.1,
                    control_norm_weight=norm_scalar * 0.1,
                    state_likelihood_weight=norm_scalar * 0.1,
                )
                val_loss = val_loss.mean()
                self.loss_fn.log_components(epoch, "validation")
                val_total_loss += val_loss

            avg_val_loss = val_total_loss / len(val_loader)

            if self.scheduler is not None:
                self.scheduler.step(avg_val_loss)

            self.writer.add_scalar("validation loss", avg_val_loss, epoch)
            self.val_loss = avg_val_loss

            self.writer.add_scalar(
                "validation final distance to actual z0",
                (x[:, -1, :] - data[0]).norm(p=2, dim=-1).mean(),
                epoch,
            )
            self.writer.add_scalar(
                "validation final output image distance",
                error_trajectory[:, -1, ...].mean(),
                epoch,
            )

            if avg_val_loss < self.best_val_loss_split:
                self.best_val_loss_split = avg_val_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= 100 and early_stop:
                    print("Early stopping")
                    return 1
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.best_model = deepcopy(self.model.state_dict())
                self.save_checkpoint(
                    self.check_path + "/best_model.pth", epoch, avg_val_loss
                )
            return 0

    def compute_replacement_rate(
        self, train_loss_slope, val_loss_slope, baseline=0, tolerance=0.5
    ):
        ratio = val_loss_slope / train_loss_slope
        replacement_rate = (
            min(1, max(baseline, 1 - ratio))
            if val_loss_slope > train_loss_slope and ratio < tolerance
            else 0
        )
        return replacement_rate

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

    def forward(self, features):
        print(
            "WARNING: Loss here is ELBO, not suitable right away with plotting functions"
        )
        self.model.eval()
        with torch.no_grad():
            x_hat, mu, log_var = self.model(features)
            loss = self.loss_fn(features, x_hat, mu, log_var)
        return x_hat, loss

    def data_generator(self, lean=False, custom_configs=None):
        if not lean:
            return dg.PredictorModelDataGenerator(
                self.dynamics_model,
                self.config if custom_configs is None else custom_configs,
            )
        else:
            return dg.LeanPredictorModelDataGenerator(
                self.dynamics_model,
                self.config if custom_configs is None else custom_configs,
            )

    def _model_type(self, model_specs, config=True):
        model_name = model_specs["model_name"]
        model = None
        try:
            module = __import__(f"{model_specs['model_path']}", fromlist=[model_name])
            ModelClass = getattr(module, model_name)
            if config:
                model = ModelClass(model_specs["model_params"])
            else:
                model = ModelClass(**model_specs["model_params"])
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
            optimizer = OptimizerClass(
                self.model.parameters(), **optimizer_specs["optimizer_params"]
            )
        except AttributeError:
            print(f"Optimizer {optimizer_name} not found in torch.optim")
        except Exception as e:
            print(f"Failed to initialize optimizer: {e}")

        return optimizer


class CTrainer_CAN:
    def __init__(self, config=None):
        if config is None:
            config = self._default_config()
        self.config = config
        self.save_path = config["save_path"]
        self.check_path = config.get("check_path", None)
        self.log_path = config["log_path"]

        self.model_specs = config["model_specs"]  # Should be a dictionary
        self.model = self._model_type(self.model_specs, config=False)

        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        torch.set_default_device(self.device)

        self.optimizer_specs = config["optimizer_specs"]
        self.optimizer = self._optimizer_type(self.optimizer_specs)
        self.lr_init = self.optimizer_specs["optimizer_params"]["lr"]

        self.latest_checkpoint = 0

        if config.get("scheduler", False):
            self.scheduler = LR.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.1,
                patience=100,
                min_lr=5e-4,
                cooldown=50,
            )
            print("Using a scheduler")
        else:
            self.scheduler = None
            print("no scheduler")

        self.loss_fn = CustomLoss(config.get("additional_loss", []), None, writer=None)
        self.need_control_trajectory = "control_norm" in config.get(
            "additional_loss", []
        )

        self.model = self.model.to(self.device)
        for module in self.model.modules():
            module = module.to(self.device)
        self.model.device = self.device

        self.best_val_loss = float("inf")
        self.best_val_loss_split = float("inf")
        self.best_model = None
        # For DataGenerator, we use the trainer's home data generator
        # print(data_loader_configs)

        self.control_period = config.get("control_period", 1000)

    def refresh(self):
        reinitialise_weights(self.model)
        self.model.out = None
        self.optimizer = self._optimizer_type(self.optimizer_specs)
        self.scheduler = LR.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.1,
            patience=100,
            min_lr=5e-4,
            cooldown=50,
        )
        self.model.to(self.device)

    def train(self, **kwargs):
        self.optimizer = self._optimizer_type(self.optimizer_specs)
        self.data_gen = kwargs.get("data_gen", None)
        assert self.data_gen is not None, "Data generator not provided"
        self.data_gen.rep_model = self.model.rep_net
        epochs = kwargs.get("epochs", 1000)
        save_interval = kwargs.get("save_interval", 10)
        reset_interval = kwargs.get("reset_interval", 200)
        save_path = kwargs.get("save_path", self.save_path)
        check_path = kwargs.get("check_path", self.check_path)
        log_path = kwargs.get("log_path", self.log_path)
        early_stop = kwargs.get("early_stop", False)
        verbose = kwargs.get("verbose", True)

        self.writer = SummaryWriter(log_path)
        self.loss_fn.writer = self.writer
        self.epochs_no_improve = 0
        self.best_val_loss_split = float("inf")

        batch_size = kwargs.get("batch_size", 128)
        mini_batch_size = kwargs.get("mini_batch_size", 32)
        print(f"batch_size: {batch_size}, mini_batch_size: {mini_batch_size}")
        self.data_gen.mini_batch_size = mini_batch_size
        val_size = round(
            self.config.get("training_config", {}).get("val_ratio", 0.1) * batch_size
        )
        replacement_baseline = kwargs.get("replacement_baseline", 0)
        replacement_tolerance = kwargs.get("replacement_tolerance", 0.5)
        epochs_to_calculate = kwargs.get("epochs_to_calculate", 5)
        refractory_period = epochs_to_calculate * 0.7
        last_resampling = 0
        train_loss_store = torch.tensor([], device=self.device)
        val_loss_store = torch.tensor([], device=self.device)
        temp = torch.arange(
            epochs_to_calculate, device=self.device, dtype=torch.get_default_dtype()
        )
        temp_mean = temp.mean()
        temp = temp - temp_mean
        temp_SS = temp.norm(p=2) ** 2
        temp = temp / temp_SS
        train_loss = 0
        val_loss = 0

        for e in range(epochs):
            self.model.train()
            if e == 0:
                train_loader, val_loader = self.data_gen.generate(
                    n_samples=batch_size, val_size=val_size, record=True
                )
            elif e - last_resampling > epochs_to_calculate + refractory_period:
                train_loss_store = torch.cat(
                    (train_loss_store, torch.tensor([train_loss], device=self.device))
                )[-epochs_to_calculate:]
                val_loss_store = torch.cat(
                    (val_loss_store, torch.tensor([val_loss], device=self.device))
                )[-epochs_to_calculate:]
                train_loss_slope = (
                    temp @ (train_loss_store - train_loss_store.mean()).float()
                )
                val_loss_slope = temp @ (val_loss_store - val_loss_store.mean()).float()
                replacement_rate = self.compute_replacement_rate(
                    train_loss_slope,
                    val_loss_slope,
                    baseline=replacement_baseline,
                    tolerance=replacement_tolerance,
                )
                if replacement_rate > 0:
                    train_loader, val_loader = self.data_gen.replace(
                        replacement_rate * 1.0
                    )  # Replace a proportion of the training set
                    self.train_loader = train_loader
                    self.val_loader = val_loader
                    last_resampling = e
                    train_loss_store = torch.tensor([], device=self.device)
                    val_loss_store = torch.tensor([], device=self.device)
            else:
                train_loss_store = torch.cat(
                    (train_loss_store, torch.tensor([train_loss], device=self.device))
                )
                val_loss_store = torch.cat(
                    (val_loss_store, torch.tensor([val_loss], device=self.device))
                )

            train_loss = 0

            for i, data in enumerate(train_loader):
                self.optimizer.zero_grad()
                init, target = data

                if not self.need_control_trajectory:
                    x, error_trajectory = self.model(
                        init, target, control_period=self.control_period
                    )
                    control_trajectory = None
                else:
                    x, control_trajectory, error_trajectory, _ = self.model(
                        init,
                        target,
                        control_period=self.control_period,
                        return_control_trajectory=True,
                    )

                # norm_scalar = 1#e/epochs
                norm_scalar = (
                    (0.1 if e / epochs <= 0.2 else e / epochs)
                    if e / epochs <= 0.7
                    else 0.7
                )
                loss = self.loss_fn(
                    x,
                    error_trajectory,
                    control_trajectory=control_trajectory,
                    state_norm_weight=norm_scalar * 0.1,
                    control_norm_weight=norm_scalar * 0.1,
                    state_likelihood_weight=norm_scalar * 0.1,
                    params=self.model.parameters(),
                )
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()

                self.writer.add_scalar(
                    "training loss", loss.item(), i + e * len(train_loader)
                )

                self.loss_fn.log_components(i + e * len(train_loader), "training")
                train_loss += loss

            train_loss /= len(train_loader)

            if (e + 1) % save_interval == 0:
                if check_path is not None:
                    self.save_checkpoint(
                        check_path + "/checkpoint" + str(e) + ".pth", e, loss
                    )
                    self.latest_checkpoint = e

            if (e + 1) % reset_interval == 0 and e / epochs < 0.75:
                if self.scheduler is not None:
                    self.scheduler = LR.ReduceLROnPlateau(
                        self.optimizer,
                        mode="min",
                        factor=0.1,
                        patience=100,
                        min_lr=5e-4,
                        cooldown=50,
                    )
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.lr_init
                self.epochs_no_improve = 0

            if self.scheduler is None:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr_init

            if self.validate(val_loader, e, early_stop):
                break
            val_loss = self.val_loss
            if verbose:
                print(
                    f"Epoch {e}, training loss: {train_loss}, validation loss: {val_loss}"
                )
        self.save_model(save_path, full=True)

    def validate(self, val_loader, epoch, early_stop=False, epochs=1000):
        self.model.eval()
        val_total_loss = 0

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                init, target = data

                if not self.need_control_trajectory:
                    x, error_trajectory = self.model(
                        init, target, control_period=self.control_period
                    )
                    control_trajectory = None
                else:
                    x, control_trajectory, error_trajectory, _ = self.model(
                        init,
                        target,
                        control_period=self.control_period,
                        return_control_trajectory=True,
                    )

                norm_scalar = (
                    (0 if epoch / epochs <= 0.2 else epoch / epochs)
                    if epoch / epochs <= 0.7
                    else 0.7
                )
                val_loss = self.loss_fn(
                    x,
                    error_trajectory,
                    control_trajectory=control_trajectory,
                    state_norm_weight=norm_scalar * 0.1,
                    control_norm_weight=norm_scalar * 0.1,
                    state_likelihood_weight=norm_scalar * 0.1,
                    params=self.model.parameters(),
                )
                val_loss = val_loss.mean()
                self.loss_fn.log_components(epoch, "validation")
                val_total_loss += val_loss

        avg_val_loss = val_total_loss / len(val_loader)

        if self.scheduler is not None:
            self.scheduler.step(avg_val_loss)

        self.writer.add_scalar("validation loss", avg_val_loss, epoch)
        self.val_loss = avg_val_loss
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}, learning rate: {self.optimizer.param_groups[0]['lr']}, validation loss: {avg_val_loss}"
            )
        if avg_val_loss < self.best_val_loss_split:
            self.best_val_loss_split = avg_val_loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= 100 and early_stop:
                print("Early stopping")
                return 1
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.best_model = deepcopy(self.model.state_dict())
            self.save_checkpoint(
                self.check_path + "/best_model.pth", epoch, avg_val_loss
            )
        return 0

    def compute_replacement_rate(
        self, train_loss_slope, val_loss_slope, baseline=0, tolerance=0.5
    ):
        ratio = val_loss_slope / train_loss_slope
        replacement_rate = (
            min(1, max(baseline, 1 - ratio))
            if val_loss_slope > train_loss_slope and ratio < tolerance
            else 0
        )
        return replacement_rate

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

    def forward(self, features):
        print(
            "WARNING: Loss here is ELBO, not suitable right away with plotting functions"
        )
        self.model.eval()
        with torch.no_grad():
            x_hat, mu, log_var = self.model(features)
            loss = self.loss_fn(features, x_hat, mu, log_var)
        return x_hat, loss

    def data_generator(self, lean=False, custom_configs=None):
        if not lean:
            return dg.PredictorModelDataGenerator(
                self.dynamics_model,
                self.config if custom_configs is None else custom_configs,
            )
        else:
            return dg.LeanPredictorModelDataGenerator(
                self.dynamics_model,
                self.config if custom_configs is None else custom_configs,
            )

    def _model_type(self, model_specs, config=True):
        model_name = model_specs["model_name"]
        model = None
        try:
            module = __import__(f"{model_specs['model_path']}", fromlist=[model_name])
            ModelClass = getattr(module, model_name)
            if config:
                model = ModelClass(model_specs["model_params"])
            else:
                model = ModelClass(**model_specs["model_params"])
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
            optimizer = OptimizerClass(
                self.model.parameters(), **optimizer_specs["optimizer_params"]
            )
        except AttributeError:
            print(f"Optimizer {optimizer_name} not found in torch.optim")
        except Exception as e:
            print(f"Failed to initialize optimizer: {e}")

        return optimizer


class CustomLoss(nn.Module):
    def __init__(self, additional_loss, U, writer):
        super(CustomLoss, self).__init__()
        self.additional_loss = additional_loss
        print(self.additional_loss)
        if U is not None:
            if not isinstance(U, torch.Tensor):
                U = torch.tensor(U, device=torch.get_default_device())
            self.U2_inv = (U @ U.T).inverse()
        else:
            self.U2_inv = None
        self.state_regs_loss = None
        self.control_regs_loss = None
        self.state_likelihood = None
        self.final_distance = None

        self.writer = writer

    def forward(
        self,
        x,
        error_trajectory,
        control_trajectory=None,
        state_norm_weight=0.1,
        control_norm_weight=0.1,
        state_likelihood_weight=1.0,
        control_likelihood_weight=0.1,
        params=None,
    ):
        loss = error_trajectory.mean(dim=-1)
        for loss_fn in self.additional_loss:
            match loss_fn:
                case "state_norm":
                    # print(f"state loss weight: {state_norm_weight}")
                    self.state_regs_loss = (
                        state_norm_weight * x.norm(p=2, dim=-1).mean()
                    )
                    loss += self.state_regs_loss

                case "control_norm":
                    # print(f"control loss weight: {control_norm_weight}")
                    if control_trajectory is not None:
                        self.control_regs_loss = (
                            control_norm_weight
                            * control_trajectory.norm(p=2, dim=-1).mean()
                        )
                        loss += self.control_regs_loss
                    else:
                        raise ValueError("Control trajectory not provided")
                case "state_likelihood":
                    assert self.U2_inv is not None, "U2_inv not provided"
                    # print(f"x shape: {x.shape}")
                    # print(f"U2_inv shape: {self.U2_inv.shape}")
                    # print(f"x_t shape: {x.T.shape}")
                    # print(f"state likelihood weight: {state_likelihood_weight}")
                    x = x.reshape(-1, x.shape[-1])
                    self.state_likelihood = (
                        state_likelihood_weight * (x @ self.U2_inv @ (x.T)).mean()
                    )
                    loss += self.state_likelihood
                case "final_distance":
                    self.final_distance = error_trajectory[:, -1]
                    loss += self.final_distance

                case "weight_decay":
                    assert params is not None, "params not provided"
                    for param in params:
                        loss += 0.05 * param.norm(p=2) ** 2
        return loss

    def log_components(self, epoch, phase):
        if self.state_regs_loss is not None:
            self.writer.add_scalar(
                f"{phase} state regularization loss", self.state_regs_loss, epoch
            )
        if self.control_regs_loss is not None:
            self.writer.add_scalar(
                f"{phase} control regularization loss", self.control_regs_loss, epoch
            )
        if self.state_likelihood is not None:
            self.writer.add_scalar(
                f"{phase} state log likelihood", -self.state_likelihood, epoch
            )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class CustomLoss_PredControl(nn.Module):
    def __init__(self, additional_loss, U, writer):
        super(CustomLoss, self).__init__()
        self.additional_loss = additional_loss
        print(self.additional_loss)
        if U is not None:
            if not isinstance(U, torch.Tensor):
                U = torch.tensor(U, device=torch.get_default_device())
            self.U2_inv = (U @ U.T).inverse()
        else:
            self.U2_inv = None
        self.state_regs_loss = None
        self.control_regs_loss = None
        self.state_likelihood = None
        self.final_distance = None
        self.action_recovery_loss = None
        self.action_magnitude_loss = None
        self.action_correction_loss = None
        self.action_pred_loss = None
        self.writer = writer

    def forward(
        self,
        x,
        error_trajectory, # error trajectory in CAN state space
        control_trajectory=None, # control trajectory from CAN to plant state space
        action_trajectory=None, # action trajectory inferred by the internal model
        action_recovery_error_trajectory=None, # error in recovering the action
        action_correction_trajectory=None, # correction to the action
        out_trajectory=None, # actual output trajectory
        state_norm_weight=0.1,
        control_norm_weight=0.1,
        state_likelihood_weight=1.0,
        action_weight=0.1,
        action_recovery_weight=0.1,
        action_correction_weight=0.1,
        control_likelihood_weight=0.1,
        params=None,
    ):
        loss = error_trajectory.mean(dim=-1)
        for loss_fn in self.additional_loss:
            match loss_fn:
                case "state_norm":
                    # print(f"state loss weight: {state_norm_weight}")
                    self.state_regs_loss = (
                        state_norm_weight * x.norm(p=2, dim=-1).mean()
                    )
                    loss += self.state_regs_loss

                case "control_norm":
                    # print(f"control loss weight: {control_norm_weight}")
                    if control_trajectory is not None:
                        self.control_regs_loss = (
                            control_norm_weight
                            * control_trajectory.norm(p=2, dim=-1).mean()
                        )
                        loss += self.control_regs_loss
                    else:
                        raise ValueError("Control trajectory not provided")
                case "state_likelihood":
                    assert self.U2_inv is not None, "U2_inv not provided"
                    # print(f"x shape: {x.shape}")
                    # print(f"U2_inv shape: {self.U2_inv.shape}")
                    # print(f"x_t shape: {x.T.shape}")
                    # print(f"state likelihood weight: {state_likelihood_weight}")
                    x = x.reshape(-1, x.shape[-1])
                    self.state_likelihood = (
                        state_likelihood_weight * (x @ self.U2_inv @ (x.T)).mean()
                    )
                    loss += self.state_likelihood
                case "final_distance":
                    self.final_distance = error_trajectory[:, -1]
                    loss += self.final_distance

                case "weight_decay":
                    assert params is not None, "params not provided"
                    for param in params:
                        loss += 0.05 * param.norm(p=2) ** 2
        # the other losses are mandatory
        if action_trajectory is not None:
            self.action_magnitude_loss = action_trajectory.norm(p=2, dim=-1).mean()
            loss += action_weight * self.action_pred_loss
        if action_recovery_error_trajectory is not None:
            self.action_recovery_loss = action_recovery_error_trajectory.mean()
            loss += action_recovery_weight * self.action_recovery_loss
        if action_correction_trajectory is not None:
            self.action_correction_loss = action_correction_trajectory.mean()
            loss += action_correction_weight * self.action_correction_loss
        if out_trajectory is not None:
            self.action_pred_loss = (out_trajectory - action_trajectory).norm(p=2, dim=-1).mean()
        return loss

    def log_components(self, epoch, phase):
        dictionary = {
            "state regularization loss": self.state_regs_loss,
            "control regularization loss": self.control_regs_loss,
            "state log likelihood": -self.state_likelihood if self.state_likelihood is not None else None,
            "final distance": self.final_distance,
            "action magnitude loss": self.action_magnitude_loss,
            "action recovery loss": self.action_recovery_loss,
            "action correction loss": self.action_correction_loss,
            "action prediction loss": self.action_pred_loss,
        }
        for key, value in dictionary.items():
            if value is not None:
                self.writer.add_scalar(f"{phase} {key}", value, epoch)
        # if self.state_regs_loss is not None:
        #     self.writer.add_scalar(
        #         f"{phase} state regularization loss", self.state_regs_loss, epoch
        #     )
        # if self.control_regs_loss is not None:
        #     self.writer.add_scalar(
        #         f"{phase} control regularization loss", self.control_regs_loss, epoch
        #     )
        # if self.state_likelihood is not None:
        #     self.writer.add_scalar(
        #         f"{phase} state log likelihood", -self.state_likelihood, epoch
        #     )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
