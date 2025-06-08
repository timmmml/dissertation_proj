"""This module wraps the specifications of the training process for the agents


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
import SimulateDatasets.RandomRenderer as rd
from importlib import reload
reload(rd)

def reinitialise_weights(model):
    for layer in model.children():
        if not hasattr(layer, 'frozen') or not layer.frozen:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            elif isinstance(layer, nn.Sequential) or isinstance(layer, nn.ModuleList) or isinstance(layer, nn.Module):
                reinitialise_weights(layer)

class VaeTrainer:
    def __init__(self, config=None):
        if config is None:
            config = self._default_config()
        self.config = config
        self.save_path = config["save_path"]
        self.check_path = config.get("check_path", None)
        self.log_path = config["log_path"]

        self.reconstruction_loss = self._reconstruction_loss_functions(config["reconstruction_loss"])
        self.kl_divergence_loss = self._KL_divergence_loss_functions(config["kl_divergence_loss"])

        self.model_specs = config["model_specs"]  # Should be a dictionary
        self.model = self._model_type(self.model_specs)

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

        self.loss_fn = ELBO(self.reconstruction_loss, self.kl_divergence_loss, config.get("annealing", False), config.get("annealing_function", None))

        self.model = self.model.to(self.device)
        for module in self.model.modules():
            module = module.to(self.device)
        self.model.device = self.device

        self.best_val_loss = float("inf")
        self.best_val_loss_split = float("inf")
        self.best_model = None

    def refresh(self):
        reinitialise_weights(self.model)
        self.model.out = None
        self.optimizer = self._optimizer_type(self.optimizer_specs)
        self.scheduler = LR.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=100, min_lr=5e-4, cooldown=50)
        self.model.to(self.device)

    def train(
        self, train_loader, val_loader, epochs=10, save_interval=10, reset_interval = 200, save_path=None, check_path=None, log_path=None, early_stopping = False):

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
            for i, (data, labels) in enumerate(train_loader):
                # Labels are not used in the VAE
                data += torch.randn_like(data) * 0.01
                data = data.to(self.device)
                self.optimizer.zero_grad()
                x_hat, mu, log_var = self.model(data)
                loss = self.loss_fn(data, x_hat, mu, log_var, e, 40000)
                # loss = loss.mean()
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar("training loss", loss, i + e * len(train_loader))

                # Log additional loss components
                self.writer.add_scalar("training reconstruction loss", self.loss_fn.current_reconstruction_loss, i + e * len(train_loader))
                self.writer.add_scalar("training KL divergence loss", self.loss_fn.current_kl_divergence_loss, i + e * len(train_loader))

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

            if self.validate(val_loader, e, 40000, early_stopping=early_stopping):
                break
        self.save_model(save_path)
    
    # Add a train_unlimited function here. long overdue!
    def train_unlimited(self, **kwargs):
        # For DataGenerator, we use the trainer's home data generator
        data_loader_configs = kwargs.get("data_loader_configs", None)
        self.data_gen = self.data_generator(
            custom_configs=data_loader_configs
        )
        self.data_gen.save_path = kwargs.get("data_save_path", None)
        epochs = kwargs.get("epochs", 1000)
        save_interval = kwargs.get("save_interval", 10)
        reset_interval = kwargs.get("reset_interval", 200)
        save_path = kwargs.get("save_path", self.save_path)
        check_path = kwargs.get("check_path", self.check_path)
        log_path = kwargs.get("log_path", self.log_path)
        early_stop = kwargs.get("early_stop", False)
        verbose = kwargs.get("verbose", True)
        load_indices = kwargs.get("load_index", {"train": 0, "val": 0})
        save_indices = kwargs.get("store_index", {"train": 0, "val": 0})

        self.writer = SummaryWriter(log_path)
        self.epochs_no_improve = 0
        self.best_val_loss_split = float("inf")

        batch_size = self.config.get("training_config", {}).get("batch_size", 128)
        mini_batch_size = self.config.get("training_config", {}).get(
            "mini_batch_size", 32
        )
        self.data_gen.mini_batch_size = mini_batch_size
        self.data_gen.load_index = load_indices
        self.data_gen.store_index = save_indices
        val_size = round(
            self.config.get("training_config", {}).get("val_ratio", 0.1) * batch_size
        )
        replacement_baseline = self.config.get("training_config", {}).get(
            "replacement_baseline", 0
        )
        replacement_tolerance = self.config.get("training_config", {}).get(
            "replacement_tolerance", 0.5
        )
        epochs_to_calculate = self.config.get("training_config", {}).get(
            "epochs_to_calculate", 5
        )
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
                # print(f"Train loss slope: {train_loss_store}")
                # print(f"Val loss slope: {val_loss_store}")
                train_loss_slope = temp @ (train_loss_store - train_loss_store.mean())
                val_loss_slope = temp @ (val_loss_store - val_loss_store.mean())
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
                # Let's only use images here.
                data = data.to(self.device)
                self.optimizer.zero_grad()
                x_hat, mu, log_var = self.model(data)

                loss = self.loss_fn(
                    data, x_hat, mu, log_var, e, 40000
                )
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar("training loss", loss, i + e * len(train_loader))

                # Log additional loss components
                self.writer.add_scalar("training reconstruction loss", self.loss_fn.current_reconstruction_loss, i + e * len(train_loader))
                self.writer.add_scalar("training KL divergence loss", self.loss_fn.current_kl_divergence_loss, i + e * len(train_loader))
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

            if self.validate(val_loader, e, 40000, early_stop):
                break
            val_loss = self.val_loss
            if verbose:
                print(
                    f"Epoch {e}, training loss: {train_loss}, validation loss: {val_loss}"
                )
        self.save_model(save_path)

    def validate(self, val_loader, epoch, epochs = 0, early_stopping = False):
        self.model.eval()
        val_total_loss = 0
        val_reconstruction_loss = 0
        val_kl_loss = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                data += torch.randn_like(data) * 0.01
                # print(data.shape)
                # data.permute(0,3,1,2)
                data = data.to(self.device)
                x_hat, mu, log_var = self.model(data)
                val_loss = self.loss_fn(data, x_hat, mu, log_var, epoch, epochs)
                val_total_loss += val_loss
                val_reconstruction_loss += self.loss_fn.current_reconstruction_loss
                val_kl_loss += self.loss_fn.current_kl_divergence_loss

            avg_val_loss = val_total_loss / len(val_loader)
            avg_val_reconstruction_loss = val_reconstruction_loss / len(val_loader)
            avg_val_kl_loss = val_kl_loss / len(val_loader)

            if self.scheduler is not None:
                self.scheduler.step(avg_val_loss)

            self.writer.add_scalar("validation loss", avg_val_loss, epoch)

            # Log additional loss components
            self.writer.add_scalar("validation reconstruction loss", avg_val_reconstruction_loss, epoch)
            self.writer.add_scalar("validation KL divergence loss", avg_val_kl_loss, epoch)
            # print(f"Epoch {epoch} validation loss: {avg_val_loss}, reconstruction loss: {avg_val_reconstruction_loss}, KL loss: {avg_val_kl_loss}")

            if avg_val_loss < self.best_val_loss_split:
                self.best_val_loss_split = avg_val_loss
                self.epochs_no_improve = 0
            elif early_stopping:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= 100:
                    print("Early stopping")
                    self.val_loss = avg_val_loss
                    return(1)
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.best_model = deepcopy(self.model.state_dict())
                self.save_checkpoint(self.check_path + "/best_model.pth", epoch, avg_val_loss)
            self.val_loss = avg_val_loss
            return(0)

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
        print("WARNING: Loss here is ELBO, not suitable right away with plotting functions")
        self.model.eval()
        with torch.no_grad():
            x_hat, mu, log_var = self.model(features)
            loss = self.loss_fn(features, x_hat, mu, log_var)
        return x_hat, loss
    
    def data_generator(self, custom_configs=None):
        return rd.RandomRenderer(
            self.config if custom_configs is None else custom_configs,
        )

    def _model_type(self, model_specs):
        model_name = model_specs["model_name"]
        model = None
        try:
            module = __import__(f"{model_specs['model_path']}", fromlist=[model_name])
            ModelClass = getattr(module, model_name)
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

    def _reconstruction_loss_functions(self, loss_name):
        match loss_name:
            case "BCE":
                return nn.BCELoss(reduction="sum")
            case "MSE":
                return nn.MSELoss(reduction="sum")
            case _:
                raise NotImplementedError

    def _KL_divergence_loss_functions(self, loss_name = "Normal"):
        match loss_name:
            case "Normal": # default; this is against a standard normal distribution
                return lambda mean, log_var: -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            case "KL_mean":
                return nn.KLDivLoss(reduction="mean")
            case "KL_batchmean":
                return nn.KLDivLoss(reduction="batchmean")
            case "KL_sum":
                return nn.KLDivLoss(reduction="sum")
            case "NO_KL":
                return lambda mean, log_var: torch.tensor(0, device = "cuda")
            case _:
                raise NotImplementedError

class ELBO(nn.Module):
    """Custom ELBO; takes a reconstruction loss and a KL divergence loss"""
    def __init__(self, reconstruction_loss, kl_divergence_loss, annealing = False, annealing_function = None):
        super(ELBO, self).__init__()
        self.reconstruction_loss = reconstruction_loss
        self.kl_divergence_loss = kl_divergence_loss
        self.current_reconstruction_loss = 0
        self.current_kl_divergence_loss = 0
        self.annealing = annealing
        self.annealing_function = Annealing_functions(annealing_function).function

    def forward(self, x, x_hat, mu, log_var, e = None, E = None):
        self.current_reconstruction_loss = self.reconstruction_loss(x, x_hat)
        self.current_kl_divergence_loss = self.kl_divergence_loss(mu, log_var)
        # print(mu.shape)
        # print(log_var.shape)
        if self.annealing:
            return self.current_reconstruction_loss + self.annealing_function(e, E) * self.current_kl_divergence_loss
        else:
            return self.current_kl_divergence_loss + self.current_reconstruction_loss

class Annealing_functions:
    def __init__(self, function_name):
        match function_name:
            case "constant":
                self.function = lambda e, E: 1/E
            case "linear":
                self.function = lambda e, E: 1e-3 + e/E
            case "exp":
                self.function = lambda e, E: 1e-3 + torch.exp(torch.tensor(e - E, device = "cuda"))

