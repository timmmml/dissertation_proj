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
import numpy as np

from importlib import reload

reload(dg)

def expand_trig(theta):
    """
    theta: torch.tensor(batch_size, dim): thetas of different dimensions
    """
    return torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)

def pos2theta(y): 
    dim = y.shape[-1]//2
    theta = torch.zeros((*y.shape[:-1], dim))
    for i in range(dim):
        theta[..., i] = torch.atan2(y[..., i + dim], y[..., i])
    return theta



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

class CTrainer_Pred_CAN:
    def __init__(self, config=None):
        assert config is not None, "No configuration provided"
        self.config = config
        self.save_path = config["save_path"]
        self.check_path = config.get("check_path", None)
        self.log_path = config["log_path"]

        self.model_specs = config["model_specs"]  # Should be a dictionary
        self.model = self._model_type(self.model_specs, config=False)

        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

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

        self.loss_fn = CustomLoss_PredControl(config.get("additional_loss", []), None, writer=None)
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
        self.training_loss_dictionary = None

        self.val_loss_dictionary = None


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
        # self.optimizer = self._optimizer_type(self.optimizer_specs)
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
        action_weight = kwargs.get("action_weight", 0.0)
        action_recovery_weight = kwargs.get("action_recovery_weight", 0)  # This can also be the prediction loss
        convergence_loss_weight = kwargs.get("convergence_loss_weight", 0)
        loss_traj_weight = kwargs.get("loss_traj_weight", 0)
        weight_decay = kwargs.get("weight_decay", 0.00)

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
        reset_dictionary = kwargs.get("reset_dictionary", True)

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

        if reset_dictionary or self.training_loss_dictionary is None:
            self.training_loss_dictionary = {
                "control norm loss": [],
                "final distance (neural)": [],
                "inferrred action magnitudes": [],
                "state prediction loss": [],  # Euclidean
                "action prediction loss": [],  # match inferred actions to actual actions
                "weight decay loss": [],
                "final arc distance": [], 
                "average neural distance": [],
            }

            self.val_loss_dictionary = {
                "control norm loss": [],
                "final distance (neural)": [],
                "inferrred action magnitudes": [],
                "state prediction loss": [],  # Euclidean
                "action prediction loss": [],  # match inferred actions to actual actions
                "weight decay loss": [],
                "final arc distance": [], 
                "average neural distance": [],
            }

        for e in range(epochs):

            local_dict = {
                "control norm loss": 0, 
                "final distance (neural)": 0,
                "inferrred action magnitudes": 0,
                "state prediction loss": 0, # Euclidean
                "action prediction loss": 0, # match inferred actions to actual actions
                "weight decay loss": 0,
                "final arc distance": 0,
                "average neural distance": 0,
            }
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

                # self.optimizer.zero_grad()
                init, target = data
                # if i == 0: 
                    # print(f"init readout {self.model.readoutnet(init[..., -self.model.rep_net.rnn.rep_units:])[0:5]}")
                    # print(f"target readout {self.model.readoutnet(target[..., -self.model.rep_net.rnn.rep_units:])[0:5]}")
                if kwargs.get("s0_and_observe", False): 
                    s0 = self.model.readoutnet(init[..., -self.model.rep_net.rnn.rep_units:])
                    use_s0 = True
                    # shuffle init such that s0 is the only thing that is useful
                    init = init[torch.randperm(init.size(0))]
                else: 
                    s0 = None
                    use_s0 = False

                # print(use_s0)
                x, control_trajectory, error_trajectory, _, _, _, action_trajectory, action_recovery_error_trajectory, action_correction_trajectory, out_trajectory = self.model(
                    init,
                    target,
                    control_period=self.control_period,
                    return_control_trajectory=True,
                    use_s0 = use_s0,
                    s0 = s0 if use_s0 else None,
                    observe = 2 if use_s0 else 0,
                )

                # norm_scalar = (
                #     (0.1 if epoch / epochs <= 0.2 else epoch / epochs)
                #     if epoch / epochs <= 0.7
                #     else 0.7
                # )
                norm_scalar = kwargs.get("control_norm_weight", 0.1)
                loss = self.loss_fn(
                    x,
                    error_trajectory,
                    control_trajectory=control_trajectory,
                    action_trajectory=action_trajectory,
                    action_recovery_error_trajectory=action_recovery_error_trajectory,
                    action_correction_trajectory=action_correction_trajectory,
                    out_trajectory=out_trajectory,
                    state_norm_weight=norm_scalar,
                    control_norm_weight=norm_scalar,
                    state_likelihood_weight=norm_scalar,
                    action_weight = action_weight,
                    action_recovery_weight=action_recovery_weight,
                    convergence_loss_weight=convergence_loss_weight,
                    loss_traj_weight=loss_traj_weight,
                    weight_decay = weight_decay,
                    compartmental_optimization=self.compartmental_optimization,
                    controller_params = self.optimizer["controller"].param_groups[0]["params"] if self.compartmental_optimization else None,
                    internal_model_params = self.optimizer["internal_model_and_predictor"].param_groups[0]["params"] if self.compartmental_optimization else None,
                )

                self.step_loss(loss)
                loss = loss.mean() if not isinstance(loss, dict) else sum(v for v in loss.values())
                with torch.no_grad():
                    s0 = self.model.readoutnet(x[..., -1, -self.model.rep_net.rnn.rep_units:])
                    s1 = self.model.readoutnet(target[..., -self.model.rep_net.rnn.rep_units:])
                q0 = expand_trig(s0) 
                q1 = expand_trig(s1)
                del s0
                del s1
                arc_distance = (q0 - q1).norm(p=2, dim=-1).mean()
                
                local_dict["final arc distance"] += arc_distance
                local_dict["control norm loss"] += self.loss_fn.control_regs_loss.item()
                local_dict["final distance (neural)"] += self.loss_fn.final_distance.item()
                local_dict["inferrred action magnitudes"] += self.loss_fn.action_magnitude_loss.item()
                local_dict["state prediction loss"] += self.loss_fn.action_recovery_loss.item()
                local_dict["action prediction loss"] += self.loss_fn.action_pred_loss.item()
                local_dict["weight decay loss"] += self.loss_fn.weight_decay_loss.item()
                local_dict["average neural distance"] += self.loss_fn.traj_loss.item()

                self.writer.add_scalar(
                    "training loss", loss.item(), i + e * len(train_loader)
                )

                self.loss_fn.log_components(i + e * len(train_loader), "training")
                train_loss += loss

            for key in local_dict.keys():
                local_dict[key] /= len(train_loader)
                self.training_loss_dictionary[key].append(local_dict[key])
                
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
                if not isinstance(self.optimizer, dict):
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.lr_init
                else:
                    for key in self.optimizer.keys():
                        for param_group in self.optimizer[key].param_groups:
                            param_group["lr"] = self.lr_init

            if self.validate(val_loader, e, early_stop, kwargs=kwargs):
                break
            val_loss = self.val_loss
            if verbose:
                print(
                    f"Epoch {e}, training loss: {train_loss}, validation loss: {val_loss}"
                )
        self.save_model(save_path)

    def validate(self, val_loader, epoch, early_stop=False, epochs=1000, kwargs={}):
        self.model.eval()
        val_total_loss = 0
        action_weight = kwargs.get("action_weight", 0.0)
        action_recovery_weight = kwargs.get("action_recovery_weight", 0)  # This can also be the prediction loss
        convergence_loss_weight = kwargs.get("convergence_loss_weight", 0)
        loss_traj_weight = kwargs.get("loss_traj_weight", 0)

        local_dict = {
            "control norm loss": 0, 
            "final distance (neural)": 0,
            "inferrred action magnitudes": 0,
            "state prediction loss": 0, # Euclidean
            "action prediction loss": 0, # match inferred actions to actual actions
            "weight decay loss": 0,
            "final arc distance": 0,
            "average neural distance": 0,
        }
        with torch.no_grad():
            for i, data in enumerate(val_loader):

                init, target = data
                # if i == 0: 
                #     print(f"init readout {self.model.readoutnet(init[..., -self.model.rep_net.rnn.rep_units:])[0:5]}")
                #     print(f"target readout {self.model.readoutnet(target[..., -self.model.rep_net.rnn.rep_units:])[0:5]}")
                if kwargs.get("s0_and_observe", False): 
                    s0 = self.model.readoutnet(init[..., -self.model.rep_net.rnn.rep_units:])
                    use_s0 = True
                    # shuffle init such that s0 is the only thing that is useful
                    init = init[torch.randperm(init.size(0))]
                else: 
                    s0 = None
                    use_s0 = False

                x, control_trajectory, error_trajectory, _, _, _, action_trajectory, action_recovery_error_trajectory, action_correction_trajectory, out_trajectory = self.model(
                    init,
                    target,
                    control_period=self.control_period,
                    return_control_trajectory=True,
                    use_s0 = use_s0,
                    s0 = s0 if use_s0 else None,
                    observe = 2 if use_s0 else 0,
                )

                # norm_scalar = (
                #     (0.1 if epoch / epochs <= 0.2 else epoch / epochs)
                #     if epoch / epochs <= 0.7
                #     else 0.7
                # )
                norm_scalar = kwargs.get("control_norm_weight", 0.1)
                val_loss = self.loss_fn(
                    x,
                    error_trajectory,
                    control_trajectory=control_trajectory,
                    action_trajectory=action_trajectory,
                    action_recovery_error_trajectory=action_recovery_error_trajectory,
                    action_correction_trajectory=action_correction_trajectory,
                    out_trajectory=out_trajectory,
                    state_norm_weight=norm_scalar,
                    control_norm_weight=norm_scalar,
                    state_likelihood_weight=norm_scalar,
                    action_weight = action_weight,
                    action_recovery_weight=action_recovery_weight,
                    convergence_loss_weight=convergence_loss_weight,
                    loss_traj_weight=loss_traj_weight,
                    compartmental_optimization=self.compartmental_optimization, 
                    weight_decay = kwargs.get("weight_decay", 0.00),
                    controller_params = self.optimizer["controller"].param_groups[0]["params"] if self.compartmental_optimization else None,
                    internal_model_params = self.optimizer["internal_model_and_predictor"].param_groups[0]["params"] if self.compartmental_optimization else None,
                )
                val_loss = val_loss.mean() if not isinstance(val_loss, dict) else sum(v for v in val_loss.values())

                with torch.no_grad():
                    s0 = self.model.readoutnet(x[..., -1, -self.model.rep_net.rnn.rep_units:])
                    s1 = self.model.readoutnet(target[..., -self.model.rep_net.rnn.rep_units:])

                q0 = expand_trig(s0) 
                q1 = expand_trig(s1)
                del s0
                del s1
                arc_distance = (q0 - q1).norm(p=2, dim=-1).mean()
                
                local_dict["final arc distance"] += arc_distance
                local_dict["control norm loss"] += self.loss_fn.control_regs_loss.item()
                local_dict["final distance (neural)"] += self.loss_fn.final_distance.item()
                local_dict["inferrred action magnitudes"] += self.loss_fn.action_magnitude_loss.item()
                local_dict["state prediction loss"] += self.loss_fn.action_recovery_loss.item()
                local_dict["action prediction loss"] += self.loss_fn.action_pred_loss.item()
                local_dict["weight decay loss"] += self.loss_fn.weight_decay_loss.item()
                local_dict["average neural distance"] += self.loss_fn.traj_loss.item()

                self.loss_fn.log_components(epoch, "validation")
                val_total_loss += val_loss

            for key in local_dict.keys():
                local_dict[key] /= len(val_loader)
                self.val_loss_dictionary[key].append(local_dict[key])

        avg_val_loss = val_total_loss / len(val_loader)

        if self.scheduler is not None:
            self.scheduler.step(avg_val_loss)

        self.writer.add_scalar("validation loss", avg_val_loss, epoch)
        self.val_loss = avg_val_loss
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}, learning rate: {self.optimizer.param_groups[0]['lr'] if not isinstance(self.optimizer, dict) else list(self.optimizer.values())[0].param_groups[0]["lr"]}, validation loss: {avg_val_loss}"
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
            print(
                f"Validation loss improved from {self.best_val_loss} to {avg_val_loss}, saving model"
            )
            self.best_val_loss = avg_val_loss
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
        checkpoint = {
            "epoch": e,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": {
                key: opt.state_dict() for key, opt in self.optimizer.items()
            } if isinstance(self.optimizer, dict) else self.optimizer.state_dict(),
            "loss": loss,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if isinstance(self.optimizer, dict):
            for key in self.optimizer.keys():
                self.optimizer[key].load_state_dict(checkpoint["optimizer_state_dict"].get(key, {}))
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model.to(self.device)


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

        # try:
        module = __import__(f"{model_specs['model_path']}", fromlist=[model_name])
        ModelClass = getattr(module, model_name)
        if config:
            model = ModelClass(model_specs["model_params"])
        else:
            model = ModelClass(**model_specs["model_params"])
        print("Model loaded")
        # except Exception as e:
            # print(f"Failed to load model: {e}")
        return model

    def _optimizer_type(self, optimizer_specs):
        optimizer_name = optimizer_specs["optimizer_name"]
        compartmental_optimization = optimizer_specs.get("compartmental_optimization", False)
        # if compartmental_optimization: 
            # print("setting up separate optimizers for the controller and the (internal model + predictor)")
        self.compartmental_optimization = compartmental_optimization

        optimizer = None

        # Get the optimizer class from the optim module
        OptimizerClass = getattr(optim, optimizer_name)
        # Instantiate the optimizer with the model parameters and provided optimizer parameters
        if not compartmental_optimization:
            optimizer = OptimizerClass(
                self.model.parameters(), **optimizer_specs["optimizer_params"]
            )
        else: 
            controller_parameters = []
            internal_model_parameters = list(self.model.im1.parameters())
            predictor_parameters = list(self.model.pred.parameters())
            # NOTE no need to optimize the operator nor the plant. No need to optimize the rep_net either (or is there?)

            controller_module_names = optimizer_specs.get("controller_module_names", ["W_hh", "W_xh", "out", "ht0", "plant"])

            for controller_module in [getattr(self.model, name, None) for name in controller_module_names]:
                if controller_module is not None:
                    if not isinstance(controller_module, nn.Parameter):
                        controller_parameters += list(controller_module.parameters())
                    else:
                        controller_parameters.append(controller_module)
                    # print(f"controller module {controller_module} found")
            
            ht_im10 = getattr(self.model, "ht_im10", None)
            if ht_im10 is not None:
                internal_model_parameters.append(ht_im10)
            
            ht_pred0 = getattr(self.model, "ht_pred0", None)
            if ht_pred0 is not None:
                predictor_parameters.append(ht_pred0)

            optimizer = {
                "controller": OptimizerClass(
                    controller_parameters, **(optimizer_specs["optimizer_params"] | {"lr": optimizer_specs["optimizer_params"]["lr"] })
                ),
                "internal_model_and_predictor": OptimizerClass(
                    internal_model_parameters+predictor_parameters, **optimizer_specs["optimizer_params"]
                ),
            }

        # except AttributeError:
        #     print(f"Optimizer {optimizer_name} not found in torch.optim")
        # except Exception as e:
        #     print(f"Failed to initialize optimizer: {e}")
        self.optimizer = optimizer
        return optimizer
    
    def step_loss(self, loss):
        """Handles optimization supporting compartmental losses"""
        assert isinstance(self.optimizer, dict) == isinstance(loss, dict), "Optimizer and loss should have the same structure"
        # print(f"loss: {loss}")

        if isinstance(self.optimizer, dict):
            for i, key in enumerate(self.optimizer.keys()):
                self.optimizer[key].zero_grad()
                # torch.autograd.detect_anomaly(True)
                loss[key] = loss[key].clone()
                # if key == "controller":
                #     if np.random.rand() < 0.9:
                #         continue
                loss[key].backward(retain_graph = True)
            for key in self.optimizer.keys():
                self.optimizer[key].step()
        else:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

class CustomLoss_PredControl(nn.Module):
    def __init__(self, additional_loss, U, writer):
        super(CustomLoss_PredControl, self).__init__()
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
        action_weight=0.01,
        action_recovery_weight=0,
        convergence_loss_weight=0.02,
        loss_traj_weight=0.01,
        weight_decay = 0.05,
        action_pred_weight = 0.00,
        compartmental_optimization=False,
        controller_params = None, 
        internal_model_params = None, 
    ):
        
        if compartmental_optimization: 
            loss = {"controller": 0, "internal_model": 0, "predictor": 0}
        else: 
            loss = 0

        
        traj_loss = error_trajectory.mean()
        self.traj_loss = traj_loss
        traj_loss = traj_loss * loss_traj_weight
        state_regs_loss, control_regs_loss, state_likelihood, final_distance = 0, 0, 0, 0
        weight_decay_loss = 0
        if controller_params is not None:
            for param in controller_params:
                weight_decay_loss += param.norm(p=2) ** 2
        if internal_model_params is not None:
            for param in internal_model_params:
                weight_decay_loss += param.norm(p=2) ** 2
            
        self.weight_decay_loss = weight_decay_loss
        weight_decay_loss = weight_decay * weight_decay_loss

        for loss_fn in self.additional_loss:
            match loss_fn:
                case "control_norm":
                    # print(f"control loss weight: {control_norm_weight}")
                    if control_trajectory is not None:
                        self.control_regs_loss = (
                            control_trajectory.norm(p=2, dim=-1).mean())
                        control_regs_loss = self.control_regs_loss * control_norm_weight
                    else:
                        raise ValueError("Control trajectory not provided")

                case "final_distance":
                    self.final_distance = error_trajectory[:, -1].mean()
                    final_distance = self.final_distance * convergence_loss_weight

        
        action_magnitude_loss, action_recovery_loss, action_correction_loss, action_pred_loss = 0, 0, 0, 0

        # the other losses are mandatory
        if action_trajectory is not None:
            self.action_magnitude_loss = action_trajectory.norm(p=2, dim=-1).mean()
            action_magnitude_loss = action_weight * self.action_magnitude_loss
        if action_recovery_error_trajectory is not None:
            self.action_recovery_loss = action_recovery_error_trajectory.mean()
            action_recovery_loss = action_recovery_weight * self.action_recovery_loss # This is also the predictor's matching loss! 
        if action_correction_trajectory is not None:
            self.action_correction_loss = action_correction_trajectory.mean()  # This is the magnitude of state predictions
            # action_correction_loss = action_correction_weight * self.action_correction_loss
        if out_trajectory is not None:
            self.action_pred_loss = (out_trajectory - action_trajectory).norm(p=2, dim=-1).mean()
            action_pred_loss =action_pred_weight * self.action_pred_loss # This is for reference only! It is not used in the loss calculation

        if compartmental_optimization:
            loss["controller"] = traj_loss  + control_regs_loss + state_likelihood + final_distance + weight_decay_loss + action_magnitude_loss + weight_decay_loss
            loss["internal_model_and_predictor"] = action_recovery_loss + action_correction_loss + action_pred_loss + weight_decay_loss
        else: 
            loss = traj_loss  + control_regs_loss + state_likelihood + final_distance + weight_decay_loss + action_magnitude_loss + action_recovery_loss  + action_correction_loss + weight_decay_loss
        
        return loss

    def log_components(self, epoch, phase):
        dictionary_alt = {
            "control regularization loss": self.control_regs_loss,
            "state log likelihood": -self.state_likelihood if self.state_likelihood is not None else None,
            "final distance": self.final_distance,
            "action magnitude loss": self.action_magnitude_loss,
            "action recovery loss": self.action_recovery_loss,
            "action correction loss": self.action_correction_loss,
            "action prediction loss": self.action_pred_loss,
            "weight decay loss": self.weight_decay_loss,
        }
        dictionary = {
            "control norm loss": self.control_regs_loss,
            "final distance (neural)": self.final_distance,
            "inferrred action magnitudes": self.action_magnitude_loss,
            "state prediction loss": self.action_recovery_loss, # Euclidean
            "action prediction loss": self.action_pred_loss, # match inferred actions to actual actions
            "weight decay loss": self.weight_decay_loss,
            "average neural distance": self.traj_loss,
        }
        for key, value in dictionary.items():
            if value is not None:
                self.writer.add_scalar(f"{phase} {key}", value.mean(), epoch)


    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        
