"""This module contains the forward model's trainer and the various helper classes.

class: FMTrainer
    This class is used to train the forward model, and is a subclass of BaseNNAgent.

class: CombinedModel
    This class is used to combine the encoder, decoder, and predictor models into one.

class: GenericLoss
    This class is used to define the loss function for the forward model.

class: LeanPredictorModelDataGenerator
    This class is used to generate data for the forward model when rendering is not needed.

class: PredictorModelDataGenerator
    This class is used to generate data for the forward model when rendering is needed.
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


import numpy as np
import pandas as pd

import SimulateDatasets.PredictorModelDataGenerator as dg
from SimulateDatasets.PredictorDataset import PredictorDataset
from importlib import reload
reload(dg)

class FMTrainer:
    def __init__(self, config=None):
        if config is None:
            raise ValueError("Please provide a configuration file.")
        self.config = config
        self.save_path = config.get("save_path", None)
        self.check_path = config.get("check_path", None)
        self.log_path = config.get("log_path", None)
        # NOTE on losses
        # Another loss needs to be added for quality of the recognition model (need to avoid the trivial solution of converging predictions to 0 in all cases)
        # Ideas for this to be achieved:
        # Adding a loss term to encourage recognized values to be divergent. This includes (from loose to strict):
        #   option A: pretraining the recognition model
        #       choice a: constrain the output to be from 0 to 1, use some differentiable histogram method (do they exist?), supervise on distribution
        #       choice b: use contrastive learning - binning several inputs together in modalities and train on dissimilarity matrices.
        #           encourage different images to be different! in this case, easy to construct the target matrix by evaluating pairwise geodesic distance.
        #       choice c: pretrain the encoder as a part of an autoencoder to learn to reconstruct the original image
        #   option B: add a loss term to co-train these models.

        self.dynamics_model_specs = config.get("dynamics_model_specs", None)
        self.dynamics_model = self._model_type(self.dynamics_model_specs)

        self.encoder_specs = config.get("encoder_specs", None)
        self.encoder = self._model_type(self.encoder_specs)
        self.encoder_frozen = (
            self.encoder_specs.get("frozen", False)
            if self.encoder_specs is not None
            else False
        )

        self.decoder_specs = config.get("decoder_specs", None)
        self.decoder = self._model_type(self.decoder_specs)
        self.decoder_frozen = (
            self.decoder_specs.get("frozen", False)
            if self.decoder_specs is not None
            else False
        )

        self.predictor_specs = config.get("predictor_specs", None)
        self.predictor = self._model_type(self.predictor_specs)
        self.predictor_frozen = self.predictor_specs.get("frozen", False)

        self.optimizer_specs = config.get("optimizer_specs", None)

        self.loss = config.get("loss", None)
        self.device = config.get("device", "cuda")

        self.scheduler = None
        self.lr_init = self.optimizer_specs["optimizer_params"].get("lr", 1e-3)

        self.best_val_loss = float("inf")
        self.best_val_loss_split = float("inf")
        self.best_model = None
        ...

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

    def train_fm(self, **kwargs):
        # data_generator, task="pretrain", epochs=1000, save_interval=10, reset_interval=200, save_path=None, check_path=None, log_path=None, early_stop=False):
        """Train the forward model.

        Here, note that we default to a paradigm in which we can generate data on the go or use pre-generated data.

        We extend the data_generator objects here to load new data instead of generate when there is available.

        The data generator looks for pre-made data in a folder.
        """
        kwargs["loss"] = kwargs.get("loss", "Predictor-Geodesic")

        models_to_freeze = []
        if kwargs["loss"][:4] == "Pred":
            # only unfreeze the predictor
            models_to_freeze = ["predictor", None, None]
        elif kwargs["loss"][:2] == "AE":
            models_to_freeze = [None, "encoder", "decoder"]
        else:
            models_to_freeze = ["predictor", "encoder", "decoder"]

        for model in models_to_freeze:
            self.unfreeze(model)

        self.loss_fn = GenericLoss(kwargs["loss"])
        # Now let's implement the generic training loop (this style should really be used for all trainer objects, so be it sufficient to say the following)
        self.train(**kwargs)

        for model in models_to_freeze:
            self.freeze(model)

    def train(self, **kwargs):
        # For DataGenerator, we use the trainer's home data generator
        lean = self.loss_fn.loss in ["Predictor-Geodesic", "Predictor-GeoGradual"]
        data_loader_configs = kwargs.get("data_loader_configs", None)
        # print(data_loader_configs)
        self.data_gen = self.data_generator(
            lean=lean, custom_configs=data_loader_configs
        )
        self.data_gen.save_path = kwargs.get("data_save_path", None)
        if self.data_gen.encoding_model_specs is not None:
            if self.data_gen.encoding_model_specs['model_name'] == "PCAEncoder": 
                self.whitening = True
                # self.whitening_vector = (self.data_gen.encoding_model.s ** 0.5)[:self.data_gen.encoding_model.encoding_dim]
                # self.whitening_vector = (self.data_gen.encoding_model.s ** -0.5)[:self.data_gen.encoding_model.encoding_dim]
                self.whitening_vector = None
            else: 
                self.whitening = False
                self.whitening_vector = None
        else: 
            self.whitening = False
            self.whitening_vector = None

        models_to_freeze = []
        if kwargs["loss"][:4] == "Pred":
            # only unfreeze the predictor
            models_to_freeze = ["predictor", None, None]
        elif kwargs["loss"][:2] == "AE":
            models_to_freeze = [None, "encoder", "decoder"]
        else:
            models_to_freeze = ["predictor", "encoder", "decoder"]

        self.model = CombinedModel(
            self.get(models_to_freeze[0]),
            self.get(models_to_freeze[1]),
            self.get(models_to_freeze[2]),
            whitening_vector=self.whitening_vector
        )


        self.optimizer = self._optimizer_type(self.optimizer_specs)

        epochs = kwargs.get("epochs", 1000)
        save_interval = kwargs.get("save_interval", 10)
        reset_interval = kwargs.get("reset_interval", 200)
        save_path = kwargs.get("save_path", self.save_path)
        check_path = kwargs.get("check_path", self.check_path)
        log_path = kwargs.get("log_path", self.log_path)
        early_stop = kwargs.get("early_stop", False)
        verbose = kwargs.get("verbose", True)
        load_indices = kwargs.get("load_index", {"train": 0, "val": 0})
        load_upper = kwargs.get("load_upper", 44) # after this index is reached, we break the training loop.
        save_indices = kwargs.get("store_index", {"train": 0, "val": 0})

        self.writer = SummaryWriter(log_path)
        self.epochs_no_improve = 0
        self.best_val_loss_split = float("inf")

        batch_size = kwargs.get("batch_size", 128)
        mini_batch_size = kwargs.get(
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
                output = self.model(data)

                loss = self.loss_fn(
                    self.model.predicted, self.model.encoded, self.model.decoded
                )
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar(
                    "training loss", loss.item(), i + e * len(train_loader)
                )

                # Log additional loss components
                self.log_relevant(i + e * len(train_loader), phase="training")

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

    def validate(self, val_loader, epoch, early_stop=False):
        self.model.eval()
        val_total_loss = 0
        if epoch % 100 == 0:
            print(
                f"Epoch {epoch}, learning rate: {self.optimizer.param_groups[0]['lr']}"
            )
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                output = self.model(data)
                val_loss = self.loss_fn(
                    self.model.predicted, self.model.encoded, self.model.decoded
                )
                val_loss = val_loss.mean()
                val_total_loss += val_loss

            avg_val_loss = val_total_loss / len(val_loader)

            if self.scheduler is not None:
                self.scheduler.step(avg_val_loss)

            self.writer.add_scalar("validation loss", avg_val_loss, epoch)
            self.val_loss = avg_val_loss

            # Log additional loss components
            self.log_relevant(epoch, phase="validation")

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

    def log_relevant(self, epoch, phase="training"):
        # the trick is to log based on the current loss configuration
        loss = self.loss_fn.loss
        assert loss in [
            "Predictor-Geodesic",
            "Predictor-GeoGradual",
            "AE-MSE",
            "Predictor-MSE",
            "Predictor-MSEGradual",
            "Combined-MSE",
        ], "Loss function not recognised."
        assert isinstance(self.writer, SummaryWriter), "Writer not initialised."
        match loss:
            case "Predictor-Geodesic":
                self.writer.add_scalar(
                    f"{phase} predictor geodesic loss",
                    self.loss_fn.prediction_loss.mean().item(),
                    epoch,
                )
            case "Predictor-GeoGradual":
                assert hasattr(self.loss_fn, "final_geodesic_loss")
                self.writer.add_scalar(
                    f"{phase} predictor total geodesic loss",
                    self.loss_fn.prediction_loss.mean().item(),
                    epoch,
                )
                self.writer.add_scalar(
                    f"{phase} predictor geodesic loss",
                    self.loss_fn.final_geodesic_loss.mean().item(),
                    epoch,
                )
            case "Predictor-MSEGradual":
                assert hasattr(self.loss_fn, "final_MSE_loss")
                self.writer.add_scalar(
                    f"{phase} predictor total MSE loss",
                    self.loss_fn.prediction_loss.mean().item(),
                    epoch,
                )
                self.writer.add_scalar(
                    f"{phase} predictor MSE loss",
                    self.loss_fn.final_MSE_loss.mean().item(),
                    epoch,
                )
            case "AE-MSE":
                self.writer.add_scalar(
                    f"{phase} autoencoder reconstruction loss",
                    self.loss_fn.encoding_loss.mean().item(),
                    epoch,
                )
            case "Predictor-MSE":
                self.writer.add_scalar(
                    f"{phase} predictor MSE loss",
                    self.loss_fn.prediction_loss.mean().item(),
                    epoch,
                )
            case "Combined-MSE":
                self.writer.add_scalar(
                    f"{phase} predictor MSE loss",
                    self.loss_fn.prediction_loss.mean().item(),
                    epoch,
                )
                self.writer.add_scalar(
                    f"{phase} autoencoder reconstruction loss",
                    self.loss_fn.encoding_loss.mean().item(),
                    epoch,
                )

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

    def get(self, attribute=None):
        if attribute is None:
            return None
        assert type(attribute) == str, "Attribute must be a string if not None."
        return getattr(self, attribute, None)

    def freeze(self, model):
        if model is None:
            return
        setattr(self, f"{model}_frozen", True)
        target = getattr(self, model)
        for param in target.parameters(recurse=True):
            param.requires_grad = False
        target.frozen = True
        target.train()

        print(f"{model} is now frozen.")
        # print(f"Test passed: {bool(getattr(self, model).frozen)}")

    def unfreeze(self, model):
        if model is None:
            return
        setattr(self, f"{model}_frozen", False)
        target = getattr(self, model)
        for param in target.parameters(recurse=True):
            param.requires_grad = True
        target.frozen = False
        target.eval()

        print(f"{model} is now unfrozen.")
        # print(f"Test passed: {not getattr(self, model).frozen}")

    def _model_type(self, model_specs):
        if model_specs is None:
            return None
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
            optimizer = OptimizerClass(
                self.model.parameters(), **optimizer_specs["optimizer_params"]
            )
        except AttributeError as a:
            print(f"Optimizer {optimizer_name} not found in torch.optim: {a}")
        except Exception as e:
            print(f"Failed to initialize optimizer: {e}")

        return optimizer

class CombinedModel(nn.Module):
    def __init__(self, predictor, encoder, decoder, loss = "Predictor-Geodesic", whitening_vector = None):
        super(CombinedModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor
        self.loss = loss
        self.whitening_vector = whitening_vector
        self.action_period = self.predictor.silence_time if hasattr(
            self.predictor, "silence_time"
        ) else 1

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            z0, I, q = x
        else:
            z0, I, q = x.z0, x.I, x.q
        if self.predictor is not None:
            self.predicted = self.predictor(z0)
            if self.predicted.dim() == 3: 
                self.predicted = self.predicted[:, -self.action_period:, :]
        else:
            self.predicted = None

        if self.encoder is not None:
            self.encoded = self.encoder(I)
        else:
            if I.dim() == 3:
                self.encoded = I # NOTE: Hard-coded for now - I.dim is 3 for when we want to use quaternions, but let's change this later.
                if self.whitening_vector is not None: 
                    self.encoded = self.encoded * self.whitening_vector.unsqueeze(0).unsqueeze(0)
                    # print(f"shape of whitened encoded: {self.encoded.shape}")
            else:
                self.encoded = q # NOTE: data is directly pulled over!

        if self.decoder is not None:
            self.decoded = self.decoder(self.encoded)
        else:
            self.decoded = None

        return self.predicted, self.encoded, self.decoded


class GenericLoss(nn.Module):
    def __init__(self, loss):
        super(GenericLoss, self).__init__()
        self.loss = loss
        self.weight_pred = 1
        self.weight_enc = 1
        self.prediction_loss = None
        self.encoding_loss = None
        self.steps = None

    def forward(self, predicted, encoded, decoded):
        if self.loss == "AE-MSE":
            # We default here to care about the reconstruction loss, using the MSE loss function only
            if encoded.dim() == 3:
                encoded = encoded[:, -1, :]
            self.encoding_loss = F.mse_loss(decoded, encoded)
            return self.encoding_loss
        elif self.loss == "Predictor-Geodesic":
            # In this case, we default to interpret the encoded and decoded as quaternions
            if encoded.dim() == 3:
                encoded = encoded[:, -1, :]
            self.prediction_loss = 2 * torch.acos(
                torch.clamp(
                    torch.abs(torch.sum(predicted * encoded, dim=-1)),
                    -1 + 1e-6,
                    1 - 1e-6,
                )
            )
            return self.prediction_loss
        elif self.loss == "Predictor-GeoGradual":
            # WARNING: YOU CAN ONLY USE RECURRENT MODELS IN THESE!
            # IN THIS CASE: predicted is of shape (batch_size, seq_len, output_features)
            # NOTE: with quaternions, we use a one-size-fit-all data generation approach.
            # For rendered datasets, we can only afford to generate much shorter encoded streams.
            if encoded.dim() == 2:
                encoded = encoded.unsqueeze(1).repeat(1, predicted.shape[1], 1)
            else: 
                # step_size = encoded.shape[1] // predicted.shape[1]
                if self.steps is None: 
                    self.steps = torch.linspace(-1, encoded.shape[1] - 1, predicted.shape[1] + 1, dtype=torch.int, device=encoded.device)[1:]
                encoded = encoded[:, self.steps, :]
            loss_train = 2 * torch.acos(
                torch.clamp(
                    torch.abs(torch.sum(predicted * encoded, dim=-1)),
                    -1 + 1e-6,
                    1 - 1e-6,
                )
            )
            self.prediction_loss = torch.sum(loss_train, dim=-1)
            self.final_geodesic_loss = loss_train[:, -1]
            return self.prediction_loss
        elif self.loss == "Predictor-MSE":
            if encoded.dim() == 3:
                encoded = encoded[:, -1, :]
            loss_train = F.mse_loss(predicted, encoded, reduction="none").mean(dim = -1)
            self.prediction_loss = loss_train.sum(dim=-1)
            self.final_MSE_loss = loss_train[:, -1]
            return self.prediction_loss
        elif self.loss == "Predictor-MSEGradual": 
            if encoded.dim() == 2:
                encoded = encoded.unsqueeze(1).repeat(1, predicted.shape[1], 1)
            else: 
                if self.steps is None: 
                    self.steps = torch.linspace(-1, encoded.shape[1] - 1, predicted.shape[1] + 1, dtype=torch.int, device=encoded.device)[1:]
                encoded = encoded[:, self.steps, :]
                
            # print(f"Predicted shape: {predicted.shape}, Encoded shape: {encoded.shape}")
            loss_train = F.mse_loss(predicted, encoded, reduction="none").mean(dim = -1)
            # print(loss_train.shape)
            self.prediction_loss = loss_train.sum(dim=-1)
            self.final_MSE_loss = loss_train[:, -1]
            return self.prediction_loss

        elif self.loss == "Combined-MSE":
            if encoded.dim() == 3:
                encoded = encoded[:, -1, :]
            self.prediction_loss = F.mse_loss(predicted, encoded)
            self.encoding_loss = F.mse_loss(decoded, encoded)
            return (
                self.weight_enc * self.encoding_loss
                + self.weight_pred * self.prediction_loss
            )
        elif self.loss == "Contrastive":
            raise NotImplementedError(
                "This loss function is not yet implemented, will be here soon."
            )
        else:
            raise NotImplementedError("This loss function is not yet implemented.")


