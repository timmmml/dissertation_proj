"""This module wraps the specifications of the training process for the agents


"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from copy import deepcopy
from utils import goto_project_root
from torch.optim import lr_scheduler as LR



def reinitialise_weights(model):
    for layer in model.children():
        if not hasattr(layer, "frozen") or not layer.frozen:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
            elif isinstance(layer, nn.Sequential) or isinstance(layer, nn.ModuleList):
                reinitialise_weights(layer)


class CANTrainer:
    def __init__(self, config=None):
        if config is None:
            config = self._default_config()
        self.config = config
        self.save_path = config["save_path"]
        self.check_path = config.get("check_path", None)
        self.log_path = config["log_path"]

        self.model_specs = config["model_specs"]  # Should be a dictionary
        self.model = self._model_type(self.model_specs)

        self.device = config["device"]
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

        self.loss_fn = ContrastiveLoss(device = self.device, **config["loss_config"])
        self.model = self.model.to(self.device)
        for module in self.model.modules():
            module = module.to(self.device)
        self.model.device = self.device

        self.best_val_loss = float("inf")
        self.best_val_loss_split = float("inf")
        self.best_model = None
        self.recorded_distance_loss = 0
        self.recorded_regularisation_loss = 0

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

    def train(
        self,
        data_generator,
        epochs=10,
        save_interval=10,
        reset_interval=500,
        save_path=None,
        check_path=None,
        log_path=None,
        early_stop=False,
        noneg=True, 
    ):
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
        self.loss_fn.writer = self.writer
        self.epochs_no_improve = 0
        self.best_val_loss_split = float("inf")

        batch_size = self.config.get("training_config", {}).get("batch_size", 128)

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
            if e == 0 and data_generator.train_data is None:
                train_loader, val_loader = data_generator.generate(
                    n_samples=batch_size, val_size=val_size, record=True
                )
            elif e == 0: 
                assert data_generator.train_data is not None and data_generator.val_data is not None, "Data generator should have data stored"
                train_loader, val_loader = data_generator.retrieve()

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
                    print("replacing data")
                    train_loader, val_loader = data_generator.replace(
                        replacement_rate
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
            for i, batch in enumerate(train_loader):
                anchor, pos, neg, anchor_len, pos_len, neg_len = [b.to(self.device) for b in batch]
                self.optimizer.zero_grad()
                if anchor.dim() == 2:
                    anchor, pos, neg = anchor.unsqueeze(-1), pos.unsqueeze(-1), neg.unsqueeze(-1)
                if anchor.dim() == 4: 
                    anchor, pos, neg = anchor.squeeze(1), pos.squeeze(1), neg.squeeze(1)
                if noneg: 
                    anchor_emb, pos_emb = self.model(anchor), self.model(pos)
                    if self.loss_fn.conIso_weight != 0: 
                        loss = self.loss_fn.forward_noneg(anchor_emb, pos_emb, anchor_len, pos_len, self.model.parameters(), velocity_traj=anchor)
                    else:
                        loss = self.loss_fn.forward_noneg(anchor_emb, pos_emb, anchor_len, pos_len, self.model.parameters())
                else:
                    anchor_emb, pos_emb, neg_emb = self.model(anchor), self.model(pos), self.model(neg)
                    loss = self.loss_fn(anchor_emb, pos_emb, neg_emb, anchor_len, pos_len, neg_len, self.model.parameters())
                self.loss_fn.log_components(e + i / len(train_loader), "train")
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()
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

            if self.validate(val_loader, e, early_stop, noneg=noneg):
                break
            val_loss = self.val_loss
        self.save_model(save_path)

    def compute_replacement_rate(
        self, train_loss_slope, val_loss_slope, baseline=0, tolerance=0.5
    ):
        """Rationale: if validation loss is improving, we can afford to replace a smaller proportion of the training set.
        Else (if val loss is static/increasing), this is sign of overfitting and we should replace more.

        Args:
            - train_loss_slope: the slope of the training loss curve, sampled across the past 5 validation points
            - val_loss_slope: the slope of the validation loss curve, sampled across the past 5 points

        Returns:
            - replacement_rate: the proportion of the training set to replace
        """
        ratio = val_loss_slope / train_loss_slope
        replacement_rate = (
            min(1, max(baseline, 1 - ratio))
            if val_loss_slope > train_loss_slope and ratio < tolerance
            else 0
        )
        return replacement_rate

    def validate(self, val_loader, epoch, early_stop=False, noneg=True):
        self.model.eval()
        val_total_loss = 0
        if epoch % 100 == 0:
            print(
                f"Epoch {epoch}, learning rate: {self.optimizer.param_groups[0]['lr']}"
            )
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                anchor, pos, neg, anchor_len, pos_len, neg_len = [b.to(self.device) for b in batch]
                if anchor.dim() == 2:
                    anchor, pos, neg = anchor.unsqueeze(-1), pos.unsqueeze(-1), neg.unsqueeze(-1)
                if anchor.dim() == 4: 
                    anchor, pos, neg = anchor.squeeze(1), pos.squeeze(1), neg.squeeze(1)
                if noneg: 
                    anchor_emb, pos_emb = self.model(anchor), self.model(pos)
                    if self.loss_fn.conIso_weight != 0: 
                        val_loss = self.loss_fn.forward_noneg(anchor_emb, pos_emb, anchor_len, pos_len, self.model.parameters(), velocity_traj=anchor)
                    else:
                        val_loss = self.loss_fn.forward_noneg(anchor_emb, pos_emb, anchor_len, pos_len, self.model.parameters())
                else: 
                    anchor_emb, pos_emb, neg_emb = self.model(anchor), self.model(pos), self.model(neg)
                    val_loss = self.loss_fn(
                        anchor_emb, pos_emb, neg_emb, anchor_len, pos_len, neg_len, self.model.parameters()
                        )
                self.loss_fn.log_components(i + epoch * len(val_loader)
                                            , "val")

                val_loss = val_loss.item()
                val_total_loss += val_loss

            avg_val_loss = val_total_loss / len(val_loader)
            print(f"Epoch {epoch}, val loss: {avg_val_loss}, rep loss: {self.loss_fn.recorded_InfoNCE_loss}")


            if self.scheduler is not None:
                self.scheduler.step(avg_val_loss)

            self.val_loss = avg_val_loss

            self.loss_fn.log_components(epoch, "val")
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

    # def forward(self, features, labels):
    #     self.model.eval()
    #     with torch.no_grad():
    #         output = self.model(features)
    #         loss = self.loss_fn(
    #             output[:, -self.action_period :, :], self.model.pred, labels
    #         )
    #     return output, loss

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
            optimizer = OptimizerClass(
                self.model.parameters(), **optimizer_specs["optimizer_params"]
            )
        except AttributeError:
            print(f"Optimizer {optimizer_name} not found in torch.optim")
        except Exception as e:
            print(f"Failed to initialize optimizer: {e}")

        return optimizer

    def _loss_functions(self, loss_name):
        match loss_name.lower():
            case "contrastiveloss":
                return ContrastiveLoss(**self.config["loss_config"])

class ContrastiveLoss(nn.Module):
    """Implements the combined loss function"""

    def __init__(self, **kwargs):
        super(ContrastiveLoss, self).__init__()
        loss_type, loss_temperature, normalize, writer, triplet_weight, weight_decay_scalar, weight_decay_thresh, random_offset, L2norm_scalar, similarity_kernel, invariance_weight, capacity_weight, conIso_weight, device = (
            kwargs.get("loss_type", "combined"),
            kwargs.get("loss_temperature", 0.7),
            kwargs.get("normalize", True),
            kwargs.get("writer", None),
            kwargs.get("triplet_weight", 0.5),
            kwargs.get("weight_decay_scalar", 0),
            kwargs.get("weight_decay_thresh", -0.1), # threshold for the other losses to initiate weight decay
            kwargs.get("random_offset", True),
            kwargs.get("L2_norm_scalar", 0),
            kwargs.get("similarity_kernel", "cosine"),
            kwargs.get("invariance_weight", 1),
            kwargs.get("capacity_weight", 0),
            kwargs.get("conIso_weight", 0),
            kwargs.get("device", "cuda"),
        )
        self.loss_type = loss_type
        self.normalize = normalize
        self.writer = writer
        self.triplet_weight = triplet_weight
        self.weight_decay_scalar = weight_decay_scalar
        self.weigth_decay_thresh = weight_decay_thresh
        self.random_offset = random_offset # randomly select the embedding position to evaluate
        self.L2_norm_scalar = L2norm_scalar if L2norm_scalar is not None else 0
        self.similarity_kernel = similarity_kernel if similarity_kernel is not None else "cosine"
        self.invariance_weight = invariance_weight
        self.capacity_weight = capacity_weight if capacity_weight is not None else 0
        self.conIso_weight = conIso_weight if conIso_weight is not None else 0
        self.device = device
        self.NCE_temperature = loss_temperature

    def forward_noneg(self, anchor_emb_orig, pos_emb_orig, anchor_len, pos_len, params, velocity_traj=None):
        self.recorded_triplet_loss = None
        self.recorded_InfoNCE_loss = None
        self.recorded_weight_regs_loss = None
        self.recorded_total_loss = None
        if anchor_len.dim() == 2:
            anchor_len, pos_len = anchor_len.squeeze(-1), pos_len.squeeze(-1)
        if self.random_offset: 
            max_len = anchor_emb_orig.shape[1] - 1
            batch_size = anchor_emb_orig.shape[0]

            # Generate random offsets as floats, scale, and add to the start index
            random_offsets_anchor = torch.rand(batch_size, device=self.device) * (max_len - anchor_len).float()
            random_offsets_positive = torch.rand(batch_size, device=self.device) * (max_len - pos_len).float()

            random_indices_anchor = (anchor_len + (random_offsets_anchor.round().int()) * (anchor_len > 1).int()).int()
            random_indices_positive = (pos_len + (random_offsets_positive.round().int()) * (pos_len > 1).int()).int()

            anchor_emb = anchor_emb_orig[torch.arange(anchor_emb_orig.shape[0]), random_indices_anchor]
            pos_emb = pos_emb_orig[torch.arange(pos_emb_orig.shape[0]), random_indices_positive]
            anchor_emb_l = anchor_emb_orig[torch.arange(anchor_emb_orig.shape[0]), anchor_len.int()]
            pos_emb_l = pos_emb_orig[torch.arange(pos_emb_orig.shape[0]), pos_len.int()]
            anchor_emb_f = anchor_emb_orig[torch.arange(anchor_emb_orig.shape[0]), -1]
            pos_emb_f = pos_emb_orig[torch.arange(pos_emb_orig.shape[0]), -1]
            similarity_emb = torch.cat([emb.unsqueeze(1) for emb in [anchor_emb, pos_emb, anchor_emb_l, pos_emb_l, anchor_emb_f, pos_emb_f]], dim=1)
        
        else: 
            anchor_emb = anchor_emb_orig[torch.arange(anchor_emb_orig.shape[0]), :]
            pos_emb = pos_emb_orig[torch.arange(pos_emb.shape[0]), -1]
        if self.normalize:
            anchor_emb, pos_emb = [F.normalize(x, dim=-1) for x in (anchor_emb, pos_emb)]

        if self.similarity_kernel == "cosine":
            if not self.random_offset:
                pos_similarity = torch.sum(anchor_emb * pos_emb, dim=-1)
                neg = anchor_emb @ pos_emb.t()
            else: 
                pos_similarity = torch.sum(similarity_emb[:, 0] * similarity_emb[:, 1], dim=-1)
                neg = similarity_emb[:, 0] @ similarity_emb[:, 1:].t()
        elif self.similarity_kernel == "euclidean":
            if not self.random_offset:
                pos_similarity = (anchor_emb - pos_emb).norm(dim=-1)
                neg = torch.cdist(anchor_emb, pos_emb)
            else: 
                pos_similarity = torch.sum(torch.cdist(similarity_emb, similarity_emb).flatten(start_dim=1), dim=-1)/25
                neg = sum([torch.cdist(similarity_emb[:, i], similarity_emb[:, j]) for i in range(1, 6) for j in range(1, 6)])/36
        
        neg_similarity_sum = torch.sum(neg) - torch.sum(torch.diag(neg))
        neg_similarity_mean = neg_similarity_sum / (pos_emb.shape[0] * (pos_emb.shape[0] - 1))

        match self.loss_type.lower():
            case "triplet": 
                self.recorded_triplet_loss = -pos_similarity.mean() + neg_similarity_mean
                self.recorded_total_loss = self.recorded_triplet_loss

    # Compute the loss for each example
            case "infonce":
                # pos_similarity = pos_similarity/self.NCE_temperature * self.invariance_weight
                pos_similarity = pos_similarity*self.invariance_weight/self.NCE_temperature
                neg = neg/self.NCE_temperature

                torch.diagonal(neg).copy_(pos_similarity)
                if self.similarity_kernel == "cosine":
                    log_sum_exp = torch.logsumexp(neg, dim=1)
                    self.recorded_InfoNCE_loss = (-pos_similarity + log_sum_exp).mean()
                    # similarity = F.log_softmax(neg, dim=1)
                    # self.recorded_InfoNCE_loss = -torch.diagonal(similarity).mean()
                elif self.similarity_kernel == "euclidean":
                    log_sum_exp = torch.logsumexp(-neg, dim=1)
                    # log_sum_exp = torch.exp(-neg / self.NCE_temperature)
                    # similarity = F.log_softmax(-neg, dim=1)
                    # self.recorded_InfoNCE_loss = -torch.diagonal(similarity).mean()
                    self.recorded_InfoNCE_loss = (pos_similarity + log_sum_exp).mean()
                if self.recorded_InfoNCE_loss != self.recorded_InfoNCE_loss:
                    print("Nan loss")  
                    # see if the embeddings are nan
                    print(anchor_emb.isnan().sum())
                    print(pos_emb.isnan().sum())
                    # assert False, "Nan loss, stop training"
                    self.recorded_InfoNCE_loss = 0
                self.recorded_total_loss = self.recorded_InfoNCE_loss
            
            case "combined": 
                self.recorded_triplet_loss = -pos_similarity.mean() + neg_similarity_mean
                pos_similarity = pos_similarity / self.NCE_temperature
                neg = neg / self.NCE_temperature
                similarity = torch.cat((pos_similarity.unsqueeze(1), neg), dim=1)
                similarity = F.log_softmax(similarity, dim=1)
                self.recorded_InfoNCE_loss = -similarity[:, 0].mean()
                self.recorded_total_loss = self.recorded_triplet_loss * self.triplet_weight + self.recorded_InfoNCE_loss * (1 - self.triplet_weight)
        
        if self.weight_decay_scalar > 0 and self.recorded_total_loss < self.weigth_decay_thresh:
            self.recorded_weight_regs_loss = self.weight_decay_scalar * sum(param.pow(2).sum() if len(param.view(-1)) != 1 else 0 for param in params)
            # spare the tau. 
            self.recorded_total_loss += self.recorded_weight_regs_loss
        
        if self.L2_norm_scalar != 0: 
            self.recorded_L2_norm_loss = self.L2_norm_scalar * torch.norm(anchor_emb_orig, dim=-1).mean()
            self.recorded_total_loss += self.recorded_L2_norm_loss
        
        if self.capacity_weight != 0: 
            self.recorded_capacity_loss = -self.capacity_weight * torch.mean(anchor_emb, dim=0).norm()
            self.recorded_total_loss += self.recorded_capacity_loss
        
        if self.conIso_weight != 0: 
            diff_emb = anchor_emb_orig[:, 1:] - anchor_emb_orig[:, :-1]
            diff_emb_norm = diff_emb.norm(dim=-1)
            conIso = diff_emb_norm/(velocity_traj[:, :-1].norm(dim=-1) + 1e-6)
            self.recorded_conIso_loss = self.conIso_weight * (torch.var(conIso.reshape(-1)))
            self.recorded_total_loss += self.recorded_conIso_loss
        return self.recorded_total_loss


    def forward(self, anchor_emb, pos_emb, neg_emb, anchor_len, pos_len, neg_len, params):
        self.recorded_triplet_loss = None
        self.recorded_InfoNCE_loss = None
        self.recorded_weight_regs_loss = None
        self.recorded_total_loss = None
        if anchor_len.dim() == 2:
            anchor_len, pos_len, neg_len = anchor_len.squeeze(-1), pos_len.squeeze(-1), neg_len.squeeze(-1)
        if self.random_offset: 
            max_len = anchor_emb.shape[1]
            batch_size = anchor_emb.shape[0]

            # Generate random offsets as floats, scale, and add to the start index
            random_offsets_anchor = torch.rand(batch_size, device=self.device) * (max_len - anchor_len).float()
            random_offsets_positive = torch.rand(batch_size, device=self.device) * (max_len - pos_len).float()
            random_offsets_negative = torch.rand(batch_size, device=self.device) * (max_len - neg_len).float()

            random_indices_anchor = (anchor_len + (random_offsets_anchor.round().int() - 1) * (anchor_len > 1).int()).int()
            random_indices_positive = (pos_len + (random_offsets_positive.round().int() - 1) * (pos_len > 1).int()).int()
            random_indices_negative = (neg_len + (random_offsets_negative.round().int() - 1) * (neg_len > 1).int()).int()

            anchor_emb = anchor_emb[torch.arange(anchor_emb.shape[0]), random_indices_anchor]
            pos_emb = pos_emb[torch.arange(pos_emb.shape[0]), random_indices_positive]
            neg_emb = neg_emb[torch.arange(neg_emb.shape[0]), random_indices_negative]
        else: 
            anchor_emb = anchor_emb[torch.arange(anchor_emb.shape[0]), -1]
            pos_emb = pos_emb[torch.arange(pos_emb.shape[0]), -1]
            neg_emb = neg_emb[torch.arange(neg_emb.shape[0]), -1]
        if self.normalize:
            anchor_emb, pos_emb, neg_emb = [F.normalize(x, dim=-1) for x in (anchor_emb, pos_emb, neg_emb)]

        pos_similarity = torch.sum(anchor_emb * pos_emb, dim=-1)
        neg = anchor_emb @ neg_emb.t()
        
        neg_similarity_sum = torch.sum(neg) - torch.sum(torch.diag(neg))
        neg_similarity_mean = neg_similarity_sum / (neg_emb.shape[0] * (neg_emb.shape[0] - 1))

        match self.loss_type.lower():
            case "triplet": 
                self.recorded_triplet_loss = -pos_similarity.mean() + neg_similarity_mean
                self.recorded_total_loss = self.recorded_triplet_loss

            case "infonce":
                pos_similarity = pos_similarity / self.NCE_temperature
                neg = neg / self.NCE_temperature
                similarity = torch.cat((pos_similarity.unsqueeze(1), neg), dim=1) 
                similarity -= torch.max(similarity, dim=1, keepdim=True).values

                similarity = F.log_softmax(similarity, dim=1)
                self.recorded_InfoNCE_loss = -similarity[:, 0].mean()
                # catch nan
                if self.recorded_InfoNCE_loss != self.recorded_InfoNCE_loss:
                    print("Nan loss")  
                    # see if the embeddings are nan
                    print(anchor_emb.isnan().sum())
                    print(pos_emb.isnan().sum())
                    print(neg_emb.isnan().sum())
                    # assert False, "Nan loss, stop training"
                    self.recorded_InfoNCE_loss = 0
                self.recorded_total_loss = self.recorded_InfoNCE_loss
            
            case "combined": 
                self.recorded_triplet_loss = -pos_similarity.mean() + neg_similarity_mean
                pos_similarity = pos_similarity / self.NCE_temperature
                neg = neg / self.NCE_temperature
                similarity = torch.cat((pos_similarity.unsqueeze(1), neg), dim=1)
                similarity = F.log_softmax(similarity, dim=1)
                self.recorded_InfoNCE_loss = -similarity[:, 0].mean()
                self.recorded_total_loss = self.recorded_triplet_loss * self.triplet_weight + self.recorded_InfoNCE_loss * (1 - self.triplet_weight)
        
        if self.weight_decay_scalar > 0 and self.recorded_total_loss < self.weigth_decay_thresh:
            self.recorded_weight_regs_loss = self.weight_decay_scalar * sum(param.pow(2).sum() if len(param.view(-1)) != 1 else 0 for param in params)
            # spare the tau. 
            self.recorded_total_loss += self.recorded_weight_regs_loss

        return self.recorded_total_loss

    def log_components(self, epoch, phase): 
        assert self.writer is not None, "No writer provided"
        if self.recorded_triplet_loss is not None: 
            self.writer.add_scalar(f"{phase} triplet loss", self.recorded_triplet_loss, epoch)
        if self.recorded_InfoNCE_loss is not None:
            self.writer.add_scalar(f"{phase} InfoNCE loss", self.recorded_InfoNCE_loss, epoch)
        if self.recorded_weight_regs_loss is not None:
            self.writer.add_scalar(f"{phase} weight regs loss", self.recorded_weight_regs_loss, epoch)
        if self.recorded_total_loss is not None:
            self.writer.add_scalar(f"{phase} total loss", self.recorded_total_loss, epoch)
        