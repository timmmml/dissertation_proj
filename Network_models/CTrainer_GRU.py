import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from copy import deepcopy

def expand_trig(theta):
    return torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)

def pos2theta(y):
    dim = y.shape[-1] // 2
    theta = torch.zeros((*y.shape[:-1], dim), device=y.device)
    for i in range(dim):
        theta[..., i] = torch.atan2(y[..., i + dim], y[..., i])
    return theta


def reinitialise_weights(model):
    for layer in model.children():
        if not hasattr(layer, "frozen") or not layer.frozen:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
            elif isinstance(layer, (torch.nn.Sequential, torch.nn.ModuleList, torch.nn.Module)):
                reinitialise_weights(layer)


class CustomLoss_GRU(torch.nn.Module):
    def __init__(self, additional_loss, writer):
        super(CustomLoss_GRU, self).__init__()
        self.additional_loss = additional_loss
        self.writer = writer
        self.traj_loss = None
        self.state_regs_loss = None
        self.control_regs_loss = None
        self.final_distance = None
        self.weight_decay_loss = None
        self.action_magnitude_loss = None

    def forward(
        self,
        h_trajectory,
        error_trajectory,
        control_trajectory=None,
        out_trajectory=None,
        random_activity=None,
        state_norm_weight=0.0,
        control_norm_weight=0.1,
        convergence_loss_weight=0.02,
        loss_traj_weight=0.01,
        action_weight=0.01,
        weight_decay =0.0,
        balanced_weight=0.0,
        params=None,
    ):
        # Trajectory (primary) loss
        raw_traj = error_trajectory.mean()
        self.traj_loss = raw_traj
        traj_loss = loss_traj_weight * raw_traj

        # Weight decay
        wd = 0
        if params is not None:
            for p in params:
                wd += p.norm(p=2) ** 2
        self.weight_decay_loss = wd
        weight_decay_loss =weight_decay * wd

        # Other components
        state_regs = 0
        control_regs = 0
        final_dist = 0
        for fn in self.additional_loss:
            match fn:
                case 'control_norm':
                    self.control_regs_loss = control_trajectory.norm(p=2, dim=-1).mean()
                    control_regs = control_norm_weight * self.control_regs_loss
                case 'final_distance':
                    self.final_distance = error_trajectory[:, -1].mean()
                    final_dist =  convergence_loss_weight *self.final_distance

        # Action magnitude
        if out_trajectory is not None:
            self.action_magnitude_loss = out_trajectory.norm(p=2, dim=-1).mean()
            action_mag = action_weight * self.action_magnitude_loss
            if balanced_weight > 0:
                assert random_activity is not None, "Random activity must be provided for balanced training"
                action_mag = action_mag + balanced_weight * random_activity.mean()**2
        else:
            self.action_magnitude_loss = None
            action_mag = 0

        loss = traj_loss + state_regs + control_regs + final_dist + weight_decay_loss + action_mag
        return loss

    def log_components(self, epoch, phase):
        comps = {
            "state regularization loss": self.state_regs_loss,
            "control regularization loss": self.control_regs_loss,
            "final distance": self.final_distance,
            "action magnitude loss": self.action_magnitude_loss,
            "weight decay loss": self.weight_decay_loss,
            "average trajectory loss": self.traj_loss,
        }
        for k, v in comps.items():
            if v is not None:
                self.writer.add_scalar(f"{phase} {k}", v.mean() if torch.is_tensor(v) else v, epoch)


class CTrainer_GRU:
    def __init__(self, config=None):
        assert config is not None, "No configuration provided"
        self.config = config
        self.save_path = config['save_path']
        self.check_path = config.get('check_path', None)
        self.log_path = config['log_path']

        self.model_specs = config['model_specs']
        self.model = self._model_type(self.model_specs, config=False)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer_specs = config['optimizer_specs']
        self.optimizer = self._optimizer_type(self.optimizer_specs)
        self.lr_init = self.optimizer_specs['optimizer_params']['lr']

        if config.get('scheduler', False):
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.1,
                patience=100, min_lr=5e-4, cooldown=50)
        else:
            self.scheduler = None

        self.loss_fn = CustomLoss_GRU(config.get('additional_loss', []), writer=None)
        self.need_control_trajectory = 'control_norm' in config.get('additional_loss', [])

        self.model = self.model.to(self.device)
        self.best_val_loss = float('inf')
        self.best_model = None
        self.training_loss_dictionary = None
        self.val_loss_dictionary = None
        self.control_period = config.get('control_period', 1000)

    def refresh(self):
        reinitialise_weights(self.model)
        self.optimizer = self._optimizer_type(self.optimizer_specs)
        if self.scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.1,
                patience=100, min_lr=5e-4, cooldown=50)
        self.model.to(self.device)

    def train(self, **kwargs):
        self.data_gen = kwargs.get('data_gen', None)
        assert self.data_gen is not None, "Data generator not provided"
        self.data_gen.operator = self.model.operator
        epochs = kwargs.get('epochs', 1000)
        save_interval = kwargs.get('save_interval', 10)
        verbose = kwargs.get('verbose', True)
        early_stop = kwargs.get('early_stop', False)
        observe = kwargs.get('observe', 0)

        self.writer = SummaryWriter(self.log_path)
        self.loss_fn.writer = self.writer

        batch_size = kwargs.get('batch_size', 128)
        val_size = round(self.config.get('training_config', {}).get('val_ratio', 0.1) * batch_size)
        train_loader, val_loader = self.data_gen.generate(n_samples=batch_size, val_size=val_size, record=True)

        # Initialize loss dictionaries
        if self.training_loss_dictionary is None:
            keys = [
                'state regularization loss', 'control regularization loss',
                'final distance', 'action magnitude loss',
                'weight decay loss', 'average trajectory loss'
            ]
            self.training_loss_dictionary = {k: [] for k in keys}
            self.val_loss_dictionary = {k: [] for k in keys}

        for e in range(epochs):
            # --- Training ---
            local = {k: 0 for k in self.training_loss_dictionary}
            self.model.train()
            train_loss = 0
            for i, (init, target) in enumerate(train_loader):
                h_traj, ctrl_traj, err_traj, out_traj = self.model(
                    init, target,
                    control_period=self.control_period,
                    return_control_trajectory=True, 
                    observe=observe
                )
                rand_act = None
                if kwargs.get('balanced_weight', 0.0) > 0:
                    rand_in = torch.randn(10000, self.model.plant.controller_dim).to(self.device)
                    rand_out = F.relu(F.linear(rand_in, self.model.plant.apply_dale_constraint(self.model.plant.W_param.data)))
                    rand_act = self.model.operator.readout_matrix(rand_out)

                loss = self.loss_fn(
                    h_traj, err_traj,
                    control_trajectory=ctrl_traj,
                    out_trajectory=out_traj,
                    random_activity=rand_act,
                    state_norm_weight=kwargs.get('state_norm_weight', 0.0),
                    control_norm_weight=kwargs.get('control_norm_weight', 0.1),
                    convergence_loss_weight=kwargs.get('convergence_loss_weight', 0.02),
                    loss_traj_weight=kwargs.get('loss_traj_weight', 0.01),
                    action_weight=kwargs.get('action_weight', 0.01),
                    balanced_weight=kwargs.get('balanced_weight', 0.0),
                    weight_decay=kwargs.get('weight_decay', 0.0),
                    params=self.model.parameters()
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Accumulate losses
                local['state regularization loss'] += self.loss_fn.state_regs_loss.item() if self.loss_fn.state_regs_loss is not None else 0
                local['control regularization loss'] += self.loss_fn.control_regs_loss.item() if self.loss_fn.control_regs_loss is not None else 0
                local['final distance'] += self.loss_fn.final_distance.item() if self.loss_fn.final_distance is not None else 0
                local['action magnitude loss'] += self.loss_fn.action_magnitude_loss.item() if self.loss_fn.action_magnitude_loss is not None else 0
                local['weight decay loss'] += self.loss_fn.weight_decay_loss.item()
                local['average trajectory loss'] += self.loss_fn.traj_loss.item()

                self.writer.add_scalar('training loss', loss.item(), i + e * len(train_loader))
                self.loss_fn.log_components(i + e * len(train_loader), 'training')
                train_loss += loss.item()

            # Normalize and store
            train_loss /= len(train_loader)
            for k in local:
                val = local[k] / len(train_loader)
                self.training_loss_dictionary[k].append(val)

            # Save checkpoint
            if (e + 1) % save_interval == 0 and self.check_path:
                torch.save({
                    'epoch': e,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': train_loss
                }, f"{self.check_path}/checkpoint{e}.pth")

            if self.scheduler:
                self.scheduler.step(train_loss)

            # --- Validation ---
            val_local = {k: 0 for k in self.val_loss_dictionary}
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for i, (init, target) in enumerate(val_loader):
                    h_traj, ctrl_traj, err_traj, out_traj = self.model(init, target, control_period=self.control_period, return_control_trajectory=True, observe = observe)
                    loss = self.loss_fn(h_traj, err_traj, control_trajectory=ctrl_traj, out_trajectory=out_traj,
                                         state_norm_weight=kwargs.get('state_norm_weight', 0.0),
                                         control_norm_weight=kwargs.get('control_norm_weight', 0.1),
                                         convergence_loss_weight=kwargs.get('convergence_loss_weight', 0.02),
                                         loss_traj_weight=kwargs.get('loss_traj_weight', 0.01),
                                         action_weight=kwargs.get('action_weight', 0.01),
                                         weight_decay=kwargs.get('weight_decay', 0.0),
                                         params=self.model.parameters())
                    val_loss += loss.item()
                    # accumulate
                    val_local['state regularization loss'] += (self.loss_fn.state_regs_loss.item() if self.loss_fn.state_regs_loss is not None else 0)
                    val_local['control regularization loss'] += (self.loss_fn.control_regs_loss.item() if self.loss_fn.control_regs_loss is not None else 0)
                    val_local['final distance'] += (self.loss_fn.final_distance.item() if self.loss_fn.final_distance is not None else 0)
                    val_local['action magnitude loss'] += (self.loss_fn.action_magnitude_loss.item() if self.loss_fn.action_magnitude_loss is not None else 0)
                    val_local['weight decay loss'] += self.loss_fn.weight_decay_loss.item()
                    val_local['average trajectory loss'] += self.loss_fn.traj_loss.item()

                    self.loss_fn.log_components(e, 'validation')

            val_loss /= len(val_loader)
            for k in val_local:
                self.val_loss_dictionary[k].append(val_local[k] / len(val_loader))

            self.writer.add_scalar('validation loss', val_loss, e)
            if verbose:
                print(f"Epoch {e}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")

            # Track best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model = deepcopy(self.model.state_dict())
                if self.check_path:
                    print(f"New best model at epoch {e}, saving to {self.check_path}/best_model.pth")
                    # torch.save(self.best_model, f"{self.check_path}/best_model.pth")
                    self.save_checkpoint(f"{self.check_path}/best_model.pth", e, val_loss)
            if early_stop and e - torch.tensor(self.val_loss_dictionary['average trajectory loss'][-1]).argmax().item() >= 100:
                print("Early stopping")
                break
        # self.save_model(self.)
        

    def _model_type(self, model_specs, config=True):
        module = __import__(f"{model_specs['model_path']}", fromlist=[model_specs['model_name']])
        ModelClass = getattr(module, model_specs['model_name'])
        return ModelClass(**(model_specs['model_params'] if not config else model_specs['model_params']))

    def _optimizer_type(self, optimizer_specs):
        OptimizerClass = getattr(optim, optimizer_specs['optimizer_name'])
        return OptimizerClass(self.model.parameters(), **optimizer_specs['optimizer_params'])

    def _reinitialise_weights(self, model):
        for layer in model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
            elif isinstance(
                layer, (torch.nn.Sequential, torch.nn.ModuleList, torch.nn.Module)
            ):
                self._reinitialise_weights(layer)

    def save_checkpoint(self, path, epoch, loss):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": {
                key: opt.state_dict() for key, opt in self.optimizer.items()
            } if isinstance(self.optimizer, dict) else self.optimizer.state_dict(),
            "loss": loss.item() if torch.is_tensor(loss) else loss,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model.to(self.device)

    def save_model(self, path, full=False):
        if full:
            torch.save(self.model, path)
        else:
            torch.save(self.model.state_dict(), path)

    def load_best_model(self):
        self.model.load_state_dict(self.best_model)

    def _model_type(self, model_specs, config=True):
        model_name = model_specs["model_name"]
        try:
            module = __import__(f"{model_specs['model_path']}", fromlist=[model_name])
            ModelClass = getattr(module, model_name)
            if config:
                model = ModelClass(model_specs["model_params"])
            else:
                model = ModelClass(**model_specs["model_params"])
            print("Model loaded")
        except Exception as e:
            print(f"Failed to load model: {e}")
            model = None
        return model

    def _optimizer_type(self, optimizer_specs):
        optimizer_name = optimizer_specs["optimizer_name"]
        OptimizerClass = getattr(optim, optimizer_name)
        optimizer = OptimizerClass(
            self.model.parameters(), **optimizer_specs["optimizer_params"]
        )
        return optimizer

    def _optimizer_type(self, optimizer_specs):
        optimizer_name = optimizer_specs["optimizer_name"]
        compartmental_optimization = optimizer_specs.get("compartmental_optimization", False)
        if compartmental_optimization: 
            print("setting up separate optimizers for the controller and the plant")
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
            plant_parameters = list(self.model.plant.parameters())

            controller_module_names = optimizer_specs.get("controller_module_names", ["gru_cells", "h0", "readout"])

            for controller_module in [getattr(self.model, name, None) for name in controller_module_names]:
                if controller_module is not None:
                    if not isinstance(controller_module, torch.nn.Parameter):
                        controller_parameters += list(controller_module.parameters())
                    else:
                        controller_parameters.append(controller_module)
                    print(f"controller module {controller_module} found")
            
            optimizer = {
                "controller": OptimizerClass(
                    controller_parameters, **(optimizer_specs["optimizer_params"] | {"lr": optimizer_specs["optimizer_params"]["lr"] })
                ),
                "plant": OptimizerClass(
                    plant_parameters, **optimizer_specs["optimizer_params"]
                ),
            }

        # except AttributeError:
        #     print(f"Optimizer {optimizer_name} not found in torch.optim")
        # except Exception as e:
        #     print(f"Failed to initialize optimizer: {e}")
        self.optimizer = optimizer
        return optimizer
 