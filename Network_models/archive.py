# A simplified custom loss function for the GRU controller.
class CustomLoss_GRU(torch.nn.Module):
    def __init__(self, additional_loss, writer):
        super(CustomLoss_GRU, self).__init__()
        self.additional_loss = additional_loss
        self.writer = writer
        self.state_regs_loss = None
        self.control_regs_loss = None
        self.final_distance = None

    def forward(
        self,
        h_trajectory,
        error_trajectory,
        control_trajectory=None,
        out_trajectory=None,
        random_activity = None,
        state_norm_weight=0.0,  # used for hidden states
        control_norm_weight=0.1,
        convergence_loss_weight=0.02,
        loss_traj_weight = 0.01,
        action_weight = 0.01, # used for out_trajectory
        balanced_weight = 0.0, # used for training the plant
        params=None,
    ):

        loss = 0

        traj_loss = error_trajectory.mean() * loss_traj_weight # primary trajectory loss

        # State norm regularization
        state_regs_loss, control_regs_loss, final_distance, weight_decay_loss = 0, 0, 0, 0
        for loss_fn in self.additional_loss:
            match loss_fn:
                case "state_norm":
                    self.state_regs_loss = state_norm_weight * h_trajectory.norm(p=2, dim=-1).mean()
                    state_regs_loss = self.state_regs_loss
                case "control_norm":
                    self.control_regs_loss = control_norm_weight * control_trajectory.norm(p=2, dim=-1).mean()
                    control_regs_loss = self.control_regs_loss
                case "final_distance":
                    self.final_distance = convergence_loss_weight * error_trajectory[:, -1].mean()
                    final_distance = self.final_distance
        for param in params:
            weight_decay_loss = weight_decay_loss + 0.0000 * param.norm(p=2) ** 2
    

        action_magnitude_loss = action_weight * out_trajectory.norm(p=2, dim=-1).mean()
        if balanced_weight > 0:
            assert random_activity is not None, "Random activity must be provided for balanced training"
            action_magnitude_loss = action_magnitude_loss + balanced_weight * random_activity.mean()**2

        loss = traj_loss + state_regs_loss + control_regs_loss + final_distance + weight_decay_loss + action_magnitude_loss
        return loss

    def log_components(self, epoch, phase):
        dictionary = {
            "state regularization loss": self.state_regs_loss,
            "control regularization loss": self.control_regs_loss,
            "final distance": self.final_distance,
            "action magnitude loss": self.action_magnitude_loss,
        }
        for key, value in dictionary.items():
            if value is not None:
                self.writer.add_scalar(f"{phase} {key}", value.mean(), epoch)

class CTrainer_GRU:
    def __init__(self, config=None):
        assert config is not None, "No configuration provided"

        self.config = config
        self.save_path = config["save_path"]
        self.check_path = config.get("check_path", None)
        self.log_path = config["log_path"]

        # model_specs should now be for the GRU controller (file1)
        self.model_specs = config["model_specs"]
        self.model = self._model_type(self.model_specs, config=False)

        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.optimizer_specs = config["optimizer_specs"]
        self.optimizer = self._optimizer_type(self.optimizer_specs)
        self.lr_init = self.optimizer_specs["optimizer_params"]["lr"]

        self.latest_checkpoint = 0

        if config.get("scheduler", False):
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
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
            print("No scheduler")

        # Use a simplified loss function for GRU controller.
        self.loss_fn = CustomLoss_GRU(config.get("additional_loss", []), writer=None)
        self.need_control_trajectory = "control_norm" in config.get(
            "additional_loss", []
        )

        self.model = self.model.to(self.device)
        self.model.device = self.device

        self.best_val_loss = float("inf")
        self.best_model = None

        self.control_period = config.get("control_period", 1000)

    def refresh(self):
        self._reinitialise_weights(self.model)
        self.optimizer = self._optimizer_type(self.optimizer_specs)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.1,
            patience=100,
            min_lr=5e-4,
            cooldown=50,
        )
        self.model.to(self.device)

    def train(self, **kwargs):
        self.data_gen = kwargs.get("data_gen", None)
        assert self.data_gen is not None, "Data generator not provided"
        self.data_gen.operator = self.model.operator
        epochs = kwargs.get("epochs", 1000)
        save_interval = kwargs.get("save_interval", 10)
        verbose = kwargs.get("verbose", True)
        early_stop = kwargs.get("early_stop", False)

        self.writer = SummaryWriter(self.log_path)
        self.loss_fn.writer = self.writer

        batch_size = kwargs.get("batch_size", 128)
        val_size = round(
            self.config.get("training_config", {}).get("val_ratio", 0.1) * batch_size
        )
        train_loader, val_loader = self.data_gen.generate(
            n_samples=batch_size, val_size=val_size, record=True
        )

        observe = kwargs.get("observe", 0)
        state_norm_weight, control_norm_weight, convergence_loss_weight, loss_traj_weight, action_weight = (
            kwargs.get("state_norm_weight", 0.0), kwargs.get("control_norm_weight", 0.1), 
            kwargs.get("convergence_loss_weight", 0.02), kwargs.get("loss_traj_weight", 0.01), kwargs.get("action_weight", 0.01))

        for e in range(epochs):
            self.model.train()
            train_loss = 0
            for i, data in enumerate(train_loader):
                init, target = data
                # For the GRU controller, forward takes x_0 and ref.
                h_trajectory, control_trajectory, error_trajectory, out_trajectory = (
                    self.model(
                        init,
                        target,
                        control_period=self.control_period,
                        return_control_trajectory=True,
                        observe=observe,
                    )
                )
                if kwargs.get("balanced_weight", 0.0) > 0:
                    random_in = torch.randn(10000, self.model.plant.controller_dim).cuda()
                    random_out = F.relu(F.linear(random_in, self.model.plant.apply_dale_constraint(self.model.plant.W_param.data)))
                    random_activity = self.model.operator.readout_matrix(random_out)
                else: 
                    random_activity = None

                # Compute loss using our simplified loss function.
                loss = self.loss_fn(
                    h_trajectory,
                    error_trajectory,
                    control_trajectory=control_trajectory,
                    out_trajectory=out_trajectory,
                    random_activity = random_activity,
                    params=self.model.parameters(),
                    state_norm_weight=state_norm_weight,
                    control_norm_weight=control_norm_weight,
                    convergence_loss_weight=convergence_loss_weight,
                    loss_traj_weight=loss_traj_weight,
                    action_weight=action_weight,
                    balanced_weight=kwargs.get("balanced_weight", 0.0),
                )
                if isinstance(self.optimizer, dict): 
                    self.optimizer["controller"].zero_grad()
                    loss.backward(retain_graph = True)
                    self.optimizer["controller"].step()
                    # if e % 10 == 0: 
                    #     self.optimizer["plant"].zero_grad()
                    #     loss.backward(retain_graph = True)
                    #     self.optimizer["plant"].step()
                else: 
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()


                self.writer.add_scalar(
                    "training loss", loss.item(), i + e * len(train_loader)
                )
                train_loss += loss.item()
            train_loss /= len(train_loader)
            if verbose:
                print(f"Epoch {e}, training loss: {train_loss}")

            if (e + 1) % save_interval == 0:
                if self.check_path is not None:
                    self.save_checkpoint(
                        self.check_path + f"/checkpoint{e}.pth", e, loss
                    )

            if self.scheduler is not None:
                self.scheduler.step(train_loss)

            if self.validate(val_loader, e, early_stop, kwargs = kwargs):
                break
        self.save_model(self.save_path, full=True)

    def validate(self, val_loader, epoch, early_stop=False, kwargs={}):
        self.model.eval()
        observe = kwargs.get("observe", 0)
        state_norm_weight, control_norm_weight, convergence_loss_weight, loss_traj_weight, action_weight = (
            kwargs.get("state_norm_weight", 0.0), kwargs.get("control_norm_weight", 0.1), 
            kwargs.get("convergence_loss_weight", 0.02), kwargs.get("loss_traj_weight", 0.01), kwargs.get("action_weight", 0.01))

        val_total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                init, target = data
                h_trajectory, control_trajectory, error_trajectory, out_trajectory = (
                    self.model(
                        init,
                        target,
                        control_period=self.control_period,
                        return_control_trajectory=True,
                        observe=observe,
                    )
                )
                loss = self.loss_fn(
                    h_trajectory,
                    error_trajectory,
                    control_trajectory=control_trajectory,
                    out_trajectory=out_trajectory,
                    params=self.model.parameters(),
                    state_norm_weight=state_norm_weight,
                    control_norm_weight=control_norm_weight,
                    convergence_loss_weight=convergence_loss_weight,
                    loss_traj_weight=loss_traj_weight,
                    action_weight=action_weight,
                )
                val_total_loss += loss.item()
        avg_val_loss = val_total_loss / len(val_loader)
        self.writer.add_scalar("validation loss", avg_val_loss, epoch)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, validation loss: {avg_val_loss}")
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.best_model = deepcopy(self.model.state_dict())
            if self.check_path is not None:
                self.save_checkpoint(
                    self.check_path + f"/best_model.pth", epoch, avg_val_loss
                )
        return 0

