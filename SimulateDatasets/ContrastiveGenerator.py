import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split, RandomSampler
from .DataGenerator import DataGenerator
from importlib import reload
import numpy as np
from Rotations import integrate_velocities

class ControlGenerator_GRU(DataGenerator):
    def __init__(self, configs): 
        super(ControlGenerator_GRU, self).__init__(additional_config=configs)
        self.device = configs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.contrastive_generator = configs.get("contrastive_generator", None)
        self.to_null = configs.get("to_null", False)
        self.null_rep = configs.get("null_rep", None)
        self.generator = torch.Generator(device="cpu")
        self.num_targets = configs.get("num_targets", 1)

    def generate_data(self, n_samples, disregard_n_samples=True):
        assert self.operator is not None
        sampled_dataset = self.contrastive_generator.generate_data(n_samples, disregard_n_samples)
        # sampled_dataset = self.contrastive_generator.generate_data(64, False)
        self.operator.set_null_states(sampled_dataset.data_base.shape[0])
        old_dt = self.operator.dt
        self.operator.dt = 1 
        for i in range(sampled_dataset.data_base.shape[1]):
            self.operator.s = torch.cat([self.operator.s, self.operator.integrate_velocity(sampled_dataset.data_base[:, i, :], self.operator.s).unsqueeze(1)], dim = 1)
        self.operator.dt = old_dt
                
        positions = self.operator.s[:, -1, :self.operator.position_size]
        dataset = ControlDataset_GRU(positions, to_null=self.to_null, null_rep=self.null_rep, num_targets = self.num_targets).to(self.device)
        # print(dataset.activations.shape)
        return dataset
    
    def replace(self, replacement_rate):
        """here, let's just always replace 100%"""
        return self.generate(len(self.train_data), len(self.val_data), record=True)
    
    def retrieve(self):
        return (DataLoader(self.train_data, batch_size=self.mini_batch_size, shuffle=False, sampler=RandomSampler(self.train_data, generator=self.generator)),
                DataLoader(self.val_data, batch_size=self.mini_batch_size, shuffle=False, sampler=RandomSampler(self.val_data, generator=self.generator)))
    
class ControlDataset_GRU(Dataset):
    def __init__(self, positions, to_null=False, null_rep=None, num_targets = 1): 
        self.activations = positions
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.activations = self.activations.to(self.device)
        self.dim = self.activations.shape[-1]
        self.to_null = to_null
        self.num_targets = num_targets
        if self.to_null:
            assert null_rep is not None
            self.null_rep = null_rep
    
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        #sample random from and to
        from_id = torch.randint(0, len(self.activations), (1,)).item()
        target_number = torch.randint(1, self.num_targets + 1, (1,)).item()
        to_id = torch.randint(0, len(self.activations), (target_number,))
        to = self.activations[to_id] if not self.to_null else self.null_rep
        complement = to[-1].unsqueeze(0).repeat(self.num_targets - target_number, 1)
        to = torch.cat([to, complement], dim = 0)
        
        return (self.activations[from_id], to)
    
    def to(self, device):
        self.activations = self.activations.to(device)
        self.device = device
        return self


class ControlGenerator(DataGenerator):
    def __init__(self, configs): 
        super(ControlGenerator, self).__init__(additional_config=configs)
        self.device = configs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.contrastive_generator = configs.get("contrastive_generator", None)
        self.rep_only = configs.get("rep_only", False) # in the returned target, do I return the representation only or the original data?
        self.to_null = configs.get("to_null", False)
        self.null_rep = configs.get("null_rep", None)
        self.generator = torch.Generator(device="cpu")

    def generate_data(self, n_samples, disregard_n_samples=True):
        self.rep_model.to(self.device)
        self.rep_model.eval()
        sampled_dataset = self.contrastive_generator.generate_data(n_samples, disregard_n_samples)
        # sampled_dataset = self.contrastive_generator.generate_data(64, False)
        with torch.no_grad(): 
            sampled_dataset.activations = self.rep_model(sampled_dataset.data_base.to(self.device), truncate_output=False)[:, -1]
        sampled_dataset.rep_units = self.rep_model.rnn.rep_units
        dataset = ControlDataset(sampled_dataset, self.rep_only, to_null=self.to_null, null_rep=self.null_rep).to(self.device)
        # print(dataset.activations.shape)
        return dataset
    
    def replace(self, replacement_rate):
        """here, let's just always replace 100%"""
        return self.generate(len(self.train_data), len(self.val_data), record=True)
    
    def retrieve(self):
        return (DataLoader(self.train_data, batch_size=self.mini_batch_size, shuffle=False, sampler=RandomSampler(self.train_data, generator=self.generator)),
                DataLoader(self.val_data, batch_size=self.mini_batch_size, shuffle=False, sampler=RandomSampler(self.val_data, generator=self.generator)))
    
class ControlDataset(Dataset):
    def __init__(self, sampled_dataset, rep_only=False, to_null=False, null_rep=None): 
        self.activations = sampled_dataset.activations
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.activations = self.activations.to(self.device)
        self.rep_units = sampled_dataset.rep_units
        self.rep_only = rep_only
        self.dim = self.activations.shape[-1]
        self.to_null = to_null
        if self.to_null:
            assert null_rep is not None
            self.null_rep = null_rep
    
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        #sample random from and to
        from_id = torch.randint(0, len(self.activations), (1,)).item()
        to_id = torch.randint(0, len(self.activations), (1,)).item()
        to = self.activations[to_id] if not self.to_null else self.null_rep
        if not self.rep_only: 
            return (self.activations[from_id], to)
        else:
            return (self.activations[from_id], to[-self.rep_units:])
    
    def to(self, device):
        self.activations = self.activations.to(device)
        self.device = device
        return self

class ContrastiveGenerator(DataGenerator):
    def __init__(self, configs):
        super(ContrastiveGenerator, self).__init__(additional_config=configs)
        self.additional_config = configs
        self.mode = configs.get("mode", "binned_AR_S1")  # or ordered_AR
        self.generator_params = configs.get("generator_params", {})
        self.device = configs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    def generate_data(self, n_samples, disregard_n_samples=True):
        match self.mode.lower():
            case "binned_ar_s1":
                data, _, trial_lengths = S1_AR1_binned_gen(**self.generator_params, device=self.device)
                return BinnedDataset(data, trial_lengths, n_samples, device=self.device)
            case "ordered_ar":
                if disregard_n_samples:
                    data, trial_lengths = AR1_ordered_data_gen(self.generator_params |{"device": self.device} )
                else:
                    data, trial_lengths = AR1_ordered_data_gen(self.generator_params | {"n_total_trial": n_samples, "device": self.device})
                # manifold = self.generator_params.get("manifold_specs", {}).get("manifold_name", "T").lower()
                # if self.generator_params.get("manifold", "T").lower() == "t":
                #     return OrderedDataset(data, trial_lengths, self.generator_params)
                # elif self.generator_params.get("manifold", "T").lower() == "s":
                print(data.shape)
                return SampledDataset(data, trial_lengths, self.generator_params.get("manifold_specs", {}), self.generator_params| {"device" : self.device})
    def replace(self, replacement_rate):
        """here, let's just always replace 100%"""
        return self.generate(len(self.train_data), len(self.val_data), record=True)
    
    def retrieve(self):
        return DataLoader(self.train_data, batch_size=self.mini_batch_size, shuffle=True), DataLoader(self.val_data, batch_size=self.mini_batch_size, shuffle=True)


class BinnedDataset(Dataset):
    def __init__(
        self,
        data,
        trial_lengths,
        samples_per_epoch=2048,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.data = data
        self.trial_lengths = trial_lengths
        self.samples_per_epoch = samples_per_epoch
        self.n_bins, self.trials_per_bin, self.max_time = data.shape
        self.device = device

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # Sample anchor bin and trials
        anchor_bin = torch.randint(0, self.n_bins, (1,)).item()
        anchor_idx, positive_idx = torch.randperm(self.trials_per_bin)[:2]

        # Sample negative from different bin
        negative_bin = (
            anchor_bin + 1 + torch.randint(0, self.n_bins - 1, (1,)).item()
        ) % self.n_bins
        negative_idx = torch.randint(0, self.trials_per_bin, (1,)).item()

        # Get trials and lengths
        anchor = self.data[anchor_bin, anchor_idx]
        positive = self.data[anchor_bin, positive_idx]
        negative = self.data[negative_bin, negative_idx]

        anchor_len = self.trial_lengths[anchor_bin, anchor_idx]
        positive_len = self.trial_lengths[anchor_bin, positive_idx]
        negative_len = self.trial_lengths[negative_bin, negative_idx]

        return anchor, positive, negative, anchor_len, positive_len, negative_len


def S1_AR1_binned_gen(
    n_bins=30,
    n_trials_per_bin=100,
    max_time_steps=10,
    ar_coef=0.8,
    noise_std=0.1,
    min_length=20,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Generate training data using AR processes, binned by their sums modulo 2π.
    Implementation uses PyTorch for GPU acceleration.

    Args:
        n_bins (int): Number of bins to divide the data into
        n_trials_per_bin (int): Number of trials to generate per bin
        max_time_steps (int): Maximum length of each trial
        ar_coef (float): AR(1) coefficient for temporal correlation
        noise_std (float): Standard deviation of the noise
        min_length (int): Minimum length of trials
        device (str): Device to use for computations ('cuda' or 'cpu')

    Returns:
        tuple: (data, bin_edges, trial_lengths)
            - data: tensor of shape (n_bins, n_trials_per_bin, max_time_steps)
            - bin_edges: tensor of bin boundaries
            - trial_lengths: tensor of actual trial lengths
    """
    # First generate all trials we might need (with some buffer)
    n_total_trials = int(n_bins * n_trials_per_bin * 10)  # 50% buffer

    # Generate AR(1) processes
    data = torch.zeros((n_total_trials, max_time_steps), device=device)
    noise = torch.randn(n_total_trials, max_time_steps, device=device) * noise_std

    # Efficient AR(1) process generation using PyTorch operations
    data[:, 0] = noise[:, 0]
    for t in range(1, max_time_steps):
        data[:, t] = ar_coef * data[:, t - 1] + noise[:, t]

    # Generate random lengths for each trial
    if min_length < max_time_steps:
        trial_lengths = torch.randint(
            min_length, max_time_steps + 1, (n_total_trials,), device=device
        )
    elif min_length == max_time_steps:
        trial_lengths = torch.full(
            (n_total_trials,), max_time_steps + 1, dtype=torch.long, device=device
        )

    # Create mask for trial lengths
    mask = torch.arange(max_time_steps, device=device).expand(n_total_trials, -1)
    mask = mask < trial_lengths.unsqueeze(1)

    masked_data = data * mask.float()

    sums = torch.sum(masked_data, dim=1) % (2 * np.pi)

    bin_edges = torch.linspace(0, 2 * np.pi, n_bins + 1, device=device)
    bin_indices = torch.bucketize(sums, bin_edges) - 1

    final_data = torch.zeros((n_bins, n_trials_per_bin, max_time_steps), device=device)
    final_lengths = torch.zeros(
        (n_bins, n_trials_per_bin), dtype=torch.long, device=device
    )

    for bin_idx in range(n_bins):
        bin_mask = bin_indices == bin_idx
        bin_trials = masked_data[bin_mask]
        bin_trial_lengths = trial_lengths[bin_mask]

        # If we don't have enough trials for this bin, generate more
        if len(bin_trials) < n_trials_per_bin:
            print(f"fill in noise for bin {bin_idx}")
            additional_trials = n_trials_per_bin - len(bin_trials)
            noise = (
                torch.randn(additional_trials, max_time_steps, device=device)
                * noise_std
            )
            bin_trials = torch.cat([bin_trials, noise])
            additional_lengths = torch.randint(
                min_length, max_time_steps, (additional_trials,), device=device
            )
            bin_trial_lengths = torch.cat([bin_trial_lengths, additional_lengths])

        # Select n_trials_per_bin random trials from this bin
        perm = torch.randperm(len(bin_trials), device=device)[:n_trials_per_bin]
        final_data[bin_idx] = bin_trials[perm]
        final_lengths[bin_idx] = bin_trial_lengths[perm]

    return final_data, bin_edges, final_lengths



class OrderedDataset(Dataset):
    def __init__(self, data, trial_lengths, kwargs={}):
        # baseline_offset_n=0.05, # percent of samples_per_dim
        # baseline_offset_p=0.01, # percent of samples_per_dim
        # samples_per_epoch=2048,
        # device="cuda" if torch.cuda.is_available() else "cpu",
        baseline_offset_n = kwargs.get("baseline_offset_n", 0.05)
        baseline_offset_p = kwargs.get("baseline_offset_p", 0.01)
        samples_per_epoch = kwargs.get("samples_per_epoch", 2048)
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.data = data
        self.trial_lengths = trial_lengths
        self.samples_per_epoch = samples_per_epoch
        self.samples_per_dim = int(data.shape[0] ** (1 / data.shape[-1]))
        self.max_time = data.shape[-2]
        self.dim = data.shape[-1]
        self.device = device
        self.baseline_offset_n = int(baseline_offset_n * self.samples_per_dim)
        self.baseline_offset_p = (
            [int(baseline_offset_p * self.samples_per_dim) for _ in range(self.dim)]
            if type(baseline_offset_p) == float
            else [int(bp * self.samples_per_dim) for bp in baseline_offset_p]
        )
        assert (
            type(self.baseline_offset_p) == list
            and len(self.baseline_offset_p) == self.dim
        )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # Sample anchor bin and trials
        anchor_id = torch.randint(0, self.samples_per_dim, [self.dim])
        anchor_id_ = sum(
            [anchor_id[-i] * self.samples_per_dim**i for i in range(self.dim)]
        )

        random_offset_pos = (
            torch.tensor(
                [
                    torch.randint(
                        -self.baseline_offset_p[i], self.baseline_offset_p[i], [1]
                    )
                    for i in range(self.dim)
                ],
                device=anchor_id.device,
            )
            % self.samples_per_dim
        )

        positive_id = (anchor_id + random_offset_pos) % self.samples_per_dim
        positive_id = sum(
            [positive_id[-i] * self.samples_per_dim**i for i in range(self.dim)]
        )

        random_offset_neg = (
            self.baseline_offset_n
            + torch.randint(
                0,
                self.samples_per_dim - self.baseline_offset_n,
                [self.dim],
            )
        ) % self.samples_per_dim
        negative_id = (anchor_id + random_offset_neg) % self.samples_per_dim
        negative_id = sum(
            [negative_id[-i] * self.samples_per_dim**i for i in range(self.dim)]
        )

        anchor = self.data[anchor_id_]
        positive = self.data[positive_id]
        negative = self.data[negative_id]

        anchor_len = self.trial_lengths[anchor_id_]
        positive_len = self.trial_lengths[positive_id]
        negative_len = self.trial_lengths[negative_id]

        return anchor, positive, negative, anchor_len, positive_len, negative_len


def AR1_ordered_data_gen(kwargs={}):
    """
    Generates n_total_trial**dim trials of AR1 process;
    - here, the output is in the form of *(n_total_trial for _ in range(dim)), max_time_steps, dim
    - they are ordered such that we would expect those that are closer to each other on the first dim dimensions would result in nearby movements
    - TODO: use this in the data generation process by constructing a way to sample anchor, pos, and neg trials from them.
    """
    # n_total_trial=100,
    # dim=2,
    # max_time_steps=10,
    # ar_coef=0.9,
    # noise_std=0.32,
    # min_length=10,
    # device="cuda" if torch.cuda.is_available() else "cpu",
    n_total_trial = kwargs.get("n_total_trial", 100)
    dim = kwargs.get("dim", 2)
    max_time_steps = kwargs.get("max_time_steps", 10)
    ar_coef = kwargs.get("ar_coef", 0.9)
    noise_std = kwargs.get("noise_std", 0.32)
    min_length = kwargs.get("min_length", 10)
    device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    # First generate all trials we might need (with some buffer)
    n_total_trials = int(n_total_trial**dim)

    # Generate AR(1) processes
    data = torch.zeros((n_total_trials, max_time_steps, dim), device=device)
    noise = torch.randn(n_total_trials, max_time_steps, dim, device=device) * noise_std

    # Efficient AR(1) process generation using PyTorch operations
    data[:, 0] = noise[:, 0]
    for t in range(1, max_time_steps):
        data[:, t] = ar_coef * data[:, t - 1] + noise[:, t]

    # Generate random lengths for each trial
    if min_length < max_time_steps:
        trial_lengths = torch.randint(
            min_length, max_time_steps + 1, (n_total_trials,), device=device
        )
    elif min_length == max_time_steps:
        trial_lengths = torch.full(
            (n_total_trials,), max_time_steps + 1, dtype=torch.long, device=device
        )

    # Create mask for trial lengths
    mask = (
        torch.arange(max_time_steps, device=device)
        .expand(n_total_trials, -1)
        .unsqueeze(-1)
    )
    mask = mask < trial_lengths.unsqueeze(1).unsqueeze(-1)

    masked_data = data * mask.float()

    manifold = kwargs.get("manifold", "T").lower()
    if manifold == "t": 
        masked_data = masked_data.reshape(
            *[n_total_trial for _ in range(dim)], max_time_steps, dim
        )
        trial_lengths = trial_lengths.reshape(*[n_total_trial for _ in range(dim)])
        masked_data = masked_data.permute(*list(range(dim)), -1, -2)
        masked_data, trial_lengths = sort_data_manifold_torus(dim, masked_data, trial_lengths)
        masked_data = masked_data.permute(*list(range(dim)), -1, -2)
    # NO SORTING FOR ROTATIONS - DOESN'T WORK THAT WAY
    masked_data = masked_data.reshape(-1, max_time_steps, dim)
    trial_lengths = trial_lengths.reshape(-1)
    return masked_data, trial_lengths

def sort_data_manifold_torus(dim, masked_data, trial_lengths):
    sums = torch.sum(masked_data, dim=-1)
    while torch.any(sums < -np.pi) or torch.any(sums > np.pi):
        sums = sums + 2 * np.pi * (sums < -np.pi) - 2 * np.pi * (sums > np.pi)
    n_total_trial = masked_data.shape[0]
    if dim == 1:
        order = torch.sort(sums[:, 0])[1]
        masked_data = masked_data[order, :]
        sums = sums[order]
        trial_lengths = trial_lengths[order]

    elif dim == 2:
        for i in range(n_total_trial):
            order = torch.sort(sums[i, :, 0])[1]
            masked_data[i, :] = masked_data[i, order, :]
            sums[i, :] = sums[i, order]
            trial_lengths[i] = trial_lengths[i][order]
        for j in range(n_total_trial):
            order = torch.sort(sums[:, j, 1])[1]
            masked_data[:, j] = masked_data[order, j, :]
            sums[:, j] = sums[order, j]
            trial_lengths[:, j] = trial_lengths[order, j]

    elif dim == 3:
        for i in range(n_total_trial):
            for j in range(n_total_trial):
                order = torch.sort(sums[i, j, :, 2])[1]
                masked_data[i, j, :] = masked_data[i, j, order, :]
                sums[i, j, :] = sums[i, j, order]
                trial_lengths[i, j, :] = trial_lengths[i, j, order]

        for k in range(n_total_trial):
            for l in range(n_total_trial):
                order = torch.sort(sums[:, k, l, 0])[1]
                masked_data[:, k, l] = masked_data[order, k, l, :]
                sums[:, k, l] = sums[order, k, l]
                trial_lengths[:, k, l] = trial_lengths[order, k, l]

        for m in range(n_total_trial):
            for n in range(n_total_trial):
                order = torch.sort(sums[m, :, n, 1])[1]
                masked_data[m, :, n] = masked_data[m, order, n, :]
                sums[m, :, n] = sums[m, order, n]
                trial_lengths[m, :, n] = trial_lengths[m, order, n]
    return masked_data, trial_lengths


class SampledDataset(Dataset):
    """
    Generates positive and negative pairs based on manifold geometry.
    """
    def __init__(self, data, trial_lengths, manifold_specs, kwargs={}):
        # self.epsilon = kwargs.get("epsilon", 0.2)
        self.percentile = kwargs.get("percentile", 10) # only sample from the top 10% of the distances as positive pairs
        self.data_base = data  # Shape: (N, max_len, dim)
        self.bi_directional = kwargs.get("bidirectional", False)
        # the data comes in whatever - but if bidirectional we expand the data space by dividing into positive and negative 
        if self.bi_directional: 
            self.data_base = torch.cat([torch.relu(self.data_base), torch.relu(-self.data_base)], dim = -1)

        self.dim = self.data_base.shape[-1]

        self.trial_lengths_base = trial_lengths
        self.samples_per_epoch = kwargs.get("samples_per_epoch", 2048)
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon_max = kwargs.get("epsilon_max", np.inf)
        self.data_base = self.data_base.to(self.device)
        self.max_time = data.shape[1]
        
        try: 
            module = __import__(f"{manifold_specs["manifold_path"]}", fromlist=[manifold_specs["manifold_name"]])
            modelclass = getattr(module, manifold_specs["manifold_name"])
            self.manifold = modelclass(**manifold_specs["manifold_params"])
        except Exception as e:
            print(f"Error: {e}")
            raise ValueError("Manifold not found")
        self.dist_mat = None
        self.positions = None
        self.lookups = None
        self.lookups_neg = None
        self._integrate()
        self._find_positive_pairs()


    def _integrate(self):
        # Integrate velocities to obtain positions on the manifold
        if self.bi_directional is True: 
            self.positions = self.manifold.integrate(self.data_base[..., :int(self.dim/2)] - self.data_base[..., int(self.dim/2):], dt=1.0)  # Shape: (N, T, D)
        else:
            self.positions = self.manifold.integrate(self.data_base, dt=1.0)  # Shape: (N, T, D)

    def _find_positive_pairs(self):
        # Compute pairwise distances using the manifold's distance function
        num_samples = self.positions.shape[0]
        self.dist_mat = None
        dist_mat = self.manifold.distance(self.positions, self.positions)  # Shape: (N, N)
        # self.dist_mat = dist_mat

        # Create lookup lists for positive pairs
        self.lookups = []
        self.lookups_neg = []
        for i in range(num_samples):
            # Find indices where distance is less than epsilon (excluding self)
            # epsilon= torch.kthvalue(dist_mat[i], int(self.percentile * num_samples / 100), dim=0).values
            epsilon=torch.quantile(dist_mat[i][torch.where(dist_mat[i] > 0)], self.percentile / 100)
            if epsilon > self.epsilon_max:
                epsilon = self.epsilon_max
            if i == 0:
                print(epsilon)
            indices = torch.where((dist_mat[i] < epsilon) & (dist_mat[i] > 0))[0]
            # indices = torch.where(dist_mat[i] < epsilon)[0]
            self.lookups.append(indices)
            # self.lookups_neg.append(torch.where(dist_mat[i] > epsilon)[0])

    def __len__(self):
        return self.samples_per_epoch
    
    # def __getitem__(self, idx):
    #     """rather than relying on lookup tables, let's just sample randomly and say positive is the closer one"""
    #     anchor_id = torch.randint(0, self.positions.shape[0], (1,)).item()
    #     index1 = (anchor_id + torch.randint(1, self.positions.shape[0], (1,)).item()) % (self.positions.shape[0] - 1)
    #     index2 = (anchor_id + torch.randint(1, self.positions.shape[0], (1,)).item()) % (self.positions.shape[0] - 1)
    #     if self.dist_mat[anchor_id, index1] >= self.dist_mat[anchor_id, index2]:
    #         index1, index2 = index2, index1
    #     return (
    #         self.data_base[anchor_id, :, :self.dim], 
    #         self.data_base[index1, :, :self.dim], 
    #         self.data_base[index2, :, :self.dim], 
    #         self.trial_lengths_base[anchor_id], 
    #         self.trial_lengths_base[index1], 
    #         self.trial_lengths_base[index2]
    #     )
    
    def __getitem__(self, idx):
        # Sample a random anchor
        anchor_id = torch.randint(0, self.positions.shape[0], (1,)).item()
        # Sample a positive example
        positive_indices = self.lookups[anchor_id]
        # negative_indices = self.lookups_neg[anchor_id]
        if len(positive_indices) == 0:
            # If no positive examples within epsilon, resample anchor

            # print(self.trial_lengths_base[anchor_id])
            # print("Resampling anchor")
            return self.__getitem__(idx)
        positive_id = positive_indices[torch.randint(0, len(positive_indices), (1,)).item()]
        negative_id = torch.randint(0, self.positions.shape[0], (1,)).item()
        # negative_id = negative_indices[torch.randint(0, len(negative_indices), (1,)).item()]

        while negative_id == anchor_id or negative_id == positive_id:
            negative_id = torch.randint(0, self.positions.shape[0], (1,)).item()

        return (
            self.data_base[anchor_id, :, :self.dim], 
            self.data_base[positive_id, :, :self.dim], 
            self.data_base[negative_id, :, :self.dim], 
            self.trial_lengths_base[anchor_id], 
            self.trial_lengths_base[positive_id], 
            self.trial_lengths_base[negative_id]
        )


#----------------------------------------------------------------------------------------------------------------------------
class _SampledDataset(Dataset):
    """Instead of using the dataset structure to reflect the manifold, we here explicitly find positive pairs surrounding each sample by specifying a distance function and a threshold
    
    """
    def __init__(self, data, trial_lengths, kwargs={}): 
        self.epsilon = kwargs.get("epsilon", 0.2)
        self.data_base = data # the shape is going to be (N, max_len, dim)
        self.dim = data.shape[-1]

        self.trial_lengths_base = trial_lengths
        self.samples_per_epoch = kwargs.get("samples_per_epoch", 2048)
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.data_base = self.data_base.to(self.device)
        if self.dim == 2:
            self.data_base = torch.cat([data, torch.zeros(data.shape[0], data.shape[1], 1, device=self.device)], dim=-1)
        elif self.dim == 1:
            self.data_base = torch.cat([data, torch.zeros(data.shape[0], data.shape[1], 2, device=self.device)], dim=-1)
        self.max_time = data.shape[-2]
        self.manifold = kwargs.get("manifold", "T").lower()
        assert self.manifold == "s", "This method for rotations hasn't been implemented"
        # assume geodesic distance function
        self.quats = None
        self.lookups = None
        self._integrate()
        self._find_positive_pairs()
        
    def _integrate(self): 
        # integrate the angular velocities to a quaternion element. In the case of dim == 2, this is equivalent to having the third channel always zero
        self.quats = integrate_velocities(self.data_base, dt=1.0) # resulting in N, 4

    def _find_positive_pairs(self, N=1):
        # create N copies of self.data_base which contains the positive "friend" of the base element
        dist_mat = torch.acos(torch.clamp(torch.abs(self.quats @ self.quats.T), -1, 1))
        self.lookups = []
    
        for i in range(self.quats.shape[0]):
            self.lookups.append(torch.where(dist_mat[i] < self.epsilon)[0])
    
    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # sample a random anchor
        anchor_id = torch.randint(0, self.quats.shape[0], (1,)).item()
        # sample a random positive from the lookups
        while len(self.lookups[anchor_id]) == 1:
            anchor_id = torch.randint(0, self.quats.shape[0], (1,)).item()
        positive_id = self.lookups[anchor_id][torch.randint(0, len(self.lookups[anchor_id]), (1,)).item()]
        while positive_id == anchor_id:
            positive_id = self.lookups[anchor_id][torch.randint(0, len(self.lookups[anchor_id]), (1,)).item()]
        # sample a random negative
        negative_id = torch.randint(0, self.quats.shape[0], (1,)).item()
        return self.data_base[anchor_id, :, :self.dim], self.data_base[positive_id, :, :self.dim], self.data_base[negative_id, :, :self.dim], self.trial_lengths_base[anchor_id], self.trial_lengths_base[positive_id], self.trial_lengths_base[negative_id]