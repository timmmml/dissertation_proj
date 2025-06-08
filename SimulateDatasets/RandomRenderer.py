"""RandomRenderer class for training VAE 

"""
import torch
import torch.nn as nn
from .RendererGenerator import RendererGenerator
from .DataGenerator import DataGenerator
from torch.utils.data import DataLoader, Dataset

from Network_models.DynamicalNetwork import DynamicalNetwork
from utils.path_settings import *
from Network_models.BaseNNAgent import BaseNNAgent
from numba import jit
from scipy.linalg import solve_continuous_lyapunov
from utils import goto_project_root
from torch.utils.data import DataLoader, Dataset

from pytorch3d.transforms import quaternion_to_matrix, quaternion_apply
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    TexturesVertex,
    look_at_view_transform,
    FoVPerspectiveCameras,
)

class RandomRenderer(RendererGenerator):
    def __init__(self, config=None):
        super(RandomRenderer, self).__init__(config)
        # NOTE: for this to work, the configs must include the renderer specs:
        # "object_path"
        # "resolution"
        # "cam_position"

        # Or the defaults are going to be used: cow mesh, 64x64 resolution, camera at 2.7 units away from the object.

        self.batch_size = config.get("batch_size", 512)
        # NOTE: additional config: target_seq_len (every seq_len//target_seq_len-th frame recorded)
        self.target_seq_len = config.get("target_seq_len", 1)
        self.save_path = None
        self.store_index = {
            "train": 0,
            "val": 0
        }
        self.load_index = {
            "train": 0,
            "val": 0
        }
        self.pointer = {
            "train": 0,
            "val": 0
        }
        self.z_dim = 200

    def generate_data(self, n_samples, save_path=None, clear=False, phase="train"):
        """Generate one batch of n_samples datapoints and maybe save it

        Args:
            n_samples (int): number of samples to generate
            save_path (str): path to save the data to;
        """
        self.save_path = save_path
        q = torch.randn(n_samples, 4, device=self.device)
        q = (q / q.norm(dim=-1).unsqueeze(-1)).unsqueeze(1)
        mesh = self.mesh.clone()
        # in a bit, I will set up an upper bound for n_samples. Rendering will happen in batches then.
        n_samples_per_batch = 100
        q_index = 0
        I = None
        while n_samples > n_samples_per_batch:
            meshes = mesh.extend(n_samples_per_batch)
            q_temp = q[q_index:(q_index + n_samples_per_batch)]
            R, T = look_at_view_transform(self.cam_position, 0, 180)
            self.cameras = FoVPerspectiveCameras(device=self.mesh.device, R=R, T=T)
            # print(f"Shape of quaternions: {q.shape}")
            # print(f"Shape of mesh: {meshes.verts_padded().shape}")
            rotated_meshes = meshes.update_padded(
                quaternion_apply(q_temp.to(self.device), meshes.verts_padded())
            )
            i = self.renderer(rotated_meshes)[..., :1].permute(0, 3, 1, 2)
            if q_index == 0: 
                I = i
            else: 
                I = torch.cat((I, i), dim=0)
            n_samples -= n_samples_per_batch
            q_index += n_samples_per_batch
        if n_samples > 0:
            meshes = mesh.extend(n_samples)
            q_temp = q[q_index:]
            R, T = look_at_view_transform(self.cam_position, 0, 180)
            self.cameras = FoVPerspectiveCameras(device=self.mesh.device, R=R, T=T)
            rotated_meshes = meshes.update_padded(
                quaternion_apply(q_temp.to(self.device), meshes.verts_padded())
            )
            i = self.renderer(rotated_meshes)[..., :1].permute(0, 3, 1, 2)
            I = torch.cat((I, i), dim=0) if I is not None else i
        
        p_dataset = customDataset(I)
        if save_path is not None:
            index = self.store_index[phase]
            while not clear and os.path.exists(save_path + f"/{phase}_{index}.pth"):
                # print(f"Already exists {save_path + f'/{index}.pth'}")
                index += 1
            torch.save(p_dataset, save_path + f"/{phase}_{index}.pth")
            self.store_index[phase] += 1
            return p_dataset
        return p_dataset

    def load_data(self, n_samples, phase):
        """Load or generate n_samples of previously unseen datapoints"""
        # if self.load_index[phase] > 20:
        #     self.load_index[phase] = 0

        if self.save_path is None:
            # Generate data  on the go.
            return self.generate_data(n_samples)
        else:
            # Find if we can load some data from the save_path
            data = torch.zeros(0, 1, 64, 64, device=self.device) 

            if not os.path.exists(self.save_path + f"/{phase}_{self.load_index[phase]}.pth"):
                left = n_samples
                while left > self.batch_size:
                    p_dataset = self.generate_data(
                        self.batch_size, f"{self.save_path}", phase=phase
                    )
                    left -= self.batch_size
                    self.load_index[phase] += 1
                    data = torch.cat((data, p_dataset.data), dim=0)
                if left > 0:
                    p_dataset = self.generate_data(
                        self.batch_size, f"{self.save_path}", phase=phase
                    )
                    data = torch.cat((data, p_dataset.data), dim=0)
                    self.pointer[phase] = self.batch_size - left
                    self.load_index[phase] += 1

            else:
                left = n_samples
                current = 0
                while (
                    os.path.exists(self.save_path + f"/{phase}_{self.load_index[phase]}.pth")
                    and left > 0
                ):
                    print(f"Loading {phase}_{self.load_index[phase]}.pth")
                    p_dataset = torch.load(
                        self.save_path + f"/{phase}_{self.load_index[phase]}.pth"
                    )
                    if len(p_dataset) - self.pointer[phase] >= left:
                        current = left + self.pointer[phase]
                        data = torch.cat(
                            (data, p_dataset.data[self.pointer[phase] : current]), dim=0
                        )
                        left = 0
                    else:
                        left = left - len(p_dataset) + self.pointer[phase]
                        self.pointer[phase] = 0
                        data = torch.cat((data, p_dataset.data[self.pointer[phase] :]), dim=0)
                    self.load_index[phase] += 1
                while left >= self.batch_size:
                    p_dataset = self.generate_data(
                        self.batch_size, save_path=self.save_path, phase=phase
                    )
                    data = torch.cat((data, p_dataset.data), dim=0)
                    left -= self.batch_size
                    self.load_index[phase] += 1
                    self.pointer[phase] = 0
                if left > 0:
                    # This one is not to be saved.
                    p_dataset = self.generate_data(left, save_path=None)
                    data = torch.cat((data, p_dataset.data), dim=0)
            dataset = customDataset(data)
            return dataset

    def generate(self, n_samples, val_size=None, record=False):
        if val_size is None:
            val_size = int(n_samples * 0.1)

        # print(f"generating {n_samples} samples, {val_size} val set data.")
        # print(f"self.mini_batch_size = {self.mini_batch_size}")
        # print(f"recording: {record}")
        train_data = self.load_data(n_samples, "train")
        val_data = self.load_data(val_size, "val")
        if val_size == 0 and record: 
            self.train_data = train_data.to("cuda")
            train_loader = DataLoader(
                train_data,
                batch_size=self.mini_batch_size,
                shuffle=True,
                generator=torch.Generator(device="cpu"),
            )
            return train_loader, None
        if record:
            self.train_data = train_data.to("cuda")
            self.val_data = val_data.to("cuda")
            train_loader = DataLoader(
                train_data,
                batch_size=self.mini_batch_size,
                shuffle=True,
                generator=torch.Generator(device="cpu"),
            )
            val_loader = DataLoader(
                val_data,
                batch_size=self.mini_batch_size,
                shuffle=True,
                generator=torch.Generator(device="cpu"),
            )
            return train_loader, val_loader
        return train_data, val_data

    def replace(self, replacement_rate):
        """Replace a fraction of the data with new data. 
        - This version of the method doesn't replace any val dataset (why would we?)

        Args:
            replacement_rate (float): the fraction of the data to replace

        Returns:
            train_loader, val_loader (pytorch DataLoader): the new data loaders
        """
        if int(len(self.train_data) * replacement_rate) >= len(self.train_data):
            train_loader, _ = self.generate(len(self.train_data), 0, record=True)
            val_loader = DataLoader(
                self.val_data,
                batch_size=self.mini_batch_size,
                shuffle=True,
                generator=torch.Generator(device="cpu"),
            )
            return train_loader, val_loader

        if self.train_data is None:
            raise ValueError("No data to replace")
        l_t = len(self.train_data)
        
        self.train_data_data = self.train_data.data[
            int(len(self.train_data) * replacement_rate) :
        ]
        
        train_2, _ = self.generate(
            l_t - len(self.train_data_data), 0 
        )

        train_data = torch.cat((self.train_data_data, train_2.data), dim=0)

        self.train_data = customDataset(train_data)

        print(
            f"Replaced {replacement_rate * 100}% of the data, new data size: {len(self.train_data)}, original size: {l_t}"
        )
        train_loader = DataLoader(
            self.train_data,
            batch_size=self.mini_batch_size,
            shuffle=True,
            generator=torch.Generator(device="cpu"),
        )
        val_loader = DataLoader(
            self.val_data,
            batch_size=self.mini_batch_size,
            shuffle=True,
            generator=torch.Generator(device="cpu"),
        )
        return train_loader, val_loader


class customDataset(Dataset):
    def __init__(self, data):
        super(customDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def to(self, device):
        self.data = self.data.to(device)
        return self