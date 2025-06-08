"""PredictorModelDataGenerator classes for the internal model

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
from .PredictorDataset import PredictorDataset
from . import CubeMesh as cube
from importlib import reload
reload(cube)

class LeanPredictorModelDataGenerator(DataGenerator):
    def __init__(self, dynamics_network: DynamicalNetwork, config=None):
        super(LeanPredictorModelDataGenerator, self).__init__(config)
        self.dynamics_network = dynamics_network
        self.batch_size = config.get("batch_size", 512)
        # NOTE: additional config: target_seq_len (every seq_len//target_seq_len-th frame recorded)
        self.stepwise = self.dynamics_network.stepwise
        self.target_seq_len = config.get("target_seq_len", 1)
        # print(f"Target sequence length: {self.target_seq_len}")
        self.step = self.dynamics_network.action_period // self.target_seq_len
        self.save_path = None
        self.store_index = {
            "train": 0,
            "val": 0
        }
        self.load_index = {
            "train": 0,
            "val": 0
        }
        self.z_dim = 200
        self.pointer = {
            "train": 0,
            "val": 0
        }
        self.device = "cuda"

    def generate_data(self, n_samples, save_path=None, clear=False, phase="train"):
        """Generate one batch of n_samples datapoints and maybe save it

        Args:
            n_samples (int): number of samples to generate
            save_path (str): path to save the data to;
        
        Returns:
            PredictorDataset: the generated dataset, it is going to be saved if a save_path is specified.
        """
        # Add recycling by setting the load indices to 0 once they reach some threshold: 
        self.save_path = save_path
        z0 = self.dynamics_network.sample_initial_condition(n_samples)
        _ = self.dynamics_network(z0 * self.dynamics_network.input_scalar)
        q = self.dynamics_network.pred
        if self.dynamics_network.stepwise: 
            q = q[:, (self.step - 1)::self.step, :]
        elif self.target_seq_len > 1: 
            return
        p_dataset = PredictorDataset(z0, z0, q).to(self.device)
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
        """Load or generate n_samples of previously unseen datapoints
        
        Args: 
            n_samples (int): number of samples to load
            phase (str): phase of the data to load
        Returns: 
            PredictorDataset: the loaded dataset. It is going to be saved according to batch-sizeIt is going to be saved according to batch-size
        """
        # if self.load_index[phase] > 20:
        #     self.load_index[phase] = 0

        if self.save_path is None:
            # Generate data  on the go.
            return self.generate_data(n_samples)
        else:
            # Find if we can load some data from the save_path
            z0, q = (
                torch.zeros(0, self.z_dim, device=self.device),
                torch.zeros(0, 4, device=self.device) if not self.stepwise else torch.zeros(0, self.dynamics_network.action_period, 4, device=self.device),
            )
            if not os.path.exists(self.save_path + f"/{phase}_{self.load_index[phase]}.pth"):
                left = n_samples
                while left > self.batch_size:
                    p_dataset = self.generate_data(
                        self.batch_size, f"{self.save_path}", phase=phase
                    )
                    left -= self.batch_size
                    self.load_index[phase] += 1
                    z0 = torch.cat((z0, p_dataset.z0), dim=0)
                    q = torch.cat((q, p_dataset.q), dim=0)
                if left > 0:
                    p_dataset = self.generate_data(
                        self.batch_size, f"{self.save_path}", phase=phase
                    )
                    z0 = torch.cat((z0, p_dataset.z0), dim=0)
                    q = torch.cat((q, p_dataset.q), dim=0)
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
                        z0 = torch.cat(
                            (z0, p_dataset.z0[self.pointer[phase] : current]), dim=0
                        )
                        q = torch.cat((q, p_dataset.q[self.pointer[phase] : current]), dim=0)
                        left = 0
                    else:
                        left = left - len(p_dataset) + self.pointer[phase]
                        self.pointer[phase] = 0
                        z0 = torch.cat((z0, p_dataset.z0[self.pointer[phase] :]), dim=0)
                        q = torch.cat((q, p_dataset.q[self.pointer[phase] :]), dim=0)
                    self.load_index[phase] += 1
                while left >= self.batch_size:
                    p_dataset = self.generate_data(
                        self.batch_size, save_path=self.save_path, phase=phase
                    )
                    z0 = torch.cat((z0, p_dataset.z0), dim=0)
                    q = torch.cat((q, p_dataset.q), dim=0)
                    left -= self.batch_size
                    self.load_index[phase] += 1
                    self.pointer[phase] = 0
                if left > 0:
                    # This one is not to be saved.
                    p_dataset = self.generate_data(left, save_path=None)
                    z0 = torch.cat((z0, p_dataset.z0), dim=0)
                    q = torch.cat((q, p_dataset.q), dim=0)
            return PredictorDataset(z0, z0, q)

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
                generator=torch.Generator(device="cuda"),
            )
            return train_loader, None
        if record:
            self.train_data = train_data.to("cuda")
            self.val_data = val_data.to("cuda")
            train_loader = DataLoader(
                train_data,
                batch_size=self.mini_batch_size,
                shuffle=True,
                generator=torch.Generator(device="cuda"),
            )
            val_loader = DataLoader(
                val_data,
                batch_size=self.mini_batch_size,
                shuffle=True,
                generator=torch.Generator(device="cuda"),
            )
            return train_loader, val_loader
        return train_data, val_data

    def replace_(self, replacement_rate):
        """Replace a fraction of the data with new data.

        Args:
            replacement_rate (float): the fraction of the data to replace

        Returns:
            train_loader, val_loader (pytorch DataLoader): the new data loaders
        """
        if int(len(self.train_data) * replacement_rate) >= len(self.train_data) or int(
            len(self.val_data) * replacement_rate
        ) >= len(self.val_data):
            return self.generate(len(self.train_data), len(self.val_data), record=True)
        if self.train_data is None:
            raise ValueError("No data to replace")
        l_t = len(self.train_data)
        l_v = len(self.val_data)
        self.train_data_z0 = self.train_data.z0[
            int(len(self.train_data) * replacement_rate) :
        ]
        self.train_data_q = self.train_data.q[
            int(len(self.train_data) * replacement_rate) :
        ]
        self.val_data_z0 = self.val_data.z0[
            int(len(self.val_data) * replacement_rate) :
        ]
        self.val_data_q = self.val_data.q[int(len(self.val_data) * replacement_rate) :]

        train_2, val_2 = self.generate(
            l_t - len(self.train_data_z0), l_v - len(self.val_data_z0)
        )

        train_z0 = torch.cat((self.train_data_z0, train_2.z0), dim=0)
        train_q = torch.cat((self.train_data_q, train_2.q), dim=0)

        self.train_data = PredictorDataset(train_z0, train_z0, train_q)

        val_z0 = torch.cat((self.val_data_z0, val_2.z0), dim=0)
        val_q = torch.cat((self.val_data_q, val_2.q), dim=0)

        self.val_data = PredictorDataset(val_z0, val_z0, val_q)

        print(
            f"Replaced {replacement_rate * 100}% of the data, new data size: {len(self.train_data)}, original size: {l_t}"
        )
        train_loader = DataLoader(
            self.train_data,
            batch_size=self.mini_batch_size,
            shuffle=True,
            generator=torch.Generator(device="cuda"),
        )
        val_loader = DataLoader(
            self.val_data,
            batch_size=self.mini_batch_size,
            shuffle=True,
            generator=torch.Generator(device="cuda"),
        )
        return train_loader, val_loader

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
                generator=torch.Generator(device="cuda"),
            )
            return train_loader, val_loader

        if self.train_data is None:
            raise ValueError("No data to replace")
        l_t = len(self.train_data)
        l_v = len(self.val_data)
        self.train_data_z0 = self.train_data.z0[
            int(len(self.train_data) * replacement_rate) :
        ]
        self.train_data_q = self.train_data.q[
            int(len(self.train_data) * replacement_rate) :
        ]
        train_2, _ = self.generate(
            l_t - len(self.train_data_z0), 0 
        )

        train_z0 = torch.cat((self.train_data_z0, train_2.z0), dim=0)
        train_q = torch.cat((self.train_data_q, train_2.q), dim=0)

        self.train_data = PredictorDataset(train_z0, train_z0, train_q)

        print(
            f"Replaced {replacement_rate * 100}% of the data, new data size: {len(self.train_data)}, original size: {l_t}"
        )
        train_loader = DataLoader(
            self.train_data,
            batch_size=self.mini_batch_size,
            shuffle=True,
            generator=torch.Generator(device="cuda"),
        )
        val_loader = DataLoader(
            self.val_data,
            batch_size=self.mini_batch_size,
            shuffle=True,
            generator=torch.Generator(device="cuda"),
        )
        return train_loader, val_loader


class PredictorModelDataGenerator(RendererGenerator):
    def __init__(self, dynamics_network: DynamicalNetwork, config=None):
        # print(f"Memory before init: {torch.cuda.memory_allocated()/1000} KiB")
        super(PredictorModelDataGenerator, self).__init__(config)
        # NOTE: for this to work, the configs must include the renderer specs:
        # "object_path"
        # "resolution"
        # "cam_position"

        # Or the defaults are going to be used: cow mesh, 64x64 resolution, camera at 2.7 units away from the object.

        self.dynamics_network = dynamics_network
        self.batch_size = config.get("batch_size", 512)
        # NOTE: additional config: target_seq_len (every seq_len//target_seq_len-th frame recorded)
        self.stepwise = self.dynamics_network.stepwise
        self.target_seq_len = config.get("target_seq_len", 1)
        self.encoding_model_specs = config.get("encoder_specs", None)
        self.step = self.dynamics_network.action_period // self.target_seq_len

        if self.encoding_model_specs is not None:
            encoding_model_name = self.encoding_model_specs.get("model_name", "PCAEncoder")
            encoding_model_path = self.encoding_model_specs.get("model_path", "Network_models.ForwardModels")
            encoding_model_module = __import__(encoding_model_path, fromlist=[encoding_model_name])
            self.encoding_model_module = getattr(encoding_model_module, encoding_model_name)
            self.encoding_model = self.encoding_model_module(self.encoding_model_specs.get("model_params", {}))
        else: 
            self.encoding_model = None

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
        # print(f"Memory after init: {torch.cuda.memory_allocated()/1000} KiB")
    def recycle_data(self, data_path, save_path=None):
        """Recycles all data in the data_path.

        Args: 
            data_path: a directory containing data to recycle, named as "train_n.pth" or "val_n.pth" 

        """
        import time
        self.save_path = save_path
        for file_name in os.listdir(data_path): 
            if file_name.endswith(".pth"): 
                p_dataset = torch.load(data_path + "/" + file_name)
                self.recycle_data_batch(p_dataset, save_path, file_name)

    def recycle_data_batch(self, p_dataset, save_path, file_name):
        q = p_dataset.q
        seq_len = q.shape[1]
        q = q.reshape(-1, 4)
        z0 = p_dataset.z0
        if self.cubemesh: 
            # in this case, use vertices instead of mesh as the object
            I = self.mesh.quat2verts(q/q.norm(dim=-1, keepdim=True))
            I = I.view(-1, seq_len, I.shape[-1])
            # automatically set the encoding model to None
            self.encoding_model = None
        else:
            n_samples_per_batch = 200
            # print(f"memory before mesh loading: {torch.cuda.memory_allocated()/1000} KiB")
            # self.mesh = load_objs_as_meshes([self.object_path], device=self.device)
            mesh = self.mesh.clone()
            # print(f"memory after mesh loading: {torch.cuda.memory_allocated()/1000} KiB")
            q_index = 0
            I = None
            n_samples_per_batch = 20

            meshes = mesh.extend(n_samples_per_batch * seq_len)
            while n_samples > n_samples_per_batch:
                # print(f"memory after mesh extension: {torch.cuda.memory_allocated()/1000} KiB")
                q_temp = q[q_index:(q_index + (n_samples_per_batch) * seq_len)].unsqueeze(1)
                # print(f"memory after camera setup: {torch.cuda.memory_allocated()/1000} KiB")
                # print(f"Shape of quaternions: {q.shape}")
                # print(f"Shape of mesh: {meshes.verts_padded().shape}")
                rotated_meshes = meshes.update_padded(
                    quaternion_apply(q_temp.to(self.device), meshes.verts_padded())
                )
                # print(f"memory after mesh rotation: {torch.cuda.memory_allocated()/1000} KiB")
                imag = self.renderer(rotated_meshes)[..., :1].permute(0, 3, 1, 2)
                del rotated_meshes
                imag = imag.reshape(-1, seq_len, 1, 64, 64)
                # print(f"memory after image generation: {torch.cuda.memory_allocated()/1000} KiB")
                if self.encoding_model is not None:
                    with torch.no_grad(): 
                        i = self.encoding_model(imag)
                    del imag
                else: 
                    i = imag

                # print(f"memory after i generation: {torch.cuda.memory_allocated()/1000} KiB")
                if q_index == 0: 
                    I = i
                else: 
                    I = torch.cat((I, i), dim=0)
                # print(f"memory after I generation: {torch.cuda.memory_allocated()/1000} KiB")
                n_samples -= n_samples_per_batch
                q_index += n_samples_per_batch * seq_len
                del i, q_temp
                torch.cuda.empty_cache()
                # print(f"memory after deletion: {torch.cuda.memory_allocated()/1000} KiB")
            del meshes
            if n_samples > 0:
                meshes = mesh.extend(n_samples * seq_len)
                q_temp = q[q_index:].unsqueeze(1)
                rotated_meshes = meshes.update_padded(
                    quaternion_apply(q_temp.to(self.device), meshes.verts_padded())
                )
                i = self.renderer(rotated_meshes)[..., :1].permute(0, 3, 1, 2)
                i = i.reshape(-1, seq_len, 1, 64, 64)
                if self.encoding_model is not None:
                    i = self.encoding_model(i)
                I = torch.cat((I, i), dim=0) if I is not None else i

        q = q.reshape(-1, seq_len, 4)
        p_dataset = PredictorDataset(z0, I, q)
        phase = file_name.split("_")[0]
        index = int(file_name.split("_")[1].split(".")[0])
        torch.save(p_dataset, save_path + f"/{phase}_{index}.pth")
        self.store_index[phase] = index + 1
        return p_dataset

    def generate_data(self, n_samples, save_path=None, clear=False, phase="train"):
        """Generate one batch of n_samples datapoints and maybe save it

        Args:
            n_samples (int): number of samples to generate
            save_path (str): path to save the data to;
        """
        import time
        self.save_path = save_path
        n_samples_per_batch = 200
        current = n_samples
        z0 = None
        print(f"Generating {n_samples} samples")
        time_start = time.time()
        while current > n_samples_per_batch:
            z0_temp = self.dynamics_network.sample_initial_condition(n_samples_per_batch)
            if current == n_samples:
                z0 = z0_temp
            else: 
                z0 = torch.cat((z0, z0_temp), dim=0)
            del z0_temp
            current -= n_samples_per_batch
        if current > 0:
            z0_temp = self.dynamics_network.sample_initial_condition(current)
            if z0 is not None:
                z0 = torch.cat((z0, z0_temp), dim=0)
            else: 
                z0 = z0_temp
            del z0_temp

        # print(f"Initial condition: {z0.shape}")
        out = self.dynamics_network(z0 * self.dynamics_network.input_scalar)
        # print(f"Predicted: {self.dynamics_network.pred.shape}")
        q = self.dynamics_network.pred
        del out
        self.dynamics_network.out = None
        self.dynamics_network.pred = None
        # print(f"memory after prediction: {torch.cuda.memory_allocated()/1000} KiB")
        if self.dynamics_network.stepwise: 
            q = q[:, (self.step - 1)::self.step, :]
            seq_len = q.shape[1]
            q = q.reshape(-1, 4)
        forward_time = time.time()
        print(f"Time for forward network dynamics : {forward_time - time_start}")
        # print(f"memory before mesh loading: {torch.cuda.memory_allocated()/1000} KiB")

        # print(f"memory after mesh loading: {torch.cuda.memory_allocated()/1000} KiB")
        q_index = 0

        I = None

        if self.cubemesh: 
            # in this case, use vertices instead of mesh as the object
            I = self.mesh.quat2verts(q/q.norm(dim=-1, keepdim=True))
            I = I.view(-1, seq_len, I.shape[-1])
            # automatically set the encoding model to None
            self.encoding_model = None

        else:
            mesh = self.mesh.clone()
            n_samples_per_batch = 20
            meshes = mesh.extend(n_samples_per_batch * seq_len)
            while n_samples > n_samples_per_batch:
                # print(f"memory after mesh extension: {torch.cuda.memory_allocated()/1000} KiB")
                q_temp = q[q_index:(q_index + (n_samples_per_batch) * seq_len)].unsqueeze(1)
                R, T = look_at_view_transform(self.cam_position, 0, 180)
                self.cameras = FoVPerspectiveCameras(device=self.mesh.device, R=R, T=T)
                # print(f"memory after camera setup: {torch.cuda.memory_allocated()/1000} KiB")
                # print(f"Shape of quaternions: {q.shape}")
                # print(f"Shape of mesh: {meshes.verts_padded().shape}")
                rotated_meshes = meshes.update_padded(
                    quaternion_apply(q_temp.to(self.device), meshes.verts_padded())
                )
                # print(f"memory after mesh rotation: {torch.cuda.memory_allocated()/1000} KiB")
                imag = self.renderer(rotated_meshes)[..., :1].permute(0, 3, 1, 2)
                del rotated_meshes
                # self.renderer = MeshRenderer(
                #     rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings),
                #     shader=SoftPhongShader(device=self.mesh.device, lights=self.lights, cameras=self.cameras)
                # )
                imag = imag.reshape(-1, seq_len, 1, 64, 64)
                # print(f"memory after image generation: {torch.cuda.memory_allocated()/1000} KiB")
                if self.encoding_model is not None:
                    with torch.no_grad(): 
                        i = self.encoding_model(imag)
                    del imag
                else: 
                    i = imag

                # print(f"memory after i generation: {torch.cuda.memory_allocated()/1000} KiB")
                if q_index == 0: 
                    I = i
                else: 
                    I = torch.cat((I, i), dim=0)
                # print(f"memory after I generation: {torch.cuda.memory_allocated()/1000} KiB")
                n_samples -= n_samples_per_batch
                q_index += n_samples_per_batch * seq_len
                del i, q_temp
                torch.cuda.empty_cache()
                # print(f"memory after deletion: {torch.cuda.memory_allocated()/1000} KiB")
            del meshes
            if n_samples > 0:
                meshes = mesh.extend(n_samples * seq_len)
                q_temp = q[q_index:].unsqueeze(1)
                R, T = look_at_view_transform(self.cam_position, 0, 180)
                self.cameras = FoVPerspectiveCameras(device=self.mesh.device, R=R, T=T)
                rotated_meshes = meshes.update_padded(
                    quaternion_apply(q_temp.to(self.device), meshes.verts_padded())
                )
                i = self.renderer(rotated_meshes)[..., :1].permute(0, 3, 1, 2)
                i = i.reshape(-1, seq_len, 1, 64, 64)
                if self.encoding_model is not None:
                    i = self.encoding_model(i)
                I = torch.cat((I, i), dim=0) if I is not None else i

        q = q.reshape(-1, seq_len, 4)
        # print(f"Shape of I: {I.shape}")

        #  # display the first batch of 5 images to check:
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(1, 5, figsize=(20, 4))
        # for i in range(5):
        #     axs[i].imshow(I[0, i, 0].cpu().numpy())

        p_dataset = PredictorDataset(z0, I, q)
        print(f"Time for rendering the dataset: {time.time() - forward_time}")
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
            # z0, I, q = (
            #     torch.zeros(0, self.z_dim, device=self.device),
            #     torch.zeros(0, self.target_seq_len, self.encoding_model.encoding_dim, device=self.device),
            #     torch.zeros(0, 4, device=self.device) if not self.stepwise else torch.zeros(0, self.target_seq_len, 4, device=self.device),
            # )
            z0, I, q = None, None, None
            begin = True
            # print(self.save_path + f"/{phase}_{self.load_index[phase]}.pth")
            if not os.path.exists(self.save_path + f"/{phase}_{self.load_index[phase]}.pth"):
                left = n_samples
                # print(self.batch_size)
                while left > self.batch_size:
                    p_dataset = self.generate_data(
                        self.batch_size, f"{self.save_path}", phase=phase
                    )
                    left -= self.batch_size
                    self.load_index[phase] += 1
                    if begin: 
                        z0, I, q = p_dataset.z0, p_dataset.I, p_dataset.q
                        begin = False
                    else:
                        z0 = torch.cat((z0, p_dataset.z0), dim=0)
                        I = torch.cat((I, p_dataset.I), dim=0)
                        q = torch.cat((q, p_dataset.q), dim=0)
                if left > 0:
                    p_dataset = self.generate_data(
                        self.batch_size, f"{self.save_path}", phase=phase
                    )
                    if begin: 
                        z0, I, q = p_dataset.z0, p_dataset.I, p_dataset.q
                        begin = False
                    else:
                        z0 = torch.cat((z0, p_dataset.z0), dim=0)
                        I = torch.cat((I, p_dataset.I), dim=0)
                        q = torch.cat((q, p_dataset.q), dim=0)
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
                    print(len(p_dataset))
                    print(self.pointer[phase])
                    if len(p_dataset) - self.pointer[phase] >= left:
                        current = left + self.pointer[phase]
                        if begin: 
                            z0, I, q = p_dataset.z0[self.pointer[phase] : current], p_dataset.I[self.pointer[phase] : current], p_dataset.q[self.pointer[phase] : current]
                            begin = False
                        else:
                            z0 = torch.cat(
                                (z0, p_dataset.z0[self.pointer[phase] : current]), dim=0
                            )
                            I = torch.cat((I, p_dataset.I[self.pointer[phase] : current]), dim=0)
                            q = torch.cat((q, p_dataset.q[self.pointer[phase] : current]), dim=0)
                        left = 0
                    else:
                        left = left - len(p_dataset) + self.pointer[phase]
                        self.pointer[phase] = 0
                        if begin: 
                            z0, I, q = p_dataset.z0[self.pointer[phase] :], p_dataset.I[self.pointer[phase] :], p_dataset.q[self.pointer[phase] :]
                            begin = False
                        else:
                            z0 = torch.cat((z0, p_dataset.z0[self.pointer[phase] :]), dim=0)
                            I = torch.cat((I, p_dataset.I[self.pointer[phase] :]), dim=0)
                            q = torch.cat((q, p_dataset.q[self.pointer[phase] :]), dim=0)
                    self.load_index[phase] += 1
                while left >= self.batch_size:
                    p_dataset = self.generate_data(
                        self.batch_size, save_path=self.save_path, phase=phase
                    )
                    if begin:
                        z0, I, q = p_dataset.z0, p_dataset.I, p_dataset.q
                        begin = False
                    else: 
                        z0 = torch.cat((z0, p_dataset.z0), dim=0)
                        I = torch.cat((I, p_dataset.I), dim=0)
                        q = torch.cat((q, p_dataset.q), dim=0)
                    left -= self.batch_size
                    self.load_index[phase] += 1
                    self.pointer[phase] = 0
                if left > 0:
                    # This one is not to be saved.
                    p_dataset = self.generate_data(left, save_path=None)
                    if begin:
                        z0, I, q = p_dataset.z0, p_dataset.I, p_dataset.q
                        begin = False
                    else:
                        z0 = torch.cat((z0, p_dataset.z0), dim=0)
                        I = torch.cat((I, p_dataset.I), dim=0)
                        q = torch.cat((q, p_dataset.q), dim=0)
            return PredictorDataset(z0, I, q)

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

    def replace_(self, replacement_rate):
        """Replace a fraction of the data with new data.

        Args:
            replacement_rate (float): the fraction of the data to replace

        Returns:
            train_loader, val_loader (pytorch DataLoader): the new data loaders
        """
        if int(len(self.train_data) * replacement_rate) >= len(self.train_data) or int(
            len(self.val_data) * replacement_rate
        ) >= len(self.val_data):
            return self.generate(len(self.train_data), len(self.val_data), record=True)
        if self.train_data is None:
            raise ValueError("No data to replace")
        l_t = len(self.train_data)
        l_v = len(self.val_data)
        self.train_data_z0 = self.train_data.z0[
            int(len(self.train_data) * replacement_rate) :
        ]
        self.train_data_q = self.train_data.q[
            int(len(self.train_data) * replacement_rate) :
        ]
        self.val_data_z0 = self.val_data.z0[
            int(len(self.val_data) * replacement_rate) :
        ]
        self.val_data_q = self.val_data.q[int(len(self.val_data) * replacement_rate) :]

        train_2, val_2 = self.generate(
            l_t - len(self.train_data_z0), l_v - len(self.val_data_z0)
        )

        train_z0 = torch.cat((self.train_data_z0, train_2.z0), dim=0)
        train_q = torch.cat((self.train_data_q, train_2.q), dim=0)

        self.train_data = PredictorDataset(train_z0, train_z0, train_q)

        val_z0 = torch.cat((self.val_data_z0, val_2.z0), dim=0)
        val_q = torch.cat((self.val_data_q, val_2.q), dim=0)

        self.val_data = PredictorDataset(val_z0, val_z0, val_q)

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
        self.train_data_z0 = self.train_data.z0[
            int(len(self.train_data) * replacement_rate) :
        ]
        self.train_data_I = self.train_data.I[
            int(len(self.train_data) * replacement_rate) :
        ]
        self.train_data_q = self.train_data.q[
            int(len(self.train_data) * replacement_rate) :
        ]
        train_2, _ = self.generate(
            l_t - len(self.train_data_z0), 0 
        )

        train_z0 = torch.cat((self.train_data_z0, train_2.z0), dim=0)
        train_I = torch.cat((self.train_data_I, train_2.I), dim=0)
        train_q = torch.cat((self.train_data_q, train_2.q), dim=0)

        self.train_data = PredictorDataset(train_z0, train_I, train_q)

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
