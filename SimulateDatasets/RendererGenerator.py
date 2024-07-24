"""This module takes care of online generation of datasets for the rendered images
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import random_split
from . import DataGenerator as d
from .RotationDataset import RotationDataset
from importlib import reload
from utils.path_settings import *
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
    FoVPerspectiveCameras
)
reload(d)


class RendererGenerator(d.DataGenerator):
    def __init__(self, configs):
        super(RendererGenerator, self).__init__(additional_config=configs)
        self.additional_config = configs
        self.output_noise = configs.get("output_noise", 0) # noise to the output, in theta
        self.object_path = configs.get("object_path", OBJECT_PATH + "\\cow_mesh\\cow.obj")
        self.resolution = configs.get("resolution", 64)
        self.cam_position = configs.get("cam_position", 2.7)

        self.device = "cuda"
        self.mesh = load_objs_as_meshes([self.object_path], device=self.device)
        R, T = look_at_view_transform(self.cam_position, 0, 180)
        self.cameras = FoVPerspectiveCameras(device=self.mesh.device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=self.resolution,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        lights = PointLights(device=self.mesh.device, location=[[0.0, 0.0, -3.0]])
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=self.mesh.device, lights=lights, cameras=self.cameras)
        )

        if isinstance(self.output_noise, tuple):
            # In this case we have specified the output noise to the angular displacement and the noise to the axis separately.
            self.output_noise_theta = self.output_noise[0]
            self.output_noise_axis = self.output_noise[1]
        else:
            self.output_noise_theta = self.output_noise
            self.output_noise_axis = self.output_noise
        self.mini_batch_size = self.additional_config.get("mini_batch_size", 128)
        self.train_data = None
        self.val_data = None

    def generate_data(self, n_samples):
        """Generate a dataset of n_samples."""
        thetas = torch.rand(n_samples, 1, 1) * torch.pi
        axes = torch.rand(n_samples, 1, 3) * 2 - 1
        thetas_label = (thetas + torch.randn_like(thetas) * self.output_noise_theta)
        axes_label = (axes + torch.randn_like(axes) * self.output_noise_axis) * ((thetas_label > torch.pi).float() * -2 + 1)
        thetas_label = thetas_label * (thetas_label < torch.pi) + (2 * torch.pi - thetas_label) * (thetas_label > torch.pi)
        q_s = torch.cat((torch.cos(thetas/2), torch.sin(thetas/2) * axes / axes.norm(dim = -1).unsqueeze(-1)), dim=-1)
        # for i in range(n_samples):
        #     q = q_s[i]
        #     R = quaternion_to_matrix(q.to(self.device))
        #     images = self.renderer(meshes_world=self.mesh, R=R, T=self.cameras.get_camera_center())[0, ..., :3].permute(2, 0, 1)
        #     features[i] = images
        mesh = self.mesh.clone()
        meshes = mesh.extend(n_samples)
        R, T = look_at_view_transform(self.cam_position, 0, 180)
        self.cameras = FoVPerspectiveCameras(device=self.mesh.device, R=R, T=T)
        rotated_meshes = meshes.update_padded(quaternion_apply(q_s.to(self.device), mesh.verts_padded()))
        # R = quaternion_to_matrix(q_s.to(self.device)).squeeze(1)
        # print(R)
        # images = self.renderer(meshes_world=meshes, R=R, T=self.cameras.get_camera_center().repeat(n_samples, 1))[..., :3].permute(0, 3, 1, 2)
        images = self.renderer(rotated_meshes)[..., :3].permute(0, 3, 1, 2)
        features = images
        labels = torch.cat((torch.cos(thetas_label/2), torch.sin(thetas_label/2) * axes_label / axes_label.norm(dim = -1).unsqueeze(-1)), dim=-1)
        return(RotationDataset(features, labels.squeeze(1)))

