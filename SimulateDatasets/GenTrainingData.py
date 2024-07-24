""""This module handles generation of training data

For specification on what each task_id maps to, see Tasks.md

"""
from .RotationDataset import RotationDataset
import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import Rotations as Rot
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
import matplotlib.pyplot as plt
from numba import jit


def gen_featured_data(q_s, training_config):
    """Accepts an array of quaternions as target quaternions and returns the corresponding features

    Args:
        q_s(np.array): An array of quaternions of shape (n,4)
        training_config(dict): A dictionary containing the configuration for the training data generation

    Returns:
        RotationDataset: A dataset containing the features and targets
    """
    dataset = None
    match training_config["task_id"]:
        case "0.1q":
            dataset =  _build_task_01q_featured(q_s, training_config)
        case "0.2q":
            dataset =  _build_task_02q_featured(q_s, training_config)
        case "1.1":
            dataset =  _build_task_11_featured(q_s, training_config)
        case _:
            raise NotImplementedError

    if "data_save_path" in training_config and training_config["data_save_path"] is not None:
        torch.save(dataset, training_config["data_save_path"])
    return dataset


def gen_training_data(training_config, overwrite = None):
    """handle to generate training data for a given task"""
    dataset = None
    saved_data_path = training_config.get("data_save_path", None)
    if saved_data_path is not None and saved_data_path.endswith(".pth"):
        saved_data_path = saved_data_path[:-4]
        training_config["data_save_path"] = saved_data_path
        print('Warning: new paradigm. Now training configs should be without the extension as we are adding it automatically.')
    for phase in ["train", "train2", "test"]:
        print(training_config["data_save_path"] + f"_{phase}.pth")
        try:
            exists = os.path.isfile(training_config["data_save_path"] + f"_{phase}.pth")
        except Exception as e:
            exists = False

        if exists and overwrite is None:
            message = f"Data already exists at {training_config['data_save_path']}_{phase}.pth\n"\
                      f"Do you want to overwrite it? (y/n): "
            response = input(message)
            if response == "n":
                continue

            elif response != "y":
                raise ValueError("Invalid response. Skipping this phase.")
            else:
                print("Overwriting data.")
        elif overwrite is False: #skip this phase
            continue
        match training_config["task_id"]:
            case "0.1q":
                dataset = _build_task_01q(training_config)
            case "0.2q":
                dataset = _build_task_02q(training_config)
            case "1.1":
                dataset =  _build_task_11(training_config)
            case _:
                raise NotImplementedError
        if training_config.get("data_save_path", None) is not None:
            torch.save(dataset, training_config["data_save_path"] + f"_{phase}.pth")
            if training_config['task_id'] == "1.1":
                begin_data = dataset.data[:, :, :, :]
                image_path = training_config["data_save_path"].removesuffix(".pth") + "_images" + f"_{phase}"
                os.makedirs(image_path, exist_ok = True)
                for i in range(begin_data.shape[0]):
                    plt.imsave(image_path + f"\\{i}.png", begin_data[i, :,  :, :].permute(1,2,0).cpu().numpy())
    return dataset


def _build_task_01q(training_config):
    """Build a dataset for task 0.1q"""
    batch_size, seq_len, prep_phase = (
        training_config["batch_size"],
        training_config["seq_len"],
        training_config["prep_phase"],
    )
    features = torch.zeros((batch_size, seq_len, 8))
    target = torch.zeros((batch_size, 4))
    for i in range(batch_size):
        target[i] = (torch.rand((4)) - 0.5) * 2
        target[i][0] = abs(target[i][0])
        target[i] = target[i] / torch.norm(target[i])
        for j in range(prep_phase):
            features[i, j, :] = torch.cat([target[i], -target[i]], dim = 0)

    return RotationDataset(features, target)

def _build_task_02q(training_config):
    """Build a dataset for task 0.2q"""
    batch_size, seq_len, prep_phase = (
        training_config["batch_size"],
        training_config["seq_len"],
        training_config["prep_phase"],
    )
    features = torch.zeros((batch_size, seq_len, 8))
    target = torch.zeros((batch_size, 4))
    for i in range(batch_size):
        target[i] = (torch.rand((4)) - 0.5) * 2
        target[i][0] = abs(target[i][0])
        target[i] = target[i] / torch.norm(target[i])
        for j in range(prep_phase):
            conj = Rot.q_conjugate(target[i])
            features[i, j, :] = torch.cat([conj, -conj], dim = -1)

    return RotationDataset(features, target)


def _build_task_01q_featured(q_s, training_config):
    seq_len, prep_phase = (
        training_config["seq_len"],
        training_config["prep_phase"],
    )
    batch_size = q_s.shape[0]
    features = torch.zeros((batch_size, seq_len, 8), dtype = torch.float32)
    target = torch.tensor(q_s, dtype = torch.float32)
    for i in range(batch_size):
        for j in range(prep_phase):
            if target[i][0] < 0:
                target[i] = -target[i]
            features[i, j, :] = torch.cat([target[i], -target[i]], dim = 0)

    return RotationDataset(features, target)

def _build_task_02q_featured(q_s, training_config):
    seq_len, prep_phase = (
        training_config["seq_len"],
        training_config["prep_phase"],
    )
    batch_size = q_s.shape[0]
    features = torch.zeros((batch_size, seq_len, 8), dtype = torch.float32)
    target = torch.tensor(q_s, dtype = torch.float32)
    for i in range(batch_size):
        for j in range(prep_phase):
            conj = Rot.q_conjugate(target[i])
            features[i, j, :] = torch.cat([conj, -conj], dim = -1)

    return RotationDataset(features, target)

def _build_task_11_featured(q_s, training_config):
    """Big step up task. Here, we use images as input

    Returns:
        RotationDataset:
            features: (batch_size, 1, resolution, resolution)
            target: (batch_size, 4)
    """
    batch_size = q_s.shape[0]
    object_path, resolution = (
        training_config.get("object_path", None),
        training_config.get("resolution", 64), # currently uses only square images for simplicity
    )
    if object_path is None:
        raise ValueError("Please specify object path")
    device = "cpu"
    mesh = load_objs_as_meshes([object_path], device = device)
    R, T = look_at_view_transform(training_config.get('cam_position', 2.7), 0, 180)
    cameras = FoVPerspectiveCameras(device = mesh.device, R = R, T = T)
    raster_settings = RasterizationSettings(
        image_size = resolution,
        blur_radius = 0.0,
        faces_per_pixel = 1,
    )
    lights = PointLights(device = mesh.device, location = [[0.0, 0.0, -3.0]])
    renderer = MeshRenderer(
        rasterizer = MeshRasterizer(cameras = cameras, raster_settings = raster_settings),
        shader = SoftPhongShader(device = mesh.device, lights = lights, cameras=cameras)
    )

    # features = torch.zeros((batch_size, 1, resolution, resolution), device=device)
    features = torch.zeros((batch_size, 3, resolution, resolution), device=device)
    target = torch.tensor(q_s, dtype = torch.get_default_dtype(), device=device)
    for i in range(batch_size):
        target[i] = target[i] / torch.norm(target[i])
        rotated_mesh = mesh.clone()
        rotated_mesh = rotated_mesh.update_padded(quaternion_apply(target[i], rotated_mesh.verts_padded()))


        #print(renderer(rotated_mesh).shape)
        # features[i, 0, :, :] = torch.tensordot(renderer(rotated_mesh)[0, ..., :3], torch.tensor([0.2989, 0.5870, 0.1140]).to(device), dims = 1)
        # NO MORE DOT PRODUCT
        features[i, :, :, :] = renderer(rotated_mesh)[0, ..., :3].permute(2,0,1)
    return RotationDataset(features.cpu(), target.cpu())

def gen_AR_quaternion(n_ts):
    """Generate a quaternion time series with autoregressive properties

    Args:
        n_ts(int): The number of time steps

    Returns:
        np.array: An array of quaternions of shape (n_ts, 4)
    """
    qs_t = np.random.normal(0, 1, (n_ts, 4))
    qs_t = qs_t / np.sqrt(np.sum(qs_t**2, axis = 1, keepdims = True))
    for t in range(1, n_ts):
      step = np.random.normal(0, 0.4, 3)
      theta = np.sqrt(np.sum(step**2))
      v = step / theta
      qt = np.concatenate([np.ones(1)*np.cos(theta), np.sin(theta)*v])
      qs_t[t, :] = Rot.q_mult_np(qt, qs_t[t-1, :])
      #print(qs_t[t, :], np.sum(qs_t[t, :]**2), theta)
    qs_t = np.sign(qs_t[:, :1]) * qs_t #consistent sign

    dt_dists = 4*( 1 - np.sum(qs_t[:-1, :] * qs_t[1:, :], axis = 1)**2 )
    print('consecutive displacements:', np.quantile(dt_dists, [0.1, 0.25, 0.5, 0.75, 0.9]))
    return qs_t

def _build_task_11(training_config, phase = "train"):
    """Big step up task. Here, we use images as input

    Returns:
        RotationDataset:
            features: (batch_size, 1, resolution, resolution)
            target: (batch_size, 4)
    """
    use_AR = training_config.get("use_AR", False)
    batch_size =0
    match phase:
        case "train":
            batch_size = training_config.get('batch_size')
        case "train2":
            batch_size = training_config.get('batch_size2', training_config.get('batch_size'))
        case "test":
            batch_size = training_config.get('test_size', training_config.get('batch_size'))
        case _:
            raise NotImplementedError

    if use_AR:
        return _build_task_11_featured(gen_AR_quaternion(batch_size), training_config)

    object_path, resolution = (
        training_config.get("object_path", None),
        training_config.get("resolution", 64), # currently uses only square images for simplicity
    )
    if object_path is None:
        raise ValueError("Please specify object path")
    device = "cpu"
    mesh = load_objs_as_meshes([object_path], device = device)
    R, T = look_at_view_transform(training_config.get('cam_position', 2.7), 0, 180)
    cameras = FoVPerspectiveCameras(device = mesh.device, R = R, T = T)
    raster_settings = RasterizationSettings(
        image_size = resolution,
        blur_radius = 0.0,
        faces_per_pixel = 1,
    )
    lights = PointLights(device = mesh.device, location = [[0.0, 0.0, -3.0]])
    renderer = MeshRenderer(
        rasterizer = MeshRasterizer(cameras = cameras, raster_settings = raster_settings),
        shader = SoftPhongShader(device = mesh.device, lights = lights, cameras=cameras)
    )

    # features = torch.zeros((batch_size, 1, resolution, resolution), device=device)
    features = torch.zeros((batch_size, 3, resolution, resolution), device=device)
    target = torch.zeros((batch_size, 4), device=device)
    for i in range(batch_size):
        target[i] = (torch.rand((4)) - 0.5) * 2
        target[i][0] = abs(target[i][0])
        target[i] = target[i] / torch.norm(target[i])
        rotated_mesh = mesh.clone()
        rotated_mesh = rotated_mesh.update_padded(quaternion_apply(target[i], rotated_mesh.verts_padded()))


        #print(renderer(rotated_mesh).shape)
        # features[i, 0, :, :] = torch.tensordot(renderer(rotated_mesh)[0, ..., :3], torch.tensor([0.2989, 0.5870, 0.1140]).to(device), dims = 1)
        # NO MORE DOT PRODUCT
        features[i, :, :, :] = renderer(rotated_mesh)[0, ..., :3].permute(2,0,1)
    return RotationDataset(features.cpu(), target.cpu())

