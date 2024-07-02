""""This module handles generation of training data

For specification on what each task_id maps to, see Tasks.md

"""
from .RotationDataset import RotationDataset
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import Rotations as Rot


def gen_training_data(training_config):
    """handle to generate training data for a given task"""
    dataset = None
    match training_config["task_id"]:
        case "0.1q":
            dataset = _build_task_01q(training_config)
        case "0.2q":
            dataset = _build_task_02q(training_config)
        case _:
            raise NotImplementedError
    torch.save(dataset, training_config["save_path"])
    return dataset


def _build_task_01q(training_config):
    """Build a dataset for task 0.1q"""
    batch_size, seq_len, prep_phase = (
        training_config["batch_size"],
        training_config["seq_len"],
        training_config["prep_phase"],
    )
    features = torch.zeros((batch_size, seq_len, 4))
    target = torch.zeros((batch_size, 4))
    for i in range(batch_size):
        target[i] = torch.rand((4))
        target[i] = target[i] / torch.norm(target[i])
        for j in range(prep_phase):
            features[i, j, :] = target[i]

    return RotationDataset(features, target)

def _build_task_02q(training_config):
    """Build a dataset for task 0.2q"""
    batch_size, seq_len, prep_phase = (
        training_config["batch_size"],
        training_config["seq_len"],
        training_config["prep_phase"],
    )
    features = torch.zeros((batch_size, seq_len, 4))
    target = torch.zeros((batch_size, 4))
    for i in range(batch_size):
        target[i] = torch.rand((4))
        target[i] = target[i] / torch.norm(target[i])
        for j in range(prep_phase):
            features[i, j, :] = Rot.q_conjugate(target[i])

    return RotationDataset(features, target)