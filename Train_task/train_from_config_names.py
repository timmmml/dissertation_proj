"""Implements batched training for config names

(* modify the build_config function to alter default training configurations *)

Args:
    network_name: a list of strings specifying netwoks to train
        Options are {
            "GRU_1layer_8hidden", # Baseline
            "RNN_1layer_8hidden", # Baseline
            "GRU_1layer_16hidden", # Investigate effect of hidden size
            "RNN_1layer_16hidden", # Investigate effect of hidden size
            "GRU_1layer_32hidden", # Investigate effect of hidden size
            "RNN_1layer_32hidden",  # Investigate effect of hidden size
            "GRU_2layer_8hidden", # Investigate effect of another layer
            "RNN_2layer_8hidden", # Investigate effect of another layer
            "FC_16_GRU_1layer_8hidden", # Investigate effect of fully connected layer b4 recurrence
            "FC_16_RNN_1layer_8hidden", # Investigate effect of fully connected layer b4 recurrence
            "FC_32_GRU_1layer_8hidden", # Investigate effect of fully connected layer b4 recurrence
            "FC_32_RNN_1layer_8hidden", # Investigate effect of fully connected layer b4 recurrence
        }
    task_name: a string for the task to train on
        Options are {
            "0.1q", # Forward task, quaternion
            "0.2q", # Backward task, quaternion
        }

Returns:
    trained_paths: a dictionary specifying the paths of the best trained networks.
"""
#%%
from utils import goto_project_root
from utils.path_settings import MODEL_SAVE_PATH, DATA_PATH, LOG_PATH, CONFIG_PATH, OBJECT_PATH
from torch.utils.tensorboard import SummaryWriter
import SimulateDatasets.GenTrainingData as g
from utils import create_splits, get_dataloaders, force_remove_dir
from Network_models import Trainer as t
from importlib import reload
import os
reload(t)
reload(g)
import time
import torch
import numpy as np
from pytorch3d.transforms import so3_relative_angle
from pytorch3d.transforms import quaternion_to_matrix
import json

def train_from_config_names(network_name_epochs, task_name, re_train = False, special_names = None):
    for network_name in network_name_epochs.keys():
        t = train_from_config_name(network_name, network_name_epochs[network_name], task_name, re_train, special_names)
        return t

def train_from_config_name(network_name, epochs, task_name, re_train, special_names):
    configs = build_config(network_name, task_name, re_train = re_train, special_names = special_names)
    if configs is None:
        print("Config already exists and re_train set to True. Skipping training.")
        return None
    print(configs["training_config"]["data_save_path"][len(DATA_PATH)+1:])
    print(os.listdir(DATA_PATH))
    # Check if there is already simulated training data for this task.
    if configs["training_config"]["data_save_path"][len(DATA_PATH)+1:] not in os.listdir(DATA_PATH):
        # Generate the training data
        print(f"Generating training data for {network_name} on task {task_name}")
        data = g.gen_training_data(configs["training_config"])
    else:
        print(f"Loading training data for {network_name} on task {task_name}")
        data = torch.load(configs["training_config"]["data_save_path"])

    dataloaders = get_dataloaders(data, batch_size = configs["training_config"]["mini_batch_size"], k = 5)
    trainer = t.Trainer(configs)

    with (open(
            CONFIG_PATH + f"\\{network_name}_{task_name}_configs.json", "w")) as f:
        json.dump(configs, f, indent=4)

    print(f"Training {network_name} on task {task_name}, saving to {configs['save_path']}; copy the below for logs")
    print(f"tensorboard --logdir={configs['log_path']}")
    for i, (train_loader, val_loader) in enumerate(dataloaders):
        print(f"Training split {i + 1}")
        if i:
            trainer.refresh()
        sub_log_path = configs["log_path"] + f"\\split_{i + 1}"
        sub_check_path = configs["check_path"] + f"\\split_{i + 1}"

        force_remove_dir(sub_log_path) # Probably redundant but helps to ensure no old logs are kept
        os.makedirs(sub_log_path, exist_ok = True)
        os.makedirs(sub_check_path, exist_ok = True)

        trainer.train(train_loader,
                          val_loader,
                          epochs=epochs,
                          save_path = configs["save_path"] + f"\\model_{i+1}.pth",
                          check_path= sub_check_path,
                          log_path = sub_log_path)

        trainer.save_model(configs["save_path"] + f"\\model_{i+1}.pth", full=1)
    trainer.load_best_model()
    print(f"Training {network_name} on task {task_name} complete. Saving best model to {configs['save_path']}.")
    trainer.save_model(configs["save_path"] + f"\\best_model.pth", full=1)


def build_config(network_name, task_name, re_train, retrieve_config = False, special_names = None):
    # The last argument is for retrieving the config if it already exists
    pretraining_networks = [
        "GRU_1layer_8hidden",
        "RNN_1layer_8hidden",
        "GRU_1layer_16hidden",
        "RNN_1layer_16hidden",
        "GRU_1layer_32hidden",
        "RNN_1layer_32hidden",
        "GRU_2layer_8hidden",
        "RNN_2layer_8hidden",
        "FC_16_GRU_1layer_8hidden",
        "FC_16_RNN_1layer_8hidden",
        "FC_32_GRU_1layer_8hidden",
        "FC_32_RNN_1layer_8hidden",
        "FC_16_GRU_1layer_16hidden",
        "FC_16_RNN_1layer_16hidden",
        "FC_32_GRU_1layer_16hidden",
        "FC_32_RNN_1layer_16hidden",
        "FC_16_GRU_1layer_32hidden",
        "FC_16_RNN_1layer_32hidden",
        "FC_32_GRU_1layer_32hidden",
        "FC_32_RNN_1layer_32hidden",
    ]  # This is incomplete. The plan is to investigate across these and implement silence-phase activity
    # suppression on the best performing networks to investigate effects on training.
    # Then can investigate effects of overall regularisation.
    cnn_networks = [
        "CNN_FC_16_GRU_1layer_8hidden",
    ]
    pretraining_tasks = ["0.1q", "0.2q"]
    cnn_tasks = ['1.1']
    network_name_short = network_name[:-len("_silence_suppressed")] if network_name.endswith("silence_suppressed") else network_name

    assert ((network_name_short in pretraining_networks and task_name in pretraining_tasks)
            or (
                        network_name_short not in pretraining_networks and task_name not in pretraining_tasks)), \
        "Invalid network-task combination"

    if network_name_short in cnn_networks:
        return build_config_cnn(network_name, task_name, re_train, retrieve_config, special_names)
    seq_len = 100
    prep_phase = 50
    base_config = {"model_specs": {
        "model_name": None,  # to be added (CustomRNN or FC_RNN)
        "model_path": "Network_models.RNN_models",
        "model_params": {
            "input_size": 8,
            "hidden_size": 8, # to be changed
            "num_layers": 1, # to be changed
            "output_size": 3,
            "cell_type": None,  # to be added
        }
    }, "model_id": network_name, "device": "cuda", "scheduler": True, # Uses the ReduceLROnPlateau scheduler
        "optimizer_specs": {
        "optimizer_name": "Adam",
        "optimizer_params": {
            "lr": 0.01
        }
    }, "distance_loss": "geodesic_gradual", "gradual_loss_weighting": "constant+linear", "regularisation_loss": "L2",
        "distance_weight": 1, "output_regs_weight": 1, "silence_activity": False,
        # silence_activity is the knob for suppressing activity in the silence phase
        "training_config": {
            "task_id": "0.2q",
            "batch_size": 128,
            "mini_batch_size": 32,
            "seq_len": seq_len,
            "prep_phase": prep_phase,
        }, 'save_path': MODEL_SAVE_PATH + f"\\{network_name}_{task_name}_model", 'log_path': LOG_PATH,
        'check_path': MODEL_SAVE_PATH + f"\\{network_name}_model_checkpoints"}
    base_config['training_config']['data_save_path'] = DATA_PATH + f"\\Data_{task_name}.pth"

    if base_config['distance_loss'][-7:] == "gradual":
        base_config['save_path'] = MODEL_SAVE_PATH + f"\\{network_name}_{task_name}_model_gradual"
        base_config['check_path'] = MODEL_SAVE_PATH + f"\\{network_name}_model_checkpoints_gradual"
        base_config['log_path'] = LOG_PATH + f"\\{network_name}_{task_name}_model_gradual"
        base_config['model_specs']['model_params']['stepwise'] = True
    else:
        base_config['model_specs']['model_params']['stepwise'] = False

    if special_names is not None:
        base_config['save_path'] = base_config['save_path'] + f"_{special_names}"
        base_config['check_path'] = base_config['check_path'] + f"_{special_names}"
        base_config['log_path'] = base_config['log_path'] + f"_{special_names}"

    for path in [base_config['save_path'], base_config['log_path'], base_config['check_path']]:
        if not os.path.exists(path):
            os.makedirs(path)
        elif not re_train:
            print(f"Path already exists for {network_name} in task {task_name}. Assume the model is already trained.")
            if not retrieve_config:
                return None
        else: # re_train
            force_remove_dir(path)
            os.makedirs(path)

    # Switch on the silence_activity if network_name ends with "silence_suppressed"
    if network_name.endswith("silence_suppressed"):
        base_config["silence_activity"] = True

    network_name = network_name[:-len("_silence_suppressed")] if network_name.endswith("silence_suppressed") else network_name

    match network_name:
        case "GRU_1layer_8hidden":
            base_config["model_specs"]["model_name"] = "CustomRNN"
            base_config["model_specs"]["model_params"]["hidden_size"] = 8
            base_config["model_specs"]["model_params"]["num_layers"] = 1
            base_config["model_specs"]["model_params"]["cell_type"] = "GRU"
        case "RNN_1layer_8hidden":
            base_config["model_specs"]["model_name"] = "CustomRNN"
            base_config["model_specs"]["model_params"]["hidden_size"] = 8
            base_config["model_specs"]["model_params"]["num_layers"] = 1
            base_config["model_specs"]["model_params"]["cell_type"] = "RNN"
        case "GRU_1layer_16hidden":
            base_config["model_specs"]["model_name"] = "CustomRNN"
            base_config["model_specs"]["model_params"]["hidden_size"] = 16
            base_config["model_specs"]["model_params"]["num_layers"] = 1
            base_config["model_specs"]["model_params"]["cell_type"] = "GRU"
        case "RNN_1layer_16hidden":
            base_config["model_specs"]["model_name"] = "CustomRNN"
            base_config["model_specs"]["model_params"]["hidden_size"] = 16
            base_config["model_specs"]["model_params"]["num_layers"] = 1
            base_config["model_specs"]["model_params"]["cell_type"] = "RNN"
        case "GRU_1layer_32hidden":
            base_config["model_specs"]["model_name"] = "CustomRNN"
            base_config["model_specs"]["model_params"]["hidden_size"] = 32
            base_config["model_specs"]["model_params"]["num_layers"] = 1
            base_config["model_specs"]["model_params"]["cell_type"] = "GRU"
        case "RNN_1layer_32hidden":
            base_config["model_specs"]["model_name"] = "CustomRNN"
            base_config["model_specs"]["model_params"]["hidden_size"] = 32
            base_config["model_specs"]["model_params"]["num_layers"] = 1
            base_config["model_specs"]["model_params"]["cell_type"] = "RNN"
        case "GRU_2layer_8hidden":
            base_config["model_specs"]["model_name"] = "CustomRNN"
            base_config["model_specs"]["model_params"]["hidden_size"] = 8
            base_config["model_specs"]["model_params"]["num_layers"] = 2
            base_config["model_specs"]["model_params"]["cell_type"] = "GRU"
        case "RNN_2layer_8hidden":
            base_config["model_specs"]["model_name"] = "CustomRNN"
            base_config["model_specs"]["model_params"]["hidden_size"] = 8
            base_config["model_specs"]["model_params"]["num_layers"] = 2
            base_config["model_specs"]["model_params"]["cell_type"] = "RNN"
        case "FC_16_GRU_1layer_8hidden":
            base_config["model_specs"]["model_name"] = "FC_RNN"
            base_config["model_specs"]["model_params"]["hidden_size"] = 8
            base_config["model_specs"]["model_params"]["num_layers"] = 1
            base_config["model_specs"]["model_params"]["cell_type"] = "GRU"
            base_config["model_specs"]["model_params"]["FC_dim"] = 16
        case "FC_16_RNN_1layer_8hidden":
            base_config["model_specs"]["model_name"] = "FC_RNN"
            base_config["model_specs"]["model_params"]["hidden_size"] = 8
            base_config["model_specs"]["model_params"]["num_layers"] = 1
            base_config["model_specs"]["model_params"]["cell_type"] = "RNN"
            base_config["model_specs"]["model_params"]["FC_dim"] = 16
        case "FC_32_GRU_1layer_8hidden":
            base_config["model_specs"]["model_name"] = "FC_RNN"
            base_config["model_specs"]["model_params"]["hidden_size"] = 8
            base_config["model_specs"]["model_params"]["num_layers"] = 1
            base_config["model_specs"]["model_params"]["cell_type"] = "GRU"
            base_config["model_specs"]["model_params"]["FC_dim"] = 32
        case "FC_32_RNN_1layer_8hidden":
            base_config["model_specs"]["model_name"] = "FC_RNN"
            base_config["model_specs"]["model_params"]["hidden_size"] = 8
            base_config["model_specs"]["model_params"]["num_layers"] = 1
            base_config["model_specs"]["model_params"]["cell_type"] = "RNN"
            base_config["model_specs"]["model_params"]["FC_dim"] = 32
        case "FC_16_GRU_1layer_16hidden":
            base_config["model_specs"]["model_name"] = "FC_RNN"
            base_config["model_specs"]["model_params"]["hidden_size"] = 16
            base_config["model_specs"]["model_params"]["num_layers"] = 1
            base_config["model_specs"]["model_params"]["cell_type"] = "GRU"
            base_config["model_specs"]["model_params"]["FC_dim"] = 16
        case "FC_16_RNN_1layer_16hidden":
            base_config["model_specs"]["model_name"] = "FC_RNN"
            base_config["model_specs"]["model_params"]["hidden_size"] = 16
            base_config["model_specs"]["model_params"]["num_layers"] = 1
            base_config["model_specs"]["model_params"]["cell_type"] = "RNN"
            base_config["model_specs"]["model_params"]["FC_dim"] = 16
        case "FC_32_GRU_1layer_16hidden":
            base_config["model_specs"]["model_name"] = "FC_RNN"
            base_config["model_specs"]["model_params"]["hidden_size"] = 16
            base_config["model_specs"]["model_params"]["num_layers"] = 1
            base_config["model_specs"]["model_params"]["cell_type"] = "GRU"
            base_config["model_specs"]["model_params"]["FC_dim"] = 32
        case "FC_32_RNN_1layer_16hidden":
            base_config["model_specs"]["model_name"] = "FC_RNN"
            base_config["model_specs"]["model_params"]["hidden_size"] = 16
            base_config["model_specs"]["model_params"]["num_layers"] = 1
            base_config["model_specs"]["model_params"]["cell_type"] = "RNN"
            base_config["model_specs"]["model_params"]["FC_dim"] = 32
        case "FC_16_GRU_1layer_32hidden":
            base_config["model_specs"]["model_name"] = "FC_RNN"
            base_config["model_specs"]["model_params"]["hidden_size"] = 32
            base_config["model_specs"]["model_params"]["num_layers"] = 1
            base_config["model_specs"]["model_params"]["cell_type"] = "GRU"
            base_config["model_specs"]["model_params"]["FC_dim"] = 16
        case "FC_16_RNN_1layer_32hidden":
            base_config["model_specs"]["model_name"] = "FC_RNN"
            base_config["model_specs"]["model_params"]["hidden_size"] = 32
            base_config["model_specs"]["model_params"]["num_layers"] = 1
            base_config["model_specs"]["model_params"]["cell_type"] = "RNN"
            base_config["model_specs"]["model_params"]["FC_dim"] = 16
        case "FC_32_GRU_1layer_32hidden":
            base_config["model_specs"]["model_name"] = "FC_RNN"
            base_config["model_specs"]["model_params"]["hidden_size"] = 32
            base_config["model_specs"]["model_params"]["num_layers"] = 1
            base_config["model_specs"]["model_params"]["cell_type"] = "GRU"
            base_config["model_specs"]["model_params"]["FC_dim"] = 32
        case "FC_32_RNN_1layer_32hidden":
            base_config["model_specs"]["model_name"] = "FC_RNN"
            base_config["model_specs"]["model_params"]["hidden_size"] = 32
            base_config["model_specs"]["model_params"]["num_layers"] = 1
            base_config["model_specs"]["model_params"]["cell_type"] = "RNN"
            base_config["model_specs"]["model_params"]["FC_dim"] = 32
        case _:
            raise ValueError(f"Unknown network_name: {network_name}")

    if task_name == "0.1q":
        base_config["training_config"]["task_id"] = "0.1q"

    return base_config

def build_config_cnn(network_name, task_name, re_train, retrieve_config = False, special_names = None):
    seq_len = 100
    prep_phase = 50
    RNN_model_name = network_name[4:]
    base_config = build_config(RNN_model_name, "0.1q", re_train=False, retrieve_config=True, special_names=special_names)
    base_config['model_specs']['model_name'] = "Custom_CNN"
    base_config['model_specs']['model_path'] = "Network_models.CNN_models"
    base_config['model_specs']['model_params']['conv_output'] = 64
    base_config['model_specs']['model_params']['conv_layers'] = [
        {"type": "Conv2d",
            "in_channels": 1,
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 1},
        {"type": "ReLU"},
        {"type": "MaxPool2d",
            "kernel_size": 2,
            "stride": 2},
        {"type": "Conv2d",
            "in_channels": 16,
            "out_channels": 32,
            "kernel_size": 3,
            "stride": 1},
        {"type": "ReLU"},
        {"type": "MaxPool2d",
            "kernel_size": 2,
            "stride": 2},
        {"type": "Conv2d",
            "in_channels": 32,
            "out_channels": 64,
            "kernel_size": 3,
            "stride": 1},
        {"type": "ReLU"},
        {"type": "MaxPool2d",
            "kernel_size": 2,
            "stride": 2},
    ]


    base_config['training_config']["task_id"] = task_name
    base_config['training_config']['data_save_path'] = DATA_PATH + f"\\Data_{task_name}.pth"
    base_config['training_config']['object_path'] = OBJECT_PATH + "\\cow_mesh\\cow.obj"
    base_config['save_path'] = MODEL_SAVE_PATH + f"\\{network_name}_{task_name}_model"
    base_config['log_path'] = LOG_PATH + f"\\{network_name}_{task_name}_model"
    base_config['check_path'] = MODEL_SAVE_PATH + f"\\{network_name}_model_checkpoints"
    for path in [base_config['save_path'], base_config['log_path'], base_config['check_path']]:
        if not os.path.exists(path):
            os.makedirs(path)
        elif not re_train:
            print(f"Path already exists for {network_name} in task {task_name}. Assume the model is already trained.")
            if not retrieve_config:
                return None
        else: # re_train
            force_remove_dir(path)
            os.makedirs(path)
    return base_config


def build_config_Imported_Module(network_name, task_name, rnn_type, re_train, retrieve_config = False, special_names = None, **kwargs):
    """TO BE IMPLEMENTED. AT THIS POINT DO THIS MANULLY (SEE IMPORTED TRAINING NOTEBOOK)"""
    # The network_name is at this point going to be "ConvNeXt_Tiny_FC2__RNN"
    seq_len = 100
    prep_phase = 50
    RNN_model_name = network_name[4:]
    base_config = build_config(RNN_model_name, "0.1q", re_train=False, retrieve_config=True, special_names=special_names)
    base_config['model_specs']['model_name'] = "Custom_CNN"
    base_config['model_specs']['model_path'] = "Network_models.CNN_models"
    base_config['model_specs']['model_params']['conv_output'] = 64


    base_config['training_config']["task_id"] = task_name
    base_config['training_config']['data_save_path'] = DATA_PATH + f"\\Data_{task_name}.pth"
    base_config['training_config']['object_path'] = OBJECT_PATH + "\\cow_mesh\\cow.obj"
    base_config['save_path'] = MODEL_SAVE_PATH + f"\\{network_name}_{task_name}_model"
    base_config['log_path'] = LOG_PATH + f"\\{network_name}_{task_name}_model"
    base_config['check_path'] = MODEL_SAVE_PATH + f"\\{network_name}_model_checkpoints"
    for path in [base_config['save_path'], base_config['log_path'], base_config['check_path']]:
        if not os.path.exists(path):
            os.makedirs(path)
        elif not re_train:
            print(f"Path already exists for {network_name} in task {task_name}. Assume the model is already trained.")
            if not retrieve_config:
                return None
        else: # re_train
            force_remove_dir(path)
            os.makedirs(path)
    return base_config

