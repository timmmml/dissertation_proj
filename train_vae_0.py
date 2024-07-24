import torch
try:
    from utils import goto_project_root
except ModuleNotFoundError:
    import os
    os.environ["PATH"] += os.pathsep + 'C:\\Users\\timmy\\Documents\\Projects_dir\\Mental_Rotations\\mental-rotations'
    print(os.environ["PATH"])
    from utils import goto_project_root
from utils.path_settings import MODEL_SAVE_PATH, DATA_PATH, LOG_PATH, CONFIG_PATH, OBJECT_PATH
from torch.utils.tensorboard import SummaryWriter
import SimulateDatasets.GenTrainingData as g
from utils import create_splits, get_dataloaders, force_remove_dir
from Network_models import VaeTrainer as t
from importlib import reload
import os
reload(t)
reload(g)
import time
import numpy as np
from pytorch3d.transforms import so3_relative_angle
from pytorch3d.transforms import quaternion_to_matrix
import json
import Train_task.train_from_config_names as train
reload(train)
import torchvision.models as models
#%%
obj_path_dict = {
    "cow": OBJECT_PATH + "\\cow_mesh\\cow.obj",
    "A1": OBJECT_PATH + "\\stim_simple\\A1_5x6x5_centered.obj",
    "A2": OBJECT_PATH + "\\stim_simple\\A2_5x6x5_centered.obj",
    "B1": OBJECT_PATH + "\\stim_simple\\B1_6x6x6_centered.obj",
    "B2": OBJECT_PATH + "\\stim_simple\\B2_6x6x6_centered.obj",
    "C1": OBJECT_PATH + "\\stim_simple\\C1_5x6x5_centered.obj",
    "C2": OBJECT_PATH + "\\stim_simple\\C2_5x6x5_centered.obj",
    "D1": OBJECT_PATH + "\\stim_simple\\D1_5x6x5_centered.obj",
    "D2": OBJECT_PATH + "\\stim_simple\\D2_5x6x5_centered.obj",
    "E1": OBJECT_PATH + "\\stim_simple\\E1_5x5x6_centered.obj",
    "E2": OBJECT_PATH + "\\stim_simple\\E2_5x5x6_centered.obj",
}
obj_index_dict = {
    "cow": None,
    "A1": "5x6x5_centered",
    # "A2": "5x6x5_centered",
    # "B1": "6x6x6_centered",
    # "B2": "6x6x6_centered",
    # "C1": "5x6x5_centered",
    # "C2": "5x6x5_centered",
    # "D1": "5x6x5_centered",
    # "D2": "5x6x5_centered",
    # "E1": "5x5x6_centered",
    # "E2": "5x5x6_centered",
}
cam_position = {
    "cow": 3,
    "A1": 8,
    "A2": 8,
    "B1": 8,
    "B2": 8,
    "C1": 8,
    "C2": 8,
    "D1": 8,
    "D2": 8,
    "E1": 8,
    "E2": 8,
}
#%%
obj_path_dict.get("cow", None)
#%%
obj_name = "cow"
obj_index = obj_index_dict.get(obj_name, None)
obj_path = obj_path_dict.get(obj_name, None)
cam_pos = cam_position.get(obj_name, None)
obj_name = obj_name + "_" + obj_index + ".obj" if obj_index is not None else obj_name + ".obj"

network_name = "simple_VAE"
task_name = "1.1"
re_train = 1
resolution_in = 64
resolution_out = resolution_in
latent_dimension = 32


config = {"model_specs": {
    "model_name": "RotationVAE",  # to be added (CustomRNN or FC_RNN)
    "model_path": "Network_models.RotationVAE",
    "model_params": {
        "input_resolution": resolution_in,
        "output_resolution": resolution_out,
        "latent_dimension": latent_dimension,
        "encoder": "simple",
        "decoder": "simple",
        },
}, "model_id": network_name, "device": "cuda", "scheduler": True, # Uses the ReduceLROnPlateau scheduler
    "optimizer_specs": {
    "optimizer_name": "Adam",
    "optimizer_params": {
        "lr": 0.01
    }
}, "reconstruction_loss": "MSE", "kl_divergence_loss": "Normal", "annealing": True, "annealing_function": "constant",
    "training_config": {
        "task_id": "1.1",
        "batch_size": 1024,
        "mini_batch_size": 128,
        "resolution": resolution_in,
        "cam_position": 2.1,#cam_pos,
        "object_path": obj_path,
        "use_AR": True,
    }, 'save_path': MODEL_SAVE_PATH + f"\\{network_name}_latent{latent_dimension}_res_in{resolution_in}_res_out"
                                        f"{resolution_out}_model",
    'log_path': LOG_PATH + f"\\{network_name}_latent{latent_dimension}_res_in{resolution_in}_res_out{resolution_out}",
    'check_path': MODEL_SAVE_PATH + f"\\{network_name}_latent{latent_dimension}_res_in{resolution_in}_res_out"
                                        f"{resolution_out}_model_checkpoints",
}
config['training_config']['data_save_path'] = (DATA_PATH +
                                               f"\\Data_{task_name}_res{config['training_config']['resolution']}_{obj_name[:-4]}_cam{config['training_config']['cam_position']}"
                                               f".pth")

for path in [config['save_path'], config['log_path'], config['check_path']]:
    if not os.path.exists(path):
        os.makedirs(path)
    elif not re_train:
        print(f"Path already exists for {network_name} in task {task_name}. Assume the model is already trained.")
    else: # re_train
        force_remove_dir(path)
        os.makedirs(path)
print(config['save_path'])
json.dump(config, open(CONFIG_PATH + f"\\{network_name}_{task_name}_res{config['training_config']['resolution']}_configs"
                                     f".json", "w"))
#%%
experimental_trainer = t.VaeTrainer(config)
experimental_trainer.model.resolution_in = config['training_config']['resolution']
data_path = config['training_config']['data_save_path'][:-4] if config['training_config']['data_save_path'][-4:] == ".pth"\
    else config['training_config']['data_save_path']
if os.path.exists(data_path + "_train.pth"):
    data= torch.load(data_path + "_train.pth")
else:
    d = g.gen_training_data(config['training_config'])
    data = torch.load(data_path + "_train.pth")
dataloaders = get_dataloaders(data, batch_size = config['training_config']['mini_batch_size'], k = 5)
epochs = 100
print(f"Training {network_name} on task {task_name}, saving to {config['save_path']}; copy the below for logs")
print(f"tensorboard --logdir={config['log_path']}")
for i in range(len(dataloaders)):
    sub_log_path = config["log_path"] + f"\\split_{i + 1}"
    sub_check_path = config["check_path"] + f"\\split_{i + 1}"

    force_remove_dir(sub_log_path) # Probably redundant but helps to ensure no old logs are kept
    os.makedirs(sub_log_path, exist_ok = True)
    os.makedirs(sub_check_path, exist_ok = True)

# keep a record on the best split:
best_split_val_loss = torch.tensor(float('inf'))
for i, (train_loader, val_loader) in enumerate(dataloaders):
    print(f"Training split {i + 1}")
    experimental_trainer.refresh()
    sub_log_path = config["log_path"] + f"\\split_{i + 1}"
    sub_check_path = config["check_path"] + f"\\split_{i + 1}"

    experimental_trainer.train(train_loader,
                      val_loader,
                      epochs=epochs,
                      save_path = config["save_path"] + f"\\model_{i+1}.pth",
                      check_path= sub_check_path,
                      log_path = sub_log_path)
    if experimental_trainer.best_val_loss < best_split_val_loss:
        best_split_val_loss = experimental_trainer.best_val_loss
        best_split = i + 1

    experimental_trainer.save_model(config["save_path"] + f"\\model_{i+1}.pth", full=1)
experimental_trainer.load_best_model()
print(f"Training {network_name} on task {task_name} complete. Saving best model to {config['save_path']}.")
experimental_trainer.save_model(config["save_path"] + f"\\best_model.pth", full=1)