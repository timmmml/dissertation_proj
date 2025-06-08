"""

This is the raw draft of my analysis, copied from jupyter notebook lines. 

Choice: use a bounded ReLU for representation


"""

# import everything
%load_ext autoreload
%autoreload 2

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch.nn.functional as F
from torch.optim import lr_scheduler as LR
from torch.utils.data import Dataset, DataLoader
import plotly.express as px
from functools import partial
from SimulateDatasets.ContrastiveGenerator import ContrastiveGenerator, AR1_ordered_data_gen, OrderedDataset, SampledDataset
from Network_models.CANTrainer import CANTrainer
from Network_models.PartitionedRNN import PartitionedNet
from utils.path_settings import *
import seaborn as sns
import os
from Rotations import integrate_velocities, exp_quat, q_mult, integrate_quat_sequential
from Plots.AttractorPlots import *
from Plots.globe_plots import *
import utils
import Operator.Operator as o
import copy
from Network_models.CTrainer_Pred_CAN import CTrainer_Pred_CAN
from SimulateDatasets.ContrastiveGenerator import ControlGenerator
from Operator.miscellaneous import *
from Network_models.GRUController import GRUControllerAlltogether
from SimulateDatasets.ContrastiveGenerator import ControlGenerator_GRU
from Network_models.CTrainer_GRU import CTrainer_GRU

# set up rep net
# adjustable knobs 
knobs = {
        "batch_size" : 4096,
        "mini_batch_size" : 256,
        "repUnits" : 128,
        "model" : "PartitionedRNN", # or "PartitionedCTRNN"
        # "model": "FlexibleRNN",
        "manifold" : "t", #"custom", # or "S"
        # "manifold" : "custom",
        "dim" : 1,
        "zeroNullRep" : False,
        "zeroNullInput" : False,
        "biasRep" : True,
        "biasInput" : True,
        "hihr_identity": False, 
        "cParam" : True,
        # "nonlinearity" : "tanh",
        "nonlinearity": "BiologicalReLU",
        # "nonlinearity": "BiologicalTanh",
        # "nonlinearity": "relu",
        # "nonlinearity": "NormedReLU", 
        "inputEnveloping" : False,
        "loss" : "infonce",
        "triplet_weight" : 0, # only turn if loss is triplet
        "normalize" : False,
        "similarity_kernel" : None,
        "weight_decay_thresh" : -10,
        "L2norm_scalar": 0,
        "manifold_name": "ModifiedRingManifold",
        # "similarity_kernel": "cosine", 
        "similarity_kernel": "euclidean",
        "capacity": False, 
        "manifold_params": {
            "R":1, 
            "A":0.2, 
            "n":5,
        },
        "temperature": 0.7, 
        "invariance_weight": 1,  
        # "bidirectional": False, 
        "epsilon_max":0.1,
    }
for key, value in knobs.items():
    exec(f"{key} = {f"{value}" if type(value) != str else f'"{value}"'}")
if knobs['hihr_identity'] == False:
    knobs['hihr_identity'] = None
if knobs["L2norm_scalar"] == 0:
    knobs["L2norm_scalar"] = None
if knobs["similarity_kernel"] == "cosine":
    knobs["similarity_kernel"] = None
exclude = ["batch_size", "mini_batch_size", "triplet_weight", "loss", "weight_decay_thresh"]
if knobs["manifold"].lower() in ["s", "t"]: 
    manifold_specs = {
        "manifold_path": "SimulateDatasets.Manifolds", 
        "manifold_name": "Sphere" if knobs["manifold"].lower() == "s" else "Torus",
        "manifold_params": {"dim": knobs["dim"], 
                            }
    }
    exclude.extend(["manifold_name", "manifold_params"])


if knobs["manifold"].lower() == "custom": 
    manifold_specs = {
        "manifold_path": "SimulateDatasets.Manifolds",
        "manifold_name": knobs["manifold_name"],
        "manifold_params": knobs["manifold_params"]
    }
configs, data_specs = utils.knobs2specs_CAN(knobs)

data_gen = ContrastiveGenerator(data_specs)

# or alternatively, define trainer and load a model
trainer = CANTrainer(configs)
path = utils.load_from_lookup(configs)
try: 
    trainer.load_model(path)
    print("Model loaded via train")
except Exception as e:
    print(e)    
    print("No model to load")
    trainer.load_model(configs["save_path"][:-9] + "best_model.pth")

rnn = trainer.model.to("cuda")
rnn.eval()
rnn.rnn.eval()
torch.save(rnn, path.replace("model.pth", "model_full.pth"))

class CosineActivation(nn.Module):
    def __init__(self, theta):
        super(CosineActivation, self).__init__()
        """theta is the angle of the cosine activation, 
        if it is a list, then output an expansive feature map
        """
        self.theta = theta

    def forward(self, x):
        if isinstance(self.theta, list):
            return torch.cat([torch.cos(x * theta) for theta in self.theta], dim=1)
        return torch.cos(x * self.theta)
class CosineSineActivation(nn.Module):
    def __init__(self, theta):
        super(CosineSineActivation, self).__init__()
        """theta is the angle of the cosine activation, 
        if it is a list, then output an expansive feature map
        """
        self.theta = theta

    def forward(self, x):
        if isinstance(self.theta, list):
            return torch.cat([torch.cos(x * theta) for theta in self.theta] + [torch.sin(x * theta) for theta in self.theta], dim=1)
        return torch.cat([torch.cos(x * self.theta), torch.sin(x * self.theta)], dim=1)
    
def expand_trig(theta):
    """
    theta: torch.tensor(batch_size, dim): thetas of different dimensions
    """
    return torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)

def pos2theta(y): 
    dim = y.shape[-1]//2
    theta = torch.zeros((*y.shape[:-1], dim))
    for i in range(dim):
        theta[..., i] = torch.atan2(y[..., i + dim], y[..., i])
    return theta

d = data_gen.generate_data(10, disregard_n_samples=False)
dpositions = d.positions.cpu().numpy()

rep_only = False
mapnet = nn.Sequential(
    # CosineActivation(theta=np.pi/2),
    # ArcCosineActivation(theta=np.pi/2),
    CosineSineActivation(theta=[0.5, 1, 2]),
    nn.Linear(dpositions.shape[1] * 6, 128), 
    nn.ReLU(),
    # CosineSineActivation(theta=[0.5, 1, 2]),
    # nn.Linear(128, 128),
    # nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, rnn.rnn.rep_units if rep_only else rnn.rnn.hidden_size),
    # nn.Linear(dpositions.shape[1], rnn.rnn.rep_units if rep_only else rnn.rnn.hidden_size),
    rnn.rnn.nonlinearity_fn if rnn.rnn.nonlinearity_fn is not torch.nn.functional.tanh else nn.Tanh(), 
)
readoutnet = nn.Sequential(
    nn.Linear(rnn.rnn.rep_units, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, dpositions.shape[1]) if dim != 1 else nn.Linear(128, dpositions.shape[1]*2),
)

mapnet = torch.load(path.replace("model.pth", "mapnet.pth"))
readoutnet = torch.load(path.replace("model.pth", "readoutnet.pth"))

plt.rcParams['figure.titlesize'] = 12
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['legend.handlelength'] = 1.5
plt.rcParams['legend.handleheight'] = 0.5
plt.rcParams['axes.spines.right'] = False
plt.rcParams[('axes.spines.top')] = False


# set seeds
torch.manual_seed(0)
np.random.seed(0)

null_action = torch.ones(1, 16, dim).cuda() * 0.000
# null_action = torch.randn(1, 16, dim).cuda()
rnn.eval()
with torch.no_grad():
    final = rnn(null_action, truncate_output=False)
readoutnet = readoutnet.float()
y = readoutnet(final[:, :, -128:].float())


# NOTE THAT THIS INITIAL POSITION IS REQUIRED FOR LATER TRAINING

# SETUP CODE FOR THE MODULAR CONTROLLER

readoutnet2 =copy.deepcopy(readoutnet)
from SimulateDatasets.ContrastiveGenerator import ControlGenerator
batch_size = 2048
mini_batch_size = 256
model_specs = {
    "model_name": "PredControlNetCAN_2",
    "model_path": "Network_models.PIDController",
    "model_params":{
        "rep_net": rnn,
        "plant": copy.deepcopy(plant),
        "operator": copy.deepcopy(fly),
        "IM1": copy.deepcopy(im1),
        # "IM2": SingleLayerLM(joint.position_size + rnn.rnn.hidden_size, action_size, 0, "tanh").cuda(),
        # "predictor": copy.deepcopy(predictor),
        "predictor": readoutnet2,
        "readoutnet": readoutnet,
        "output_size": output_size,
        "rep_units_only": True, 
        "recurrent_control": True, 
        # "activation": "Tanh", 
        "activation": "ReLU", 
        # "recurrent_control_dim": 128, 
        "hidden_control_dim": 128,
        "recurrent_control_dim": 32,
        # "hidden_control_dim": 0,
        "device": "cuda",
        "use_rep": use_rep, 
        "use_feedback": use_feedback,
    }
}

optimizer_specs = {
    "compartmental_optimization": True,
    "optimizer_name": "Adam",
    "optimizer_params": {
        "lr": 0.003,
    }
}

trainer_config = {
    "model_specs": model_specs,
    "optimizer_specs": optimizer_specs,
    "control_period": 20, 
    "random_initial_condition": True,
    "additional_loss": ["final_distance", "control_norm"]#, "state_norm"], 
    # "additional_loss": ["control_norm"], # choices: "state_norm", "control_norm" 
}

model_path = "/LAL" + utils.gen_model_path_Pred_Controller()

trainer_config = trainer_config | {
    "save_path": MODEL_SAVE_PATH + model_path + "/",
    "check_path": MODEL_SAVE_PATH + model_path + "/",
    "log_path": LOG_PATH + model_path + "/",
}

data_gen_config = {
    "mini_batch_size": mini_batch_size,
    "contrastive_generator": data_gen, 
    "rep_only": True, 
    "to_null": False, 
    "null_rep": final[0, -1, :],
}

import time
print(f"batch_size: {batch_size}, mini_batch_size: {mini_batch_size}"), "state_norm"
training_args = {
    "data_gen": ControlGenerator(data_gen_config),
    "batch_size": batch_size, 
    "mini_batch_size": mini_batch_size, 
    "epochs_to_calculate": 5,
    "epochs": 300, 
    "load_upper": 47,
    "save_interval": 10, 
    "reset_interval": 200,
    "early_stop": False,
    "verbose": True,
    "replacement_tolerance": 0.1,
}

trainer = CTrainer_Pred_CAN(trainer_config)
time_start = time.time()

for path in [trainer_config["save_path"], trainer_config["check_path"], trainer_config["log_path"]]:
    os.makedirs(path, exist_ok = True)
training_args['save_path'] = trainer_config["save_path"] + "model.pth"
print(f"tensorboard --logdir {trainer_config['log_path']}")


# TRAINING PHASE ONE: TRAIN THE INTERNAL MODEL

# trainer = CTrainer_Pred_CAN(trainer_config)
# torch.autograd.detect_anomaly(True)
trainer._optimizer_type(trainer.optimizer_specs)
trainer.model.operator.feedback_noise = 0.5
trainer.scheduler=None
trainer.model.float()
trainer.model.freeze_controller()
trainer.model.freeze_rep_net()
trainer.model.freeze_readoutnet()
trainer.model.unfreeze_pred()
trainer.model.unfreeze_im1()
# trainer.model.freeze_im1()
trainer.model.freeze_plant()
trainer.model.freeze_operator()
trainer.model.freeze_pred()

trainer.model.efference_only()
trainer.model.full_feedback(freeze_efference=False)
trainer.model.im1.silence_efference2 = True
trainer.model.disactivate_controller(scale =0.5)
trainer.model.introduce_disturbance(scale=5)
# trainer.model.activate_controller()
# trainer.model.disactivate_im1()
# trainer.model.remove_disturbance()

trainer.train(**training_args, action_recovery_weight=1, s0_and_observe = False)


# TODO: add a checkpoint here and train controllers of different calibres. 

# TRAINING PHASE TWO: TRAIN THE CONTROLLER. 


trainer.control_period = 20
trainer.lr_init = 0.0001

trainer.model.remove_disturbance()
trainer.model.introduce_disturbance(scale=3)
trainer.model.activate_controller()
trainer.model.unfreeze_controller()
trainer.model.unfreeze_im1()
trainer.model.unfreeze_pred()

trainer.train(**training_args, convergence_loss_weight = 0.3, loss_traj_weight = 0.5, action_weight = 0.5, action_recovery_weight = 1, control_norm_weight = 0.1, s0_and_observe = False, weight_decay = 0.0001)
