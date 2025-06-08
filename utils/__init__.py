from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import torch
import os
import shutil
from .path_settings import *
import json
from Network_models import Trainer as t
import datetime
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

 


def create_splits(data, n_splits=5):
    if n_splits == 1:
        # regular train-test split
        train_size = int(0.8 * len(data))
        test_size = len(data) - train_size
        train_set, test_set = torch.utils.data.random_split(data, [train_size, test_size])
        return [(train_set, test_set)]
    kf = KFold(n_splits=n_splits, shuffle=True)
    splits = []
    for train_idx, test_idx in kf.split(data):
        train_set = torch.utils.data.Subset(data, train_idx)
        test_set = torch.utils.data.Subset(data, test_idx)
        splits.append((train_set, test_set))
    return splits


def get_dataloaders(dataset, batch_size=32, k=5):
    splits = create_splits(dataset, k)
    dataloaders = []

    for train_set, test_set in splits:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        dataloaders.append((train_loader, test_loader))

    return dataloaders

def force_remove_dir(dir_path):
    # Check if the directory exists
    if os.path.exists(dir_path):
        try:
            # Use shutil.rmtree to forcefully remove the directory and all its contents
            shutil.rmtree(dir_path)
            print(f"Successfully removed directory: {dir_path}")
        except Exception as e:
            print(f"Error while removing directory: {dir_path}")
            print(f"Exception: {e}")
    else:
        print(f"Directory does not exist: {dir_path}")

def load_trainer_model(model_name, special_name):
    """This module returns a trainer with the appropriate model loaded if exists"""
    try:
        configs = json.load(open(CONFIG_PATH + f"\\{model_name}.json"))
        if not hasattr(configs['training_config'], "resolution"):
            configs['training_config']["resolution"] =None
        trainer = t.Trainer(configs)
        if special_name is not None:
            trainer.load_model(configs['save_path'] + "_" + special_name + "\\best_model.pth", full = 1)
        else:
            trainer.load_model(configs['save_path'] + "\\best_model.pth", full = 1)
        return trainer
    except Exception as e:
        print(f"Error while loading model: {model_name}")
        print(f"Exception: {e}")
        return None


def knobs2specs_CAN(knobs): 
    if not knobs.get('hihr_identity', True):
        knobs['hihr_identity'] = None
    if knobs.get("L2norm_scalar", 1) == 0:
        knobs["L2norm_scalar"] = None
    if knobs.get("similarity_kernel", "") == "cosine":
        knobs["similarity_kernel"] = None
    exclude = ["batch_size", "mini_batch_size", "triplet_weight", "loss", "weight_decay_thresh"]
    if knobs.get("manifold", "").lower() in ["s", "t"]: 
        manifold_specs = {
            "manifold_path": "SimulateDatasets.Manifolds", 
            "manifold_name": "Sphere" if knobs["manifold"].lower() == "s" else "Torus",
            "manifold_params": {"dim": knobs.get("dim", None)}
        }
        exclude.extend(["manifold_name", "manifold_params"])

    if knobs.get("manifold", "").lower() == "custom": 
        manifold_specs = {
            "manifold_path": "SimulateDatasets.Manifolds",
            "manifold_name": knobs.get("manifold_name", ""),
            "manifold_params": knobs.get("manifold_params", {})
        }
        exclude.extend(["manifold", "manifold_params", "dim"])
    knobs["triplet_weight"] = None if knobs.get("loss", "") != "combined" else knobs.get("triplet_weight", None)


    model_path = gen_model_path_CAN()
    training_args = {
        "batch_size": knobs.get("batch_size", 32), 
        "mini_batch_size": knobs.get("mini_batch_size", 32), 
        "epochs_to_calculate": 10,
        "epochs": 100, 
        "save_interval": 10, 
        "reset_interval": 200,
        "early_stop": False,
        "verbose": True,
        "replacement_tolerance": 0.6,
    }

    model_specs = {
        "model_name": "PartitionedNet" if knobs.get("model", "PartitionedRNN") == "PartitionedRNN" else "FlexibleNet",
        "model_path": "Network_models.PartitionedRNN",
        "model_params": {
            "network_type": knobs.get("model"),
            "input_size": knobs.get("dim") * (1 + knobs.get("bidirectional", False) * 1),
            "rep_units": knobs.get("repUnits"),
            "input_units": knobs.get("repUnits"), 
            "nonlinearity": knobs.get("nonlinearity"), 
            "bias_input": knobs.get("biasInput"),
            "bias_rep": knobs.get("biasRep"),
            "zero_null_input": knobs.get("zeroNullInput"),
            "zero_null_rep": knobs.get("zeroNullRep"),
            "c_param": knobs.get("cParam"),
            "input_connected": True, 
            "input_N": 1, 
            "freeze_tau": True, 
            "hihr_identity": knobs.get("hihr_identity"),
        }
    }

    data_specs = {
        "mini_batch_size": knobs.get("mini_batch_size"),

        "mode": "ordered_AR", 
        "generator_params": {
            "n_total_trial": 100 if knobs.get("dim") == 2 else (1000 if knobs.get("dim") == 1 else 20),
            "dim": knobs.get("dim"),
            "bidirectional": knobs.get("bidirectional"),
            "max_time_steps": 10,
            "min_length": 8,
            "noise_std": 0.2, 
            "ar_coef": 0.8,
            "baseline_offset_p": 0.01, 
            "manifold_specs": manifold_specs,
            "percentile": knobs.get("percentile", 1), 
            "epsilon_max": knobs.get("epsilon_max", torch.inf),
            "input_enveloping": knobs.get("inputEnveloping"),
        }
    }

    optimizer_specs = {
        "optimizer_name": "Adam",
        "optimizer_params": {"lr": 0.003, }
    }

    loss_config = {
        "loss_type": knobs.get("loss"), 
        "triplet_weight": knobs.get("triplet_weight"),
        "temperature": knobs.get("temperature", 0.32),
        "invariance_weight": knobs.get("invariance_weight", 1), 
        "normalize": knobs.get("normalize"),
        "weight_decay_scalar": 0.01, 
        "L2norm_scalar": knobs.get("L2norm_scalar"),
        "weight_decay_thresh": knobs.get("weight_decay_thresh"),
        "random_offset": True, 
        "similarity_kernel": knobs.get("similarity_kernel"),
        "capacity_weight" : 0.5 if knobs.get("capacity") else 0,
        "conIso_weight": 0
    }

    configs = {
        "model_specs": model_specs,
        "optimizer_specs": optimizer_specs,
        "loss_config": loss_config, 
        "training_config": training_args,
        "manifold_specs": manifold_specs,
        "device": "cuda",
        "save_path": MODEL_SAVE_PATH + model_path + "/", 
        "check_path": MODEL_SAVE_PATH + model_path + "/",
        "log_path": LOG_PATH + model_path + "/",
    }

    # make the directories in case they dont exist
    os.makedirs(configs["save_path"], exist_ok=True)
    os.makedirs(configs["check_path"], exist_ok=True)
    os.makedirs(configs["log_path"], exist_ok=True)

    configs['save_path'] = configs['save_path'] + "model.pth"

    config_without_date = configs.copy()
    config_without_date.pop("save_path")
    config_without_date.pop("check_path")
    config_without_date.pop("log_path")
    config_without_date.pop("device")
    config_without_date.pop("training_config")
    config_without_date.pop("optimizer_specs")
    # save the specs into json file in the save path
    with open(configs['save_path'][:-4] + ".json", "w") as f:
        json.dump(config_without_date, f, indent=4)

    return configs, data_specs

def gen_model_path_CAN():
    # just name by today's date
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_path = "/CAN/" + date 
    return model_path

def gen_model_path_Pred_Controller():
    # just name by today's date
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_path = "/Pred_Controller/" + date 
    return model_path

def gen_model_path_GRU_Controller():
    # just name by today's date
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_path = "/GRU_Controller/" + date 
    return model_path

def update_lookup(configs, lookup_path = "/CAN_lookup.json"): 
    # update the lookup table with the new model: 
    barcode = gen_config_barcode(configs)
    if os.path.exists(MODEL_SAVE_PATH + lookup_path):
        lookup = json.load(open(MODEL_SAVE_PATH + lookup_path))
    else: 
        lookup = {}
    lookup[barcode] = configs['save_path']
    with open(MODEL_SAVE_PATH + lookup_path, "w") as f: 
        json.dump(lookup, f, indent=4)
    print(f"Successfully updated lookup table with barcode: {barcode}")
    print(f"Model saved at: {configs['save_path']}")


def gen_config_barcode(configs): 
    # generate a barcode for the config
    config_without_date = configs.copy()
    config_without_date.pop("save_path")
    config_without_date.pop("check_path")
    config_without_date.pop("log_path")
    config_without_date.pop("device")
    config_without_date.pop("training_config")
    config_without_date.pop("optimizer_specs")

    # Serialize the config to a JSON string, ensuring keys are sorted
    config_str = json.dumps(config_without_date, sort_keys=True)
    # Hash the string using SHA256
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()
    return config_hash

def load_from_lookup(configs, lookup_path = "/CAN_lookup.json"): 
    # load the lookup table
    barcode = gen_config_barcode(configs)
    if os.path.exists(MODEL_SAVE_PATH + lookup_path):
        lookup = json.load(open(MODEL_SAVE_PATH + lookup_path))
    else: 
        lookup = {}
    return lookup.get(barcode, None)


def plot_joint_tuning(population, subsets, theta1, theta2, subsample_index, controller_signs, controller_mags = None, squared = False, scaled = True, save = False, show = True): 
    """
    Plot the joint tuning curves for a given population, given mGPLVM fit 

    Args:
        population (str): The population to plot.
        W_control (torch.Tensor): the weights of the controller

    """
    theta = (theta1 + np.pi) % (2 * np.pi) - np.pi
    theta = theta[:-1]

    controller_means = subsets[population]

    if isinstance(controller_signs, torch.Tensor):
        controller_signs = controller_signs.numpy()
    
    if isinstance(controller_mags, torch.Tensor):
        controller_mags = controller_mags.numpy()
    
    if isinstance(subsample_index, torch.Tensor):
        subsample_index = np.array(subsample_index.numpy(), dtype=int)
    
    if controller_mags is None:
        if scaled:
            print("Controller magnitudes not provided, using ones")
        controller_mags = np.ones(controller_signs.shape)

    indices = np.linspace(0, 1, 50)
    colors = plt.cm.twilight_shifted(indices)
    # colors = plt.cm.hsv(indices)
    # Separate indices based on controller signs
    positive_sign_indices = np.where(controller_signs > 0)[0]
    positive_sign_indices_ = []
    for i in range(len(positive_sign_indices)):
        if np.where(subsample_index == positive_sign_indices[i])[0].shape[0] == 0:
            continue
        positive_sign_indices_.append(np.where(subsample_index == positive_sign_indices[i])[0][0])
    positive_sign_indices = np.array(positive_sign_indices_)

    negative_sign_indices = np.where(controller_signs <= 0)[0]
    negative_sign_indices_ = []
    for i in range(len(negative_sign_indices)):
        if np.where(subsample_index == negative_sign_indices[i])[0].shape[0] == 0:
            continue
        negative_sign_indices_.append(np.where(subsample_index == negative_sign_indices[i])[0][0])
    negative_sign_indices = np.array(negative_sign_indices_)

    fig, axes = plt.subplots(2, 8, figsize=(8, 4), sharey=True)
    axes = axes.flatten()


    # Plot for positive sign controllers
    for i, idx in enumerate(positive_sign_indices):
        
        if idx >= controller_means.shape[-1]:
            continue
        trace = controller_means[..., idx] * (controller_mags[idx] if scaled else 1)
        shift = np.argmax(trace.mean(axis=1))  # Find the peak to align
        angle_pref = theta[shift]
        relative_goal = (theta2 + theta1 - angle_pref + np.pi) % (2 * np.pi) - np.pi
        shifted_trace = np.roll(trace, -25-shift, axis=0).clip(min = 0)
        shifted_theta = np.roll(theta, -25, axis=0)

        sorted_indices = np.argsort(relative_goal)
        relative_goal = relative_goal[sorted_indices]
        shifted_trace = shifted_trace[:, sorted_indices]
        for j, heading in enumerate(np.linspace(1, 49, 8)):
            plotted = shifted_trace[:-1, round(heading)] ** (2 if squared else 1)
            axes[j].plot(shifted_theta, plotted, color=colors[round(heading)], alpha=0.5)
            axes[j].set_title(f"Goal: {relative_goal[round(heading)]:.2f}")
            axes[j].set_xlabel(r"$\theta_t$")
            if j == 0:
                axes[j].set_ylabel("Rectified Activity")
            else:
                axes[j].tick_params(labelleft=False)

    # Plot for negative sign controllers
    for i, idx in enumerate(negative_sign_indices):

        # print(idx, subsample_index[idx])
        trace = controller_means[..., idx]  * (controller_mags[idx] if scaled else 1)
        if idx >= controller_means.shape[-1]:
            continue

        shift = np.argmax(trace.mean(axis=1))  # Find the peak to align
        # print(shift)
        angle_pref = theta[shift]
        # print(angle_pref)
        shifted_trace = np.roll(trace[:-1, ...], -25-shift, axis=0).clip(min=0)
        shifted_theta = np.roll(theta, -25, axis=0)
        # shifted_trace = trace.clip(min =0)
        # shifted_theta = theta
        relative_goal = (theta2 + theta1 - angle_pref + np.pi) % (2 * np.pi) - np.pi
        # sort the indices such that the relative goal is low to high
        sorted_indices = np.argsort(relative_goal)
        relative_goal = relative_goal[sorted_indices]
        shifted_trace = shifted_trace[:, sorted_indices]
        for j, heading in enumerate(np.linspace(1, 49, 8)):
            # print(round(heading))
            plotted = shifted_trace[:, round(heading)] ** (2 if squared else 1)
            axes[j+ 8].plot(shifted_theta, plotted, color=colors[round(heading)], alpha=0.5)
            axes[j+ 8].set_title(f"Goal: {relative_goal[round(heading)]:.2f}")
            axes[j+ 8].set_xlabel(r"$\theta_t$")
            if j == 0:
                axes[j+ 8].set_ylabel("Rectified Activity")
            else:
                axes[j + 8].tick_params(labelleft=False)

    for i in range(16):
        axes[i].set_xlim(-np.pi, np.pi)
        axes[i].set_xticks([-np.pi, 0, np.pi])
        axes[i].set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
        axes[i].grid(True, linestyle="--", alpha=0.6)
    plt.suptitle(f"{population} Population Tuning Curves with Signs, Preferred-relative Goals")
    plt.tight_layout()
    if save: 
        plt.savefig(f"{FIG_PATH}/{population}_tuning_curves_relative_goal.png")
    if show: 
        plt.show()

    fig, axes = plt.subplots(2, 8, figsize=(8, 4), sharey=True)
    axes = axes.flatten()

    # Plot for positive sign controllers
    for i, idx in enumerate(positive_sign_indices):
        if idx >= controller_means.shape[-1]:
            continue
        trace = controller_means[..., idx] * (controller_mags[idx] if scaled else 1)

        shift = np.argmax(trace.mean(axis=1))  # Find the peak to align
        angle_pref = theta[shift]
        relative_goal = ((theta2  + np.pi) % (2 * np.pi) - np.pi)[1:-1]
        shifted_trace = np.roll(trace, -25-shift, axis=0).clip(min = 0)
        shifted_theta = np.roll(theta, -25, axis=0)
        for j, heading in enumerate(np.linspace(0, 47, 8)):
            plotted = shifted_trace[:-1, round(heading)] ** (2 if squared else 1)
            axes[j].plot(shifted_theta, plotted, color=colors[round(heading)], alpha=0.5)
            axes[j].set_title(f"Residual: {relative_goal[round(heading)]:.2f}")
            axes[j].set_xlabel(r"$\theta_t$")
            if j == 0:
                axes[j].set_ylabel("Rectified Activity")
            else:
                axes[j].tick_params(labelleft=False)

    # Plot for negative sign controllers
    for i, idx in enumerate(negative_sign_indices):
        if idx >= controller_means.shape[-1]:
            continue
        trace = controller_means[..., idx] * (controller_mags[idx] if scaled else 1)

        shift = np.argmax(trace.mean(axis=1))  # Find the peak to align
        angle_pref = theta[shift]
        shifted_trace = np.roll(trace, -25-shift, axis=0).clip(min=0)
        shifted_theta = np.roll(theta, -25, axis=0)
        relative_goal = ((theta2 + np.pi) % (2 * np.pi) - np.pi)[1:-1]
        # sort the indices such that the relative goal is low to high
        for j, heading in enumerate(np.linspace(0, 47, 8)):
            plotted = shifted_trace[:-1, round(heading)] ** (2 if squared else 1)
            axes[j+ 8].plot(shifted_theta, plotted, color=colors[round(heading)], alpha=0.5)
            axes[j+ 8].set_title(f"Residual: {relative_goal[round(heading)]:.2f}")
            axes[j+ 8].set_xlabel(r"$\theta_t$")
            if j == 0:
                axes[j+ 8].set_ylabel("Rectified Activity")
            else:
                axes[j + 8].tick_params(labelleft=False)

    for i in range(16):
        axes[i].set_xlim(-np.pi, np.pi)
        axes[i].set_xticks([-np.pi, 0, np.pi])
        axes[i].set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
        axes[i].grid(True, linestyle="--", alpha=0.6)
    plt.suptitle(f"{population} Population Tuning Curves with Signs, Residuals")
    plt.tight_layout()
    if save: 
        plt.savefig(f"{FIG_PATH}/{population}_tuning_curves_residuals.png")
    if show: 
        plt.show()



def plot_joint_tuning_binned(population,
                             activations,
                             thetas,
                             controller_signs,
                             controller_mags=None,
                             subsample_index = None, 
                             n_bins=8,
                             n_theta_bins=50,
                             squared=False,
                             scaled=True,
                             save=False,
                             show=True, 
                             rectify = True
                             ):
    """
    activations:       array or tensor of shape (B, N)
    thetas:            array or tensor of shape (B, 2)  [theta1, theta2]
    controller_signs:  array or tensor of length N
    controller_mags:   array or tensor of length N (optional)
    """
    # -- convert tensors to numpy
    if isinstance(activations, torch.Tensor):
        activations = activations.detach().cpu().numpy()
    if isinstance(thetas, torch.Tensor):
        thetas = thetas.detach().cpu().numpy()
    if isinstance(controller_signs, torch.Tensor):
        controller_signs = controller_signs.detach().cpu().numpy()
    if isinstance(subsample_index, torch.Tensor):
        subsample_index = np.array(subsample_index.numpy(), dtype=int)
    if controller_mags is None:
        if scaled:
            print("controller_mags not provided, defaulting to ones")
        controller_mags = np.ones_like(controller_signs)
    elif isinstance(controller_mags, torch.Tensor):
        controller_mags = controller_mags.detach().cpu().numpy()

    B, N = activations.shape
    # unwrap thetas and wrap to [-pi,pi)
    theta1 = (thetas[:,0] + np.pi) % (2*np.pi) - np.pi
    theta2 = (thetas[:,1] + np.pi) % (2*np.pi) - np.pi

    # rectify and optional square then scale per neuron
    if rectify: 
        act = np.clip(activations, 0, None)
    else:
        act = activations

    if squared:
        act = act**2
    if scaled:
        act = act * controller_mags[None, :]

    # define self-angle bins
    theta_edges   = np.linspace(-np.pi, np.pi, n_theta_bins+1)
    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2

    # compute each neuron's preferred theta1
    preferred = np.zeros(N)
    preferred_index = np.zeros(N, dtype=int)
    for n in range(N):
        means = []
        for m in range(n_theta_bins):
            mask_m = (theta1 >= theta_edges[m]) & (theta1 < theta_edges[m+1])
            means.append(act[mask_m, n].mean() if np.any(mask_m) else 0)
        preferred_index[n] = np.argmax(means)
        preferred[n] = theta_centers[np.argmax(means)]

    # compute condition values
    rel_goal = ((theta2[:,None] + theta1[:,None] - preferred[None,:]) + np.pi) % (2*np.pi) - np.pi
    residual = (theta2 + np.pi) % (2*np.pi) - np.pi

    # bin edges for conditions
    cond_edges   = np.linspace(-np.pi, np.pi, n_bins+1)
    cond_centers = (cond_edges[:-1] + cond_edges[1:]) / 2

    # split neurons by sign
    pos_idx = np.where(controller_signs > 0)[0]
    neg_idx = np.where(controller_signs <= 0)[0]

    if subsample_index is not None:
        pos_idx_ = []
        for i in range(len(pos_idx)):
            if np.where(subsample_index == pos_idx[i])[0].shape[0] == 0:
                continue
            pos_idx_.append(np.where(subsample_index == pos_idx[i])[0][0])
        pos_idx = np.array(pos_idx_)
        neg_idx_ = []
        for i in range(len(neg_idx)):
            if np.where(subsample_index == neg_idx[i])[0].shape[0] == 0:
                continue
            neg_idx_.append(np.where(subsample_index == neg_idx[i])[0][0])
        neg_idx = np.array(neg_idx_)

    colors = plt.cm.twilight_shifted(np.linspace(0, 1, n_bins))
    # colors = plt.cm.hsv(np.linspace(0, 1, n_bins))

    def _plot_cond(cond_vals, kind):
        fig, axes = plt.subplots(2, n_bins, figsize=(8, 4), sharey=True)
        for j in range(n_bins):
            for row, idxs in enumerate((pos_idx, neg_idx)):
                
                ax = axes[row, j]
                ax.set_title(f"{kind}: {cond_centers[j]:.2f}")
                ax.set_xlabel(r"$\theta_t$")
                if j == 0:
                    ax.set_ylabel("Rectified Activity" if rectify else "Activity")
                else:
                    ax.tick_params(labelleft=False)
                ax.set_xlim(-np.pi, np.pi)
                ax.set_xticks([-np.pi, 0, np.pi])
                ax.set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
                ax.grid(True, linestyle="--", alpha=0.6)

                c = colors[j]
                # print(idxs)
                for n in idxs:
                    if n >= N: 
                        continue
                    # mask for this condition and this neuron
                    if cond_vals.ndim == 2:
                        cond_mask = (cond_vals[:, n] >= cond_edges[j]) & (cond_vals[:, n] < cond_edges[j+1])
                    else:
                        cond_mask = (cond_vals >= cond_edges[j]) & (cond_vals < cond_edges[j+1])

                    # compute self-angle tuning within this cond_bin
                    tuning = np.zeros(n_theta_bins)
                    for m in range(n_theta_bins):
                        m2 = (theta1 >= theta_edges[m]) & (theta1 < theta_edges[m+1]) & cond_mask
                        tuning[m] = act[m2, n].mean() if np.any(m2) else 0
                    
                    # tuning = np.roll(theta_centers, -preferred_index[n])
                    centered_theta = ((theta_centers - preferred[n] + np.pi) % (2*np.pi)) - np.pi
                    sorting_index = np.argsort(centered_theta)
                    centered_theta = centered_theta[sorting_index]
                    tuning = tuning[sorting_index]
                    ax.plot(centered_theta, tuning, color=c, alpha=0.5)

        # plt.suptitle(f"{population} Population Tuning Curves Binned, {kind}s")
        plt.suptitle(f"Controller Tuning Curves by Identity")
        plt.tight_layout()
        if save: 
            plt.savefig(f"{FIG_PATH}/{population}_tuning_curves_{kind.lower()}_binned.png")
        if show: 
            plt.show()
        return fig

    # plot preferred-relative goal
    fig1 = _plot_cond(rel_goal, "Goal")
    # plot residual
    fig2 = _plot_cond(residual, "Residual")
    
    return fig1, fig2


def plot_binned_metric_by_angle(residuals,
                                velocities,
                                n_bins=8,
                                plot_velocity=True,
                                plot_speed=False,
                                shade_stds=True,
                                model_name = None,
                                xlab = "Residual Angle",
                                ylab = "Average Velocity / Speed",
                                save=False,
                                save_dir='.',
                                show=True,
                                ax = None, 
                                marker = None, 
                                label = None, 
                                ):
    """
    Plot binned average velocity and/or speed with shaded SEM as a function of residual angle.

    Args:
        residuals:       array or tensor of shape (...,) or (...,1) with angles in [-pi, pi]
        velocities:      array or tensor matching residuals, giving angular velocity
        n_bins (int):    number of angular bins to average over
        plot_velocity:   if True, plot mean velocity
        plot_speed:      if True, plot mean speed (abs velocity)
        shade_stds (bool): if True, shade mean ± SEM for each plotted metric
        save (bool):     if True, save the figure to save_dir
        save_dir (str):  directory path for saving
        show (bool):     if True, display the plot

    Returns:
        centers: array of bin center angles
        stats:   dict with keys 'velocity' and/or 'speed', each a tuple (means, sems)
    """
    # Convert tensors to numpy
    if isinstance(residuals, torch.Tensor):
        residuals = residuals.detach().cpu().numpy()
    if isinstance(velocities, torch.Tensor):
        velocities = velocities.detach().cpu().numpy()

    # Flatten
    res = residuals.ravel()
    vel = velocities.ravel()

    # Define bins
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    # Prepare stats container
    stats = {}

    # Compute stats for velocity
    if plot_velocity:
        means_v = np.zeros(n_bins)
        sems_v  = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (res >= edges[i]) & (res < edges[i+1])
            vals = vel[mask]
            if vals.size:
                means_v[i] = vals.mean()
                sems_v[i]  = vals.std(ddof=1) / np.sqrt(vals.size)
            else:
                means_v[i] = np.nan
                sems_v[i]  = np.nan
        stats['velocity'] = (means_v, sems_v)

    # Compute stats for speed
    if plot_speed:
        spd = np.abs(vel)
        means_s = np.zeros(n_bins)
        sems_s  = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (res >= edges[i]) & (res < edges[i+1])
            vals = spd[mask]
            if vals.size:
                means_s[i] = vals.mean()
                sems_s[i]  = vals.std(ddof=1) / np.sqrt(vals.size)
            else:
                means_s[i] = np.nan
                sems_s[i]  = np.nan
        stats['speed'] = (means_s, sems_s)


    # Plot
    if ax is None: 
        fig, ax = plt.subplots(figsize=(6, 4))
    else: 
        fig = ax.get_figure()
    cmap = plt.cm.twilight_shifted

    # Velocity plot
    if plot_velocity:
        means_v, sems_v = stats['velocity']
        color_v = cmap(0.2)
        if shade_stds:
            ax.fill_between(centers,
                            means_v - sems_v,
                            means_v + sems_v,
                            color=color_v, alpha=0.3)
        if marker is not None:
            ax.plot(centers, means_v, marker=marker, color=color_v, label=f'Velocity{"" if label is None else ": " + label}')                    
        else:
            ax.plot(centers, means_v, '-o', color=color_v, label= f'Velocity{"" if label is None else ": " + label}')

    # Speed plot
    if plot_speed:
        means_s, sems_s = stats['speed']
        color_s = cmap(0.8)
        if shade_stds:
            ax.fill_between(centers,
                            means_s - sems_s,
                            means_s + sems_s,
                            color=color_s, alpha=0.3)
        if marker is not None:
            ax.plot(centers, means_s, marker=marker, color=color_s, label= f'Speed{"" if label is None else ": " + label}')
        else:
            ax.plot(centers, means_s, '-o', color=color_s, label= f'Speed{"" if label is None else ": " + label}')

    # Styling
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'])

    ax.set_xlabel(xlab)
    if ylab is not None:
        ax.set_ylabel(ylab)
    else: 
        ylabel_parts = []
        if plot_velocity: ylabel_parts.append('Velocity')
        if plot_speed:    ylabel_parts.append('Speed')
        ax.set_ylabel(' / '.join(['Average'] + ylabel_parts))
    ax.grid(True, linestyle='--', alpha=0.6)

    if plot_velocity or plot_speed:
        ax.legend()

    plt.suptitle(f"Binned Metrics vs Residual Angle{(", model " + model_name) if model_name is not None else ""}")
    plt.tight_layout()

    # Save or show
    if save:
        fname = f"{save_dir}/binned_vs_angle.png"
        plt.savefig(fname)
    if show:
        plt.show()

    return centers, stats, fig, ax



def collect_trial_stats(
    conditions,
    iterations,
    action_weights,
    trials,
    model_save_path,
    trainer_factory,
    mapnet,
    loss_traj_weight = 0.5,
    convergence_loss_weight=0.3,
    control_norm_weight=0.1,

    control_period=20,
    inspect=False,#"single_sheet" if conditions[cond]["hidden_control_dim"] == 0 else "hidden" + str(conditions[cond]["hidden_control_dim"])}_{conditions[cond]["output_size"]}_{loss_traj_weight}_{action_weight}_{convergence_loss_weight}_{control_norm_weight}_iteration{it}{"no_recurrence" if not conditions[cond]["recurrent_control"] else ""}.pth
    use_s0=False,
    observe=5,
):
    """
    Run experiments across multiple conditions, iterations, and action_weights,
    collecting per-trial statistics into a DataFrame.

    Parameters
    ----------
    conditions : dict
        Mapping condition_name -> dict of model params to pass to trainer_factory.
    iterations : list of int
        Which checkpoint iterations to load.
    action_weights : list of float
        List of action_weight hyperparameters to test.
    trials : int
        Number of total trials (must be a perfect square for meshgrid).
    model_save_path : str
        Base path prefix where checkpoints are stored. Checkpoint filenames are assumed to be:
            {model_save_path}/{condition_name}_iteration{it}_aw{action_weight}.pth
            checkpath = MODEL_SAVE_PATH + f"/LAL/modularized_cp_
    trainer_factory : callable
        Function taking a condition_params dict and returning a fresh trainer instance.
    mapnet : nn.Module
        The "readout" network mapping q0/q1 to states, already on CUDA.
    control_period : int, default=20
    inspect : bool, default=False
        Whether to pass inspect to forward_return_hidden.
    use_s0 : bool, default=False
        Whether to use s0 in the model forward.
    observe : int, default=5
        Observe parameter for forward_return_hidden.

    Returns
    -------
    pd.DataFrame
        Rows: one per trial per condition/iteration/action_weight.
        Columns:
         - condition
         - iteration
         - action_weight
         - q0, q1 (float radians)
         - operator_final_mse
         - predicted_final_mse
         - final_distance (neural)
         - avg_distance (neural)
         - inferred_action_magnitude
         - control_norm
    """
    # Prepare q0, q1 grid
    side = int(np.sqrt(trials))
    q0_lin = torch.linspace(0, 2*np.pi, side, device='cuda')
    q1_lin = torch.linspace(0, 2*np.pi, side, device='cuda')
    q0g, q1g = torch.meshgrid(q0_lin, q1_lin, indexing='ij')
    q0 = q0g.reshape(-1, 1)  # [trials,1]
    q1 = q1g.reshape(-1, 1)  # [trials,1]

    records = []
    # Loop over conditions
    for cond_name, cond_params in conditions.items():
        # Instantiate trainer
        trainer = trainer_factory(cond_params)
        # For each iteration and action weight
        for it in iterations:
            for aw in action_weights:
                # Build checkpoint path
                ckpt = f"{model_save_path}{"single_sheet" if conditions[cond_name]["hidden_control_dim"] == 0 else "hidden" + str(conditions[cond_name]["hidden_control_dim"])}_{conditions[cond_name]["output_size"]}_{loss_traj_weight}_{aw}_{convergence_loss_weight}_{control_norm_weight}_iteration{it}{"no_recurrence" if not conditions[cond_name]["recurrent_control"] else ""}.pth"
                trainer.load_checkpoint(ckpt)

                # Prepare inputs
                init = mapnet(q0)
                final_state = mapnet(q1)  # [..., -128:] slicing if needed

                trainer.model.activate_controller()
                trainer.model.remove_disturbance()
                with torch.no_grad():
                    returned, _ = trainer.model.forward_return_hidden(
                        init, final_state,
                        control_period=control_period,
                        inspect=inspect,
                        use_s0=use_s0,
                        s0=(q0 if use_s0 else None),
                        observe=observe,
                        include={"im1":["hidden","output"], "controller":["hidden","output"]}
                    )
                # Unpack returned
                (hi_traj, ctrl_traj, err_traj,
                 ht_traj, ht_im1_traj, ht_pred_traj,
                 s_vec,
                 action_inf_traj,
                 state_pred_err_traj,
                 state_pred_traj,
                 out_traj) = returned

                # Compute metrics per trial
                # operator and predicted positions:
                op_pos = s_vec[:, :, :2].cpu()
                pred_pos = state_pred_traj[:, :, :2].cpu()
                op_theta = pos2theta(op_pos) % (2*np.pi)
                pred_theta = pos2theta(pred_pos) % (2*np.pi)
                # ground truth angle q1g
                q1_cpu = q1.cpu().squeeze()
                # final arc MSE per trial
                op_err = (expand_trig(op_theta[:, -1]) - expand_trig(q1_cpu[:, -1])).norm(dim=1)
                pred_err = (expand_trig(pred_theta[:, -1]) - expand_trig(q1_cpu[:, -1])).norm(dim=1)

                # neural distances per trial
                # final_distance: error at last step
                final_dist = err_traj[:, -1].cpu().norm(dim=1)
                # avg_distance: mean over time
                avg_dist = err_traj.norm(dim=-1).mean(dim=1).cpu()

                # inferred action magnitude per trial
                inf_mag = action_inf_traj.norm(p=2, dim=-1).mean(dim=1).cpu()
                # control norm per trial
                ctrl_norm = ctrl_traj.norm(p=2, dim=-1).mean(dim=1).cpu()

                # Flatten arrays to shape [trials]
                for i in range(trials):
                    records.append({
                        'condition': cond_name,
                        'iteration': it,
                        'action_weight': aw,
                        'q0': float(q0[i].cpu()),
                        'q1': float(q1[i].cpu()),
                        'operator_final_mse': float(op_err[i]),
                        'predicted_final_mse': float(pred_err[i]),
                        'final_distance': float(final_dist[i]),
                        'avg_distance': float(avg_dist[i]),
                        'inferred_action_magnitude': float(inf_mag[i]),
                        'control_norm': float(ctrl_norm[i]),
                    })

    df = pd.DataFrame(records)
    return df


def collect_test_stats(
    tests: dict,
    trainers,
    trainer_names,
    trials: int,
    control_period: int,
):
    """
    Runs a batch of controller tests under varying noise/disturbance conditions.

    Args:
        tests: dict mapping parameter names to lists of values to try (keys: 'actuation_noise', 'feedback_noise', 'disturbance_size', 'N_targets').
        trainers: a Trainer or list of Trainer instances to evaluate.
        trainer_names: list of strings naming each trainer (must match trainers list).
        trials: number of random q0 samples (per test).
        control_period: control period to use in each run.

    Returns:
        DataFrame of per-trial metrics for each condition.
    """
    # Default settings
    default_params = {
        'actuation_noise': 0.0,
        'feedback_noise': 0.5,
        'disturbance_size': 3.0,
        'N_targets': 1,
    }

    # Ensure trainers is a list
    if not isinstance(trainers, (list, tuple)):
        trainers = [trainers]
    if len(trainer_names) != len(trainers):
        raise ValueError("trainer_names must match number of trainers")

    records = []
    # Loop over each trainer
    for trainer, name in zip(trainers, trainer_names):
        device = trainer.device if hasattr(trainer, 'device') else 'cuda'
        # Loop over each parameter to test
        for param, values in tests.items():
            for v in values:
                # Reset to defaults
                trainer.model.operator.actuation_noise = default_params['actuation_noise']
                trainer.model.operator.feedback_noise = default_params['feedback_noise']
                trainer.model.remove_disturbance()
                N_targets = int(default_params['N_targets'])

                # Apply the test parameter
                if param == 'actuation_noise':
                    trainer.model.operator.actuation_noise = v
                elif param == 'feedback_noise':
                    trainer.model.operator.feedback_noise = v
                elif param == 'disturbance_size':
                    trainer.model.introduce_disturbance(scale=v, index=10)
                elif param == 'N_targets':
                    N_targets = int(v)
                else:
                    raise ValueError(f"Unknown test parameter: {param}")

                # Sample random initial and target angles
                q0 = torch.rand(trials, 1, device=device) * 2 * np.pi  # [trials, 1]
                q1 = torch.rand(trials, N_targets, 1, device=device) * 2 * np.pi  # [trials, N_targets, 1]

                # Prepare controller inputs
                # init = mapnet(q0)  # maps angle to initial CAN state
                init =final[0, -1, :].expand(trials, -1)
                final_ = mapnet(q1.view(-1, 1))
                # extract last 128 dims and reshape to [trials, N_targets, -1]
                final_state = final_[..., -128:].reshape(trials, N_targets, -1).float()

                # Activate controller and run
                trainer.model.activate_controller()
                with torch.no_grad():
                    returned, _ = trainer.model.forward_return_hidden(
                        init,
                        final_state,
                        control_period=control_period,
                        inspect=False,
                        use_s0=True,
                        s0=q0,
                        observe=10,
                        include={"im1": ["hidden", "output"], "controller": ["hidden", "output"]},
                        
                    )

                # Unpack network activities
                (
                    hi_traj,
                    ctrl_traj,
                    err_traj,
                    ht_traj,
                    ht_im1_traj,
                    ht_pred_traj,
                    s_vec,
                    action_inf_traj,
                    state_pred_err_traj,
                    state_pred_traj,
                    out_traj,
                ) = returned

                # Compute metrics per trial
                op_pos = s_vec[:, 1:, :2].cpu()
                pred_pos = state_pred_traj[:, :, :2].cpu()
                op_theta = pos2theta(op_pos) % (2 * np.pi)
                pred_theta = pos2theta(pred_pos) % (2 * np.pi)
                q1_cpu = q1.cpu()
                idx = torch.arange(1, q1_cpu.size(1)+1, device=op_theta.device) * control_period - 1  

                # grab the final angles for operator & predictor at each target
                op_theta_final   = op_theta[:, idx]    # [trials, N_targets]
                pred_theta_final = pred_theta[:, idx]  # [trials, N_targets]

                print(op_theta_final.shape, pred_theta_final.shape, q1_cpu.shape)
                # squeeze out the extra dim on q1_cpu -> [trials, N_targets]
                q1_vals = q1_cpu           # [trials, N_targets]

                # compute per‐target arc‐error vectors and then norm+mean
                op_diff  = expand_trig(op_theta_final)   - expand_trig(q1_vals)   # [trials, N_targets, 2]
                pred_diff= expand_trig(pred_theta_final) - expand_trig(q1_vals)   # [trials, N_targets, 2]

                op_err   = op_diff.norm(dim=-1).mean(dim=1)    # [trials]
                pred_err = pred_diff.norm(dim=-1).mean(dim=1)  # [trials]

                # Average arc MSE per trial
            
                op_err_traj = (expand_trig(op_theta).reshape(trials, N_targets, control_period, -1) - expand_trig(q1_cpu).unsqueeze(-2)).norm(dim=-1).reshape(trials, N_targets * control_period, -1).mean(dim=1)
                # pred_err_traj = (expand_trig(pred_theta) - expand_trig(q1_cpu)).norm(dim=-1).mean(dim=1)
                pred_err_traj = (expand_trig(pred_theta).reshape(trials, N_targets, control_period, -1) - expand_trig(q1_cpu).unsqueeze(-2)).norm(dim=-1).reshape(trials, N_targets * control_period, -1).mean(dim=1)

                # Neural distances
                final_dist = err_traj[:, -1].cpu()
                avg_dist = err_traj.mean(dim=-1).cpu()

                # Inferred action magnitude and control norm
                inf_mag = action_inf_traj.norm(p=2, dim=-1).mean(dim=1).cpu()
                ctrl_norm = ctrl_traj.norm(p=2, dim=-1).mean(dim=1).cpu()

                # Record results
                for i in range(trials):
                    records.append({
                        'trainer': name,
                        'parameter': param,
                        'value': v,
                        'q0': float(q0[i]),
                        'q1': float(q1_cpu[i, -1]),
                        'operator_final_mse': float(op_err[i]),
                        'predicted_final_mse': float(pred_err[i]),
                        'operator_avg_mse': float(op_err_traj[i]),
                        'final_distance': float(final_dist[i]),
                        'avg_distance': float(avg_dist[i]),
                        'inferred_action_magnitude': float(inf_mag[i]),
                        'control_norm': float(ctrl_norm[i]),
                    })

    return pd.DataFrame(records)

def collect_test_stats(
        tests: dict, 
        trainers,  # for now I have one GRU trainer that can load them all! 
        trainer_names: list = ["LAL_GRU"],
        its = [0, 1, 2],
        n_targs_trained = [1, 2, 3, 4],
        trials = 1024, 
        control_period = 20,
    ): 

    
    default_params = {
        'actuation_noise': 0.0,
        'feedback_noise': 0.5,
        'disturbance_size': 3.0,
        'N_targets': 1,
    }

    # Ensure trainers is a list
    if not isinstance(trainers, (list, tuple)):
        trainers = [trainers]
    if len(trainer_names) != len(trainers):
        raise ValueError("trainer_names must match number of trainers")
    records = []
    # Loop over each trainer
    for trainer, name in zip(trainers, trainer_names):
        for num_targets in n_targs_trained:
            for it in its: 
                path = MODEL_SAVE_PATH + f"/LAL/gru_{num_targets}_iteration{it}"
                device = trainer.device if hasattr(trainer, 'device') else 'cuda'
                # Loop over each parameter to test
                for param, values in tests.items():
                    for v in values:
                        # Reset to defaults
                        trainer.model.operator.actuation_noise = default_params['actuation_noise']
                        trainer.model.operator.feedback_noise = default_params['feedback_noise']
                        trainer.model.remove_disturbance()
                        N_targets = int(default_params['N_targets'])

                        # Apply the test parameter
                        if param == 'actuation_noise':
                            trainer.model.operator.actuation_noise = v
                        elif param == 'feedback_noise':
                            trainer.model.operator.feedback_noise = v
                        elif param == 'disturbance_size':
                            trainer.model.introduce_disturbance(scale=v, index=10)
                        elif param == 'N_targets':
                            N_targets = int(v)
                        else:
                            raise ValueError(f"Unknown test parameter: {param}")

                        # Sample random initial and target angles
                        q0 = torch.rand(trials, 1, device=device) * 2 * np.pi  # [trials, 1]
                        q1 = torch.rand(trials, N_targets, 1, device=device) * 2 * np.pi  # [trials, N_targets, 1]

                        # Prepare controller inputs
                        # init = mapnet(q0)  # maps angle to initial CAN state
                        init =final[0, -1, :].expand(trials, -1)
                        final_ = mapnet(q1.view(-1, 1))
                        # extract last 128 dims and reshape to [trials, N_targets, -1]
                        final_state = final_[..., -128:].reshape(trials, N_targets, -1).float()

                        # Activate controller and run
                        trainer.model.activate_controller()
                        with torch.no_grad():
                            returned, _ = trainer.model.forward_return_hidden(
                                init,
                                final_state,
                                control_period=control_period,
                                inspect=False,
                                use_s0=True,
                                s0=q0,
                                observe=10,
                                include={"im1": ["hidden", "output"], "controller": ["hidden", "output"]},
                                
                            )

                        # Unpack network activities
                        (
                            hi_traj,
                            ctrl_traj,
                            err_traj,
                            ht_traj,
                            ht_im1_traj,
                            ht_pred_traj,
                            s_vec,
                            action_inf_traj,
                            state_pred_err_traj,
                            state_pred_traj,
                            out_traj,
                        ) = returned

                        # Compute metrics per trial
                        op_pos = s_vec[:, 1:, :2].cpu()
                        pred_pos = state_pred_traj[:, :, :2].cpu()
                        op_theta = pos2theta(op_pos) % (2 * np.pi)
                        pred_theta = pos2theta(pred_pos) % (2 * np.pi)
                        q1_cpu = q1.cpu()
                        idx = torch.arange(1, q1_cpu.size(1)+1, device=op_theta.device) * control_period - 1  

                        # grab the final angles for operator & predictor at each target
                        op_theta_final   = op_theta[:, idx]    # [trials, N_targets]
                        pred_theta_final = pred_theta[:, idx]  # [trials, N_targets]

                        print(op_theta_final.shape, pred_theta_final.shape, q1_cpu.shape)
                        # squeeze out the extra dim on q1_cpu -> [trials, N_targets]
                        q1_vals = q1_cpu           # [trials, N_targets]

                        # compute per‐target arc‐error vectors and then norm+mean
                        op_diff  = expand_trig(op_theta_final)   - expand_trig(q1_vals)   # [trials, N_targets, 2]
                        pred_diff= expand_trig(pred_theta_final) - expand_trig(q1_vals)   # [trials, N_targets, 2]

                        op_err   = op_diff.norm(dim=-1).mean(dim=1)    # [trials]
                        pred_err = pred_diff.norm(dim=-1).mean(dim=1)  # [trials]

                        # Average arc MSE per trial
                    
                        op_err_traj = (expand_trig(op_theta).reshape(trials, N_targets, control_period, -1) - expand_trig(q1_cpu).unsqueeze(-2)).norm(dim=-1).reshape(trials, N_targets * control_period, -1).mean(dim=1)
                        # pred_err_traj = (expand_trig(pred_theta) - expand_trig(q1_cpu)).norm(dim=-1).mean(dim=1)
                        pred_err_traj = (expand_trig(pred_theta).reshape(trials, N_targets, control_period, -1) - expand_trig(q1_cpu).unsqueeze(-2)).norm(dim=-1).reshape(trials, N_targets * control_period, -1).mean(dim=1)

                        # Neural distances
                        final_dist = err_traj[:, -1].cpu()
                        avg_dist = err_traj.mean(dim=-1).cpu()

                        # Inferred action magnitude and control norm
                        inf_mag = action_inf_traj.norm(p=2, dim=-1).mean(dim=1).cpu()
                        ctrl_norm = ctrl_traj.norm(p=2, dim=-1).mean(dim=1).cpu()

                        # Record results
                        for i in range(trials):
                            records.append({
                                'trainer': name,
                                'parameter': param,
                                'value': v,
                                'q0': float(q0[i]),
                                'q1': float(q1_cpu[i, -1]),
                                'operator_final_mse': float(op_err[i]),
                                'predicted_final_mse': float(pred_err[i]),
                                'operator_avg_mse': float(op_err_traj[i]),
                                'final_distance': float(final_dist[i]),
                                'avg_distance': float(avg_dist[i]),
                                'inferred_action_magnitude': float(inf_mag[i]),
                                'control_norm': float(ctrl_norm[i]),
                                "iteration": it,
                                "n_targets_trained": num_targets,
                            })

            return pd.DataFrame(records)
