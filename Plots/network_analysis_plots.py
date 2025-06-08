"""This function is used to create panelled plots for analysing networks'
performance

Args:
    networks: a list of names specifying networks to compare with.
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
    #
    #
    # plots: a list of plots to be made.
    #     Options are {"total loss",
    #         "output norm loss",
    #         "final geodesic loss",
    #         "gradual geodesic loss",
    #         "silence period activity norm"
    #     }
"""
import Rotations as Rot
import seaborn as sns
import matplotlib.pyplot as plt
from utils.path_settings import MODEL_SAVE_PATH, DATA_PATH, LOG_PATH, CONFIG_PATH
import SimulateDatasets.GenTrainingData as g
from Network_models import Trainer as t
from importlib import reload
import os
reload(t)
reload(g)
import time
import torch
import numpy as np
import json

from Train_task.train_from_config_names import build_config

def network_analysis_plots_CNN(network_config_paths, output_dir = None, special_name = None, stage = "CNN"):
    # Generate test data for all networks:
    "NOT IMPLEMENTED"
    if stage != "CNN":
        raise NotImplementedError
    # Initialise a dictionary to store the results
    results = {
        "Total Loss": {},
        "Output L2 Norm (rad/s)": {},
        "Final Geodesic Distance (rad)": {},
        "Gradually Summed Geodesic Distance (rad)": {},
    }

    for network_config_path in network_config_paths:
        if not network_config_path.endswith(".json"):
            network_config_path = network_config_path + ".json"
        config = json.load(open(network_config_path))
        network = config['model_id'] + f"_res{config['training_config']['resolution']}"
        trainer = t.Trainer(config) # initialise a trainer
        trainer.load_model(config['save_path'] + f"\\best_model.pth", full=1)
        test_data = torch.load(config['training_config']['data_save_path'][:-4] + "_test.pth")
        # Use dataloader if needed (for large datasets split it up)
        # set up output storage to concatenate onto (dim 0)
        output, loss = None, None
        if True: #not os.path.isfile(config['save_path'] + f"\\best_model_out_loss.pth"):
            test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)
            for i, (features, target) in enumerate(test_dataloader):
                features = features.to(trainer.device)
                target = target.to(trainer.device)
                if output is None: 
                    output, loss = trainer.forward(features, target)
                else: 
                    output_n, loss_n = trainer.forward(features, target)
                    output = torch.cat((output, output_n), dim=0)
                    loss = torch.cat((loss, loss_n), dim=0)
            torch.save((output, loss), config['save_path'] + f"\\best_model_out_loss.pth")
        else:
            output, loss = torch.load(config['save_path'] + f"\\best_model_out_loss.pth")
        results["Total Loss"][network] = loss.detach().to("cpu").numpy()
        results["Output L2 Norm (rad/s)"][network] = trainer.loss_fn.recorded_regularisation_loss.to("cpu").numpy()
        results["Final Geodesic Distance (rad)"][network] = trainer.loss_fn.final_distance_loss.to("cpu").numpy()
        results["Gradually Summed Geodesic Distance (rad)"][network] = trainer.loss_fn.recorded_distance_loss.to("cpu").numpy()
    results['Final Geodesic Distance (rad)']["Random"] =Rot.geodesic_distance(target, torch.rand_like(target)).cpu().numpy()
    plot_violin(results, output_dir)
    return results


def network_analysis_plots_pretrain(networks, task, output_dir = None, special_name = None, stage = "pretrain", test_data_path = None):
    if stage != "pretrain":
            raise NotImplementedError
    # Generate test data for all networks:
    test_data = []
    training_config = {
        "task_id": task,
        "batch_size": 1000,
        "seq_len": 100,
        "prep_phase": 50,
    }
    if test_data_path is not None:
        test_data = torch.load(test_data_path)
    else:
        test_data = g.gen_training_data(training_config)

    # Initialise a dictionary to store the results
    results = {
        "Total Loss": {},
        "Output L2 Norm (rad/s)": {},
        "Final Geodesic Distance (rad)": {},
        "Gradually Summed Geodesic Distance (rad)": {},
    }

    for network in networks:
        config = build_config(network, task, 0, 1, special_names=special_name)
        trainer = t.Trainer(config) # initialise a trainer
        trainer.load_model(config['save_path'] + f"\\best_model.pth", full=1)
        features, target = test_data.data, test_data.labels
        features = features.to(trainer.device)
        target = target.to(trainer.device)
        output, loss = trainer.forward(features, target)
        results["Total Loss"][network] = loss.detach().to("cpu").numpy()
        results["Output L2 Norm (rad/s)"][network] = trainer.loss_fn.recorded_regularisation_loss.to("cpu").numpy()
        results["Final Geodesic Distance (rad)"][network] = trainer.loss_fn.final_distance_loss.to("cpu").numpy()
        results["Gradually Summed Geodesic Distance (rad)"][network] = trainer.loss_fn.recorded_distance_loss.to("cpu").numpy()
    results['Final Geodesic Distance (rad)']["Random"] =Rot.geodesic_distance(target, torch.rand_like(target)).cpu().numpy()
    # plot_violin(results, output_dir)
    plot_box(results, output_dir)
    return results

def plot_violin(results, output_dir, labels = None, figsize = None):
    if figsize is None:
        fig, axes = plt.subplots(len(results), 1, figsize=(15, 15 * (len(results))))
    else: 
        fig, axes = plt.subplots(len(results), 1, figsize=figsize)



    for n, key in enumerate(results.keys()):

        data = results[key]

        # Convert dictionary to list of lists for seaborn
        data_list = [data[net] for net in data.keys()]
        if labels is None:
            labels = list(data.keys())

        sns.violinplot(data=data_list, ax=axes[n], inner=None, palette="Set2")
        # sns.boxplot(data=data_list, ax=axes[n], width=0.2, showcaps=True,
        #             boxprops={'facecolor': 'None'}, showfliers=False, whiskerprops={'linewidth': 2})
        axes[n].set_title(key)
        axes[n].set_xticks(np.arange(len(labels)))
        axes[n].set_xticklabels(labels, rotation=45)

        medians = [np.median(data[net]) for net in data.keys()]
        tops = [np.max(data[net]) for net in data.keys()]
        bots = [np.min(data[net]) for net in data.keys()]
        for tick, label in enumerate(labels):
            axes[n].text(tick, medians[tick], f'{medians[tick]:.2f}',
                            horizontalalignment='center', color='black', weight='light')
            if np.isinf(min(bots)):
                continue
            axes[n].set_ylim(min(bots)- 0.1, max(tops) * 1.2)

    plt.tight_layout()
    if output_dir is not None:
        fig.savefig(output_dir)
    else:
        plt.show()

def plot_box(results, output_dir, labels = None, figsize = None):
    if figsize is None:
        fig, axes = plt.subplots(len(results), 1, figsize=(15, 15 * (len(results))))
    else: 
        fig, axes = plt.subplots(len(results), 1, figsize=figsize)

    for n, key in enumerate(results.keys()):
        data = results[key]

        # Convert dictionary to list of lists for seaborn
        data_list = [data[net] for net in data.keys()]
        if labels is None:
            labels = list(data.keys())
        transparent_palette = [(1, 1, 1, 0) for _ in range(len(labels) + 1)]
        sns.set_palette(transparent_palette)

        sns.boxplot(data=data_list, ax=axes[n])
        # sns.boxplot(data=data_list, ax=axes[i, j], width=0.2, showcaps=True,
        #             boxprops={'facecolor': 'None'}, showfliers=False, whiskerprops={'linewidth': 2})
        axes[n].set_title(key)
        axes[n].set_xticks(np.arange(len(labels)))
        axes[n].set_xticklabels(labels, rotation=30)
        medians = [np.median(data[net]) for net in data.keys()]
        tops = [np.max(data[net]) for net in data.keys()]
        bots = [np.min(data[net]) for net in data.keys()]
        for tick, label in enumerate(labels):
            axes[n].text(tick, medians[tick], f'{medians[tick]:.2f}',
                            horizontalalignment='center', color='black', weight='light')
            if np.isinf(min(bots)):
                continue
            axes[n].set_ylim(min(bots)- 0.1, max(tops) * 1.2)
    plt.tight_layout()
    if output_dir is not None:
        fig.savefig(output_dir)
    else:
        plt.show()
