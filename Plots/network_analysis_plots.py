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
        network = config['model_id'] + f"_res{config['resolution']}"
        trainer = t.Trainer(config) # initialise a trainer
        trainer.load_model(config['save_path'] + f"\\best_model.pth", full=1)
        test_data = torch.load(config['training_config']['data_save_path'][:-4] + "_test.pth")
        if not os.path.isfile(config['save_path'] + f"\\best_model_out_loss.pth"):
            features, target = test_data[0].data, test_data[0].labels
            features = features.to(trainer.device)
            target = target.to(trainer.device)
            output, loss = trainer.forward(features, target)
        else:
            output, loss = torch.load(config['save_path'] + f"\\best_model_out_loss.pth")
        results["Total Loss"][network] = loss.detach().to("cpu").numpy()
        results["Output L2 Norm (rad/s)"][network] = trainer.loss_fn.recorded_regularisation_loss.to("cpu").numpy()
        results["Final Geodesic Distance (rad)"][network] = trainer.loss_fn.final_distance_loss.to("cpu").numpy()
        results["Gradually Summed Geodesic Distance (rad)"][network] = trainer.loss_fn.recorded_distance_loss.to("cpu").numpy()
    results['Final Geodesic Distance (rad)']["Random"] =Rot.geodesic_distance(target, torch.rand_like(target)).cpu().numpy()
    plot_violin(results, output_dir)
    return results


def network_analysis_plots_pretrain(networks, task, output_dir = None, special_name = None, stage = "pretrain"):
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
    task_data = g.gen_training_data(training_config)
    test_data.append(task_data)

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
        features, target = test_data[0].data, test_data[0].labels
        features = features.to(trainer.device)
        target = target.to(trainer.device)
        output, loss = trainer.forward(features, target)
        results["Total Loss"][network] = loss.detach().to("cpu").numpy()
        results["Output L2 Norm (rad/s)"][network] = trainer.loss_fn.recorded_regularisation_loss.to("cpu").numpy()
        results["Final Geodesic Distance (rad)"][network] = trainer.loss_fn.final_distance_loss.to("cpu").numpy()
        results["Gradually Summed Geodesic Distance (rad)"][network] = trainer.loss_fn.recorded_distance_loss.to("cpu").numpy()
    results['Final Geodesic Distance (rad)']["Random"] =Rot.geodesic_distance(target, torch.rand_like(target)).cpu().numpy()
    plot_violin(results, output_dir)
    return results

def plot_violin(results, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for n, key in enumerate(results.keys()):
        i = n // 2
        j = n % 2
        data = results[key]

        # Convert dictionary to list of lists for seaborn
        data_list = [data[net] for net in data.keys()]
        labels = list(data.keys())

        sns.violinplot(data=data_list, ax=axes[i, j], inner=None, palette="Set2")
        # sns.boxplot(data=data_list, ax=axes[i, j], width=0.2, showcaps=True,
        #             boxprops={'facecolor': 'None'}, showfliers=False, whiskerprops={'linewidth': 2})
        axes[i, j].set_title(key)
        axes[i, j].set_xticks(np.arange(len(labels)))
        axes[i, j].set_xticklabels(labels, rotation=45)
        medians = [np.median(data[net]) for net in data.keys()]
        for tick, label in zip(range(len(labels)), labels):
            axes[i, j].text(tick, medians[tick], f'Median:{medians[tick]:.2f}',
                            horizontalalignment='center', color='black', weight='semibold')
    plt.tight_layout()
    if output_dir is not None:
        fig.savefig(output_dir)
    else:
        plt.show()
