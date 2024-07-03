from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import torch
import os
import shutil


def create_splits(data, n_splits=5):
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
