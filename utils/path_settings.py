""" Contains the main paths used
"""
from pathlib import Path
import os
print(os.getcwd())
from .goto_project_root import run

MODEL_SAVE_PATH = os.path.abspath(Path('saved_models/'))
DATA_PATH = os.path.abspath(Path('training_data/'))
LOG_PATH = os.path.abspath(Path("logs/"))
CONFIG_PATH = os.path.abspath(Path("saved_configs/"))
OBJECT_PATH = os.path.abspath(Path("data/"))
FIG_PATH = os.path.abspath(Path("figures/"))

for path in [MODEL_SAVE_PATH, DATA_PATH, LOG_PATH, CONFIG_PATH]:
    os.makedirs(path, exist_ok=True)