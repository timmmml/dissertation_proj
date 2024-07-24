""" Contains the main paths used
"""
from pathlib import Path
import os

MODEL_SAVE_PATH = os.path.abspath(Path('D:/Projects/mental-rotations/models/'))
DATA_PATH = os.path.abspath(Path('D:/Projects/mental-rotations/data/'))
LOG_PATH = os.path.abspath(Path("D:/Projects/mental-rotations/logs/"))
CONFIG_PATH = os.path.abspath(Path("C:/Users/timmy/Documents/Projects_dir/Mental_Rotations/mental-rotations/saved_configs/"))
OBJECT_PATH = os.path.abspath(Path("C:/Users/timmy/Documents/Projects_dir/Mental_Rotations/mental-rotations/data/"))
FIG_PATH = os.path.abspath(Path("C:/Users/timmy/Documents/Projects_dir/Mental_Rotations/mental-rotations/figures/"))

for path in [MODEL_SAVE_PATH, DATA_PATH, LOG_PATH, CONFIG_PATH]:
    os.makedirs(path, exist_ok=True)