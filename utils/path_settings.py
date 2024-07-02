""" Contains the main paths used
"""
from pathlib import Path
import os

MODEL_SAVE_PATH = os.path.abspath(Path('./saved_model/'))
DATA_PATH = os.path.abspath(Path('./data/'))
LOG_PATH = os.path.abspath(Path("./runs/"))