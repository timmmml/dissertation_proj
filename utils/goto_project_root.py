import os
from pathlib import Path

def run():
    """Navigates to the project root"""
    path = Path.cwd()
    while len(path.name) and path.name != "mental-rotations":
        path = path.parent

    if len(path.name):
        os.chdir(path)
    else:
        raise ValueError("Cannot find the root directory of the project.")

run()