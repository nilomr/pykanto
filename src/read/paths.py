# Define paths, make if they don't exist

import pathlib2
from pathlib2 import Path
import os
from datetime import date

year = date.today().year


# Define directories:

PROJECT_PATH = (
    Path("__file__").resolve().parents[2]
)  # Ego = notebook!! # 0 in vscode, 2 in notebook
DATA_PATH = PROJECT_PATH / "data"
FIGURE_PATH = PROJECT_PATH / "reports" / "figures"
RESOURCES_PATH = PROJECT_PATH / "resources" / "fieldwork" / str(year)

# ----------------------

# Make these directiories should they not currently exist:


def safe_makedir(file_path):
    """Make a safely nested directory.
    From Tim Sainburg: https://github.com/timsainb/avgn_paper/blob/vizmerge/avgn/utils/paths.py

    Args:
        file_path (str or PosixPath): Path to be created.
    """
    if type(file_path) == str:
        if "." in os.path.basename(os.path.normpath(file_path)):
            directory = os.path.dirname(file_path)
        else:
            directory = os.path.normpath(file_path)
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except FileExistsError as e:
                # multiprocessing can cause directory creation problems
                print(e)
    elif type(file_path) == pathlib2.PosixPath:
        # if this is a file
        if len(file_path.suffix) > 0:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            file_path.mkdir(parents=True, exist_ok=True)


for path in DATA_PATH, FIGURE_PATH, RESOURCES_PATH:
    safe_makedir(path)
