# This script stores paths in variables and makes directories if they don't exist

# Libraries

import pathlib2
import numpy as np
import pandas as pd
from pathlib2 import Path
import os
from datetime import date


# Setup

year = date.today().year
level = 0 if __name__ == "__main__" else 2


# Define directories:

PROJECT_PATH = Path("__file__").resolve().parents[level]
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


# -----------------------

# files_path = DATA_PATH / "raw" / str(year)

# filearray = np.sort(list(files_path.glob("**/*.WAV")))
# nestboxes = pd.DataFrame(set([file.parent.name for file in filelist]))

# recorded_gretis = pd.read_csv(RESOURCES_PATH / "recorded_greti_boxes.csv")[
#     "nestbox"
# ].to_list()

# filelist = pd.Series([str(path) for path in filearray])
# greti_files = filelist[filelist.str.contains("|".join(recorded_gretis))]
# greti_files.to_frame().to_csv("greti_files.csv")

# # All recordings and recorded nests:
# pd.set_option("display.max_colwidth", -1)
# display(pd.DataFrame(filearray), nestboxes)

