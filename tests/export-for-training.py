from pathlib import Path

import pkg_resources
from pykanto.dataset import KantoData
from pykanto.parameters import Parameters

from pykanto.utils.paths import ProjDirs, pykanto_data
from pykanto.utils.read import load_dataset


DATASET_ID = "GRETI_2021"
DATA_PATH = Path(pkg_resources.resource_filename("pykanto", "data"))
PROJECT = Path("/home/nilomr/Downloads/")
RAW_DATA = PROJECT
DIRS = ProjDirs(PROJECT, RAW_DATA, mkdir=True)

dataset_path = Path("/home/nilomr/Downloads") / DATASET_ID / f"{DATASET_ID}.db"
dataset = load_dataset(dataset_path, DIRS)
dataset.plot(dataset.vocs.index[0])

len(dataset.vocs)

import ast
import numpy as np
import pandas as pd


def from_np_array(array_string):
    array_string = ",".join(array_string.replace("[ ", "[").split())
    return np.array(ast.literal_eval(array_string))


def str_to_path(path):
    return Path(path)


import pandas as pd

recover_csv = pd.read_csv(
    dataset.DIRS.DATASET.parent / f"{DATASET_ID}_VOCS.csv",
    index_col=0,
    converters={
        "unit_durations": from_np_array,
        "onsets": from_np_array,
        "offsets": from_np_array,
        "silence_durations": eval,  # TODO why is this a list!
        "spectrogram_loc": str_to_path,  # and all other paths!
    },
)

dataset.vocs = recover_csv

dataset.save_to_disk(dataset_path)


len(set(dataset.vocs.auto_type_label))


dataset.vocs.groupby(["ID", "auto_type_label"]).size()
