# ─── DESCRIPTION ──────────────────────────────────────────────────────────────

# This script does the following:
#  - Segments raw recordings into songs Using segmentation information in the
#    Sonic Visualiser format
#  - These are saved as new .wav files + .json files with metadata

# TODO:
# Convert 2020 metadata to same format as 2021 so that the below works for any
# year.


# %%──── LIBRARIES ─────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import os
import pickle
import shutil
import warnings
from pathlib import Path
from tabnanny import verbose
from typing import Any, Dict, List
from xml.etree import ElementTree

import git
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
import pykanto.plot as kplot
import pytest
import ray
import seaborn as sns
import soundfile as sf
from bokeh.palettes import Set3_12
from pykanto.dataset import KantoData
from pykanto.parameters import Parameters
from pykanto.signal.segment import get_segment_info, segment_files_parallel
from pykanto.utils.compute import flatten_list, to_iterator, with_pbar
from pykanto.utils.custom import (
    chipper_units_to_json,
    parse_sonic_visualiser_xml,
)
from pykanto.utils.paths import (
    ProjDirs,
    get_file_paths,
    get_wavs_w_annotation,
    pykanto_data,
)
from pykanto.utils.write import makedir

warnings.simplefilter("always", ImportWarning)
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"

# REVIEW - remove when complete
# %load_ext autoreload
# %autoreload 2

# %%
# ──── INTERACTIVE APP DEMO ────────────────────────────────────────────────────

DATASET_ID = "LABEL_APP_DEMO"
PROJECT_ROOT = Path("/home/nilomr/projects/great-tit-song")
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "LABEL_APP_DEMO_DATA"
DIRS = ProjDirs(PROJECT_ROOT, RAW_DATA, mkdir=True)
print(DIRS)


# wav_filepaths, xml_filepaths = [get_file_paths(
#     DIRS.RAW_DATA, [ext]) for ext in ['.WAV', '.xml']]
# files_to_segment = get_wavs_w_annotation(wav_filepaths, xml_filepaths)


# segment_files_parallel(
#     files_to_segment,
#     DIRS,
#     resample=22050,
#     parser_func=parse_sonic_visualiser_xml,
#     min_duration=.1,
#     min_freqrange=100,
#     labels_to_ignore=["NOISE"]
# )


# # Define parameters
# params = Parameters(
#     # Spectrogramming
#     window_length=1024,
#     hop_length=128,
#     n_fft=1024,
#     num_mel_bins=224,
#     sr=22050,
#     top_dB=65,                  # top dB to keep
#     lowcut=2000,
#     highcut=10000,
#     # Segmentation,
#     max_dB=-30,                 # Max threshold for segmentation
#     dB_delta=5,                 # n thresholding steps, in dB
#     silence_threshold=0.1,      # Between 0.1 and 0.3 tends to work
#     max_unit_length=0.4,        # Maximum unit length allowed
#     min_unit_length=0.02,       # Minimum unit length allowed
#     min_silence_length=0.001,   # Minimum silence length allowed
#     gauss_sigma=3,              # Sigma for gaussian kernel
#     # general settings
#     song_level=True,
#     subset=None,
#     verbose=False,
#     num_cpus=None,
# )

# # np.random.seed(123)
# # random.seed(123)
# dataset = KantoData(
#     DATASET_ID, DIRS, parameters=params, overwrite_dataset=True,
#     random_subset=None, overwrite_data=False)


# dataset.segment_into_units()
# dataset.vocs['ID'] = 'TR43633'

# dataset.get_units()
# dataset.cluster_ids(min_sample=20)


out_dir = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
dataset = pickle.load(open(out_dir, "rb"))


dataset.prepare_interactive_data()
dataset.open_label_app()

dataset.save_to_disk()
