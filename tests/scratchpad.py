
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
from tabnanny import verbose
import warnings
from pathlib import Path
from typing import Any, Dict, List
from xml.etree import ElementTree

import audio_metadata
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
from pykanto.dataset import SongDataset
from pykanto.parameters import Parameters
from pykanto.signal.segment import get_segment_info, segment_files_parallel
from pykanto.utils.compute import flatten_list, to_iterator, tqdmm
from pykanto.utils.custom import chipper_units_to_json, parse_sonic_visualiser_xml
from pykanto.utils.paths import (ProjDirs, get_file_paths,
                                 get_wavs_w_annotation)
from pykanto.utils.paths import pykanto_data
from pykanto.utils.write import makedir

warnings.simplefilter('always', ImportWarning)
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"

# REVIEW - remove when complete
# %load_ext autoreload
# %autoreload 2

# %%──── SETTINGS ──────────────────────────────────────────────────────────────

# Import package data

# DATASETS_ID = ["STORM-PETREL", "BENGALESE_FINCH", "GREAT_TIT"]
# for dataset in DATASETS_ID:
#     DIRS = pykanto_data(dataset=dataset)
#     print(DIRS)


DATASET_ID = "GREAT_TIT"
DIRS = pykanto_data(dataset=DATASET_ID)
print(DIRS)

# %%
params = Parameters(dereverb=True, verbose=False)
dataset = SongDataset(DATASET_ID, DIRS, parameters=params,
                      overwrite_dataset=True, overwrite_data=True)

dataset.vocs.head()
# %%

# storm petrel
DATASET_ID = "STORM-PETREL"
DATA_PATH = Path(pkg_resources.resource_filename('pykanto', 'data'))
PROJECT = Path(DATA_PATH).parent
RAW_DATA = DATA_PATH / 'raw' / DATASET_ID

DIRS = ProjDirs(PROJECT, RAW_DATA, mkdir=True)

wav_filepaths, xml_filepaths = [get_file_paths(
    DIRS.RAW_DATA, [ext]) for ext in ['.wav', '.xml']]
files_to_segment = get_wavs_w_annotation(wav_filepaths, xml_filepaths)


segment_files_parallel(
    files_to_segment,
    DIRS,
    resample=22050,
    parser_func=parse_sonic_visualiser_xml,
    min_duration=.1,
    min_freqrange=100,
    labels_to_ignore=["NOISE"]
)

outfiles = [get_file_paths(DIRS.SEGMENTED, [ext])
            for ext in ['.wav', '.JSON']]


# 2. Chipper outputs segmentation information to a gzip file. Let's add this to
# the JSON metadata created above.

# # These files can be anywhere, but here the external segmentation files
# are in a subdirectory within `data/segmented``

chipper_units_to_json(DIRS.SEGMENTED)

# %%


# Define parameters
params = Parameters(
    # Spectrogramming
    window_length=512,
    hop_length=32,
    n_fft=2048,
    num_mel_bins=240,
    sr=22050,
    top_dB=65,                  # top dB to keep
    lowcut=300,
    highcut=10000,
    dereverb=True,
    # general settings
    song_level=True,
    subset=(0, -1),
    verbose=False,
)

# np.random.seed(123)
# random.seed(123)
dataset = SongDataset(
    DATASET_ID, DIRS, parameters=params, overwrite_dataset=True,
    overwrite_data=False)

# Segmente into individual units using information from chipper,
# then check a few.
dataset.segment_into_units()

for voc in dataset.vocs.index:
    dataset.plot_voc_seg(voc)


# %%

to_rm = [dataset.DIRS.DATASET.parent,
         dataset.DIRS.SEGMENTED/'WAV', dataset.DIRS.SEGMENTED/'JSON']
for path in to_rm:
    if path.exists():
        shutil.rmtree(str(path))
assert all([f.exists() for f in to_rm]) == False
