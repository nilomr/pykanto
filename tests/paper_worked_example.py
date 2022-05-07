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
from dateutil.parser import parse

import json
import os
import pickle
import re
import shutil
from tabnanny import verbose
import warnings
from pathlib import Path
from typing import Any, Dict, List
from xml.etree import ElementTree
import attr

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
from pykanto.dataset import KantoData
from pykanto.parameters import Parameters
from pykanto.signal.segment import (
    segment_files,
    segment_files_parallel,
    ReadWav,
    SegmentMetadata,
)
from pykanto.utils.compute import flatten_list, to_iterator, tqdmm
from pykanto.utils.custom import (
    chipper_units_to_json,
    parse_sonic_visualiser_xml,
)
from pykanto.utils.paths import ProjDirs, get_file_paths, get_wavs_w_annotation
from pykanto.utils.paths import pykanto_data
from pykanto.utils.types import Annotation, AudioAnnotation
from pykanto.utils.write import makedir
from attr import validators
import pykanto
import datetime as dt

warnings.simplefilter("always", ImportWarning)
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"

# REVIEW - remove when complete
# %load_ext autoreload
# %autoreload 2

# %%──── SETTINGS ──────────────────────────────────────────────────────────────

# For example, let's say you are using AudioMoth recorders and want to retrieve
# some non-standard metadata from its audio files: the device ID and the date
# and time time of an audio segment.

# To make it easier to see what data are available, you can create a `ReadWav`
# object and get all its available metadata, like so:

DIRS = pykanto_data(dataset="AM")  # Loads a sample AudioMoth file
wav_dir = get_file_paths(DIRS.RAW_DATA, [".WAV"])[0]
meta = ReadWav(wav_dir).all_metadata
print(meta)

# So let's acess the metadata of interest and tell `pykanto` that we want to add
# these to the .JSON files and, later, to our database.

# First, add any new attributes, along with their data type annotations and any
# validators to the Annotation class.


@attr.s
class CustomAnnotation(Annotation):
    rec_unit: str = attr.ib(validator=validators.instance_of(str))
    # This is intended as a short example, but in reality you could make sure that
    # this string can be parsed as a datetime object.
    datetime: str = attr.ib(validator=validators.instance_of(str))


Annotation.__init__ = CustomAnnotation.__init__

# Then, monkey-patch the `get_metadata` methods of the ReadWav and
# SegmentMetadata classes to add any extra fields that your project might
# require. This will save you from having to define the full classes and their
# methods again from scratch. Some people would say this is ugly, but it is the
# most concise way of doing this that I could think of while preserving enough
# flexibility.


def ReadWav_patch(self) -> Dict[str, Any]:
    comment = self.all_metadata["tags"].comment[0]
    add_to_dict = {
        "rec_unit": str(
            re.search(r"AudioMoth.(.*?) at gain", comment).group(1)
        ),
        "datetime": str(
            parse(re.search(r"at.(.*?) \(UTC\)", comment).group(1))
        ),
    }
    return {**self.metadata.__dict__, **add_to_dict}


def SegmentMetadata_patch(self) -> Dict[str, Any]:
    start = (
        self.all_metadata.start_times[self.index]
        / self.all_metadata.sample_rate
    )
    datetime = parse(self.all_metadata.datetime) + dt.timedelta(seconds=start)
    add_to_dict = {
        "rec_unit": self.all_metadata.rec_unit,
        "datetime": str(datetime),
    }
    return {**self.metadata.__dict__, **add_to_dict}


ReadWav.get_metadata = ReadWav_patch
SegmentMetadata.get_metadata = SegmentMetadata_patch

# Now you can segment your annotated files like you would normally do - their
# metadata will contain your custom fields.

wav_filepaths, xml_filepaths = [
    get_file_paths(DIRS.RAW_DATA, [ext]) for ext in [".WAV", ".xml"]
]
files_to_segment = get_wavs_w_annotation(wav_filepaths, xml_filepaths)

wav_outdir, json_outdir = [
    makedir(DIRS.SEGMENTED / ext) for ext in ["WAV", "JSON"]
]

segment_files(
    files_to_segment,
    wav_outdir,
    json_outdir,
    resample=22050,
    parser_func=parse_sonic_visualiser_xml,
    min_duration=0,
    min_freqrange=0,
    labels_to_ignore=["NOISE", "FIRST"],
)

# Note: if you want to run this in paralell (as in `segment_files_parallel`)
# this method will not work - for now, you will have to adapt the source code.

# %%

DATA_PATH = Path(pkg_resources.resource_filename("pykanto", "data"))
PROJECT = Path(DATA_PATH).parent
RAW_DATA = DATA_PATH / "raw"
DIRS = ProjDirs(PROJECT, RAW_DATA, mkdir=True)

annotation_paths = get_file_paths(DIRS.RAW_DATA, [".xml"])
wav_filepaths = get_file_paths(DIRS.RAW_DATA, [".wav"])
datapaths = get_wavs_w_annotation(wav_filepaths, annotation_paths)


ReadWav_inst = ReadWav(datapaths[0][0])
assert isinstance(ReadWav_inst.get_wav(), sf.SoundFile)
assert isinstance(ReadWav_inst.get_metadata(), AudioAnnotation)

# %%

DATASET_ID = "GRETI_2021"
DATA_PATH = Path("/home/nilomr/projects/great-tit-song/data")

PROJECT = Path(DATA_PATH).parent
RAW_DATA = DATA_PATH / "raw" / DATASET_ID
DIRS = ProjDirs(PROJECT, RAW_DATA, mkdir=True)

wav_filepaths, xml_filepaths = [
    get_file_paths(DIRS.RAW_DATA, [ext]) for ext in [".WAV", ".xml"]
]
files_to_segment = get_wavs_w_annotation(wav_filepaths, xml_filepaths)


# %%


wav_outdir, json_outdir = [
    makedir(DIRS.SEGMENTED / ext) for ext in ["WAV", "JSON"]
]


segment_files(
    files_to_segment[:2],
    wav_outdir,
    json_outdir,
    resample=22050,
    parser_func=parse_sonic_visualiser_xml,
    min_duration=0.5,
    min_freqrange=200,
    labels_to_ignore=["NOISE", "FIRST"],
)

# %%

segment_files_parallel(
    files_to_segment[:2],
    DIRS,
    resample=22050,
    parser_func=parse_sonic_visualiser_xml,
    min_duration=0.5,
    min_freqrange=200,
    labels_to_ignore=["NOISE", "FIRST"],
)

# %%
params = Parameters(dereverb=True, verbose=False)
dataset = KantoData(
    DATASET_ID,
    DIRS,
    parameters=params,
    overwrite_dataset=True,
    overwrite_data=True,
    random_subset=10,
)

dataset.vocs.head()
# %%

# storm petrel
DATASET_ID = "STORM-PETREL"
DATA_PATH = Path(pkg_resources.resource_filename("pykanto", "data"))
PROJECT = Path(DATA_PATH).parent
RAW_DATA = DATA_PATH / "raw" / DATASET_ID

DIRS = ProjDirs(PROJECT, RAW_DATA, mkdir=True)

wav_filepaths, xml_filepaths = [
    get_file_paths(DIRS.RAW_DATA, [ext]) for ext in [".wav", ".xml"]
]
files_to_segment = get_wavs_w_annotation(wav_filepaths, xml_filepaths)


segment_files_parallel(
    files_to_segment,
    DIRS,
    resample=22050,
    parser_func=parse_sonic_visualiser_xml,
    min_duration=0.1,
    min_freqrange=100,
    labels_to_ignore=["NOISE"],
)

outfiles = [get_file_paths(DIRS.SEGMENTED, [ext]) for ext in [".wav", ".JSON"]]


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
    top_dB=65,  # top dB to keep
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
dataset = KantoData(
    DATASET_ID,
    DIRS,
    parameters=params,
    overwrite_dataset=True,
    overwrite_data=False,
)

# Segmente into individual units using information from chipper,
# then check a few.
dataset.segment_into_units()

for voc in dataset.vocs.index:
    dataset.plot_segments(voc)


# %%

to_rm = [
    dataset.DIRS.DATASET.parent,
    dataset.DIRS.SEGMENTED / "WAV",
    dataset.DIRS.SEGMENTED / "JSON",
]
for path in to_rm:
    if path.exists():
        shutil.rmtree(str(path))
assert all([f.exists() for f in to_rm]) == False
