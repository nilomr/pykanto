# Code to:
#
# - Read song bout segmentation information from AviaNZ
#   and save .wav files of individual bouts.
#
# - Read syllable segmentation information from Chipper
#
# - Save JSON dictionaries including all pertinent
#   information for each bout.


import gzip
import pickle
import numpy as np
import wavio
import os
import json
import wave
import sys
from tqdm.auto import tqdm
import pathlib2
from pathlib2 import Path
from src.read.paths import safe_makedir


# ---------------------------------------------------------
#   Read song bout segmentation information from AviaNZ
#   and save .wav files of individual bouts.
# ---------------------------------------------------------


def segment_bouts(wavfile, destination, subset=None):
    """Saves segmented bouts (from AviaNZ) to individual .wav files

    Args:

        wavfile (PosixPath): Path to file
        destination (PosixPath): Destination folder. /year/individual is added.
        subset (str, optional): Subset to save, e.g "GRETI_HQ" only. Default = None.
    """
    wavfile_str = str(wavfile)
    datfile = wavfile_str + ".data"

    if Path(datfile).is_file():

        wavobj = wavio.read(str(wavfile))
        sampleRate = wavobj.rate
        data = wavobj.data

        f = open(datfile)
        segments = json.load(f)[1:]  # Remove header
        cnt = 1

        tmpdir = Path(destination / wavfile.parts[-3] / wavfile.parts[-2])
        safe_makedir(tmpdir)  # create output directory

        for seg in segments:

            filename = tmpdir / (
                str(
                    wavfile.parts[-2]
                    + "-"
                    + str(seg[4][0]["species"])
                    + "-"
                    + wavfile.with_suffix("").parts[-1]
                    + "-"
                    + str(cnt)
                    + ".wav"
                )
            )
            cnt += 1

            if not subset:

                s = int(seg[0] * sampleRate)
                e = int(seg[1] * sampleRate)

            elif subset == seg[4][0]["species"]:  # select segments with this label

                s = int((seg[0] - 1) * sampleRate)
                e = int((seg[1] + 1) * sampleRate)

            temp = data[s:e]
            wavio.write(
                str(filename),
                temp.astype("int16"),
                sampleRate,
                scale="dtype-limits",
                sampwidth=2,
            )

    else:
        print(
            """No .data file exists for this .wav
        There might be files with no segmentation information or
        you might have included an unwanted directory"""
        )


def batch_segment_bouts(origin, destination, subset=None):
    """Extracts all sound segments found in a folder/subfolders.

    Based on code by Stephen Marsland, Nirosha Priyadarshani & Julius Juodakis.

    Args:

        origin (PosixPath): folder with raw data to be segmented
        destination (PosixPath): Destination folder. /year/individual is added.
        subset (str, optional): Subset to save, e.g "GREAT-TIT" only. Defaults to None.
    """
    for root, dirs, files in os.walk(str(origin)):

        for wavfile in tqdm(
            files,
            desc="{Reading, trimming and saving song bouts}",
            position=0,
            leave=True,
        ):

            if (
                wavfile.endswith(".wav")
                or wavfile.endswith(".WAV")
                and wavfile + ".data" in files
            ):
                wavfile = Path(root) / wavfile
                segment_bouts(wavfile, destination, subset=subset)


####################################

# 1 - Make function to split wavs to processed data folder
# (you need to create a directory in paths.py for this purpose)
# update: now can subset, ***needs better filenames AND platform-independent path creation

# 2 - get data from the .data file, add coordinates and other information
# and make a nice, tidy .jason file following avgn format


# * parse chipper gzips (see /utils.py)


# ---------------------------------------------------------
#   Read syllable segmentation information from Chipper
# ---------------------------------------------------------


def open_gzip(file):
    """Reads syllable segmentation generated with chipper
    
    Args:
        file (path): path to the .gzip file
    
    Returns:
        list: params, onsets, offsets
    """
    with gzip.open(file, "rb") as f:
        data = f.read()

    song_data = pickle.loads(data, encoding="utf-8")

    return song_data[0], song_data[1]


# ---------------------------------------------------------
#   Save JSON dictionaries including all pertinent
#   information for each bout.
# ---------------------------------------------------------

# file = "/media/nilomr/SONGDATA/interim/2020/W100/SegSyllsOutput_20200407_T191217/SegSyllsOutput_W100-BLUETI-20200327_040000-27.gzip"
# open_gzip(file)[0]['BoutRange']

# json_dict = {}
# json_dict["species"] = "European starling"
# json_dict["common_name"] = "Sturnus vulgaris"

