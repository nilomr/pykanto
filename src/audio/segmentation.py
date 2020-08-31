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
import contextlib
import sys
from tqdm.auto import tqdm
import pathlib2
from pathlib2 import Path
import datetime as dt
from src.read.paths import safe_makedir
import audio_metadata
import re


# ---------------------------------------------------------
#   Read song bout segmentation information from AviaNZ
#   and save .wav files of individual bouts.
# ---------------------------------------------------------

# segment_bouts(wavfile, destination, subset=None, **kwargs):
"""Saves segmented bouts (from AviaNZ) to individual .wav files
Args:
    wavfile (PosixPath): Path to file
    destination (PosixPath): Destination folder. /year/individual is added.
    subset (str, optional): Subset to save, e.g "GRETI_HQ" only. Defaults to None.
"""

### remove
path = Path("/home/nilomr/projects/0.0_great-tit-song/test")
origin = path / "raw" / "2020"
wavfile = origin / "W58" / "20200410_050000.WAV"
destination = path / "interim"
subset = "GRETI_HQ"
threshold = None
###


wavfile_str = str(wavfile)
datfile = wavfile_str + ".data"
datetime = dt.datetime.strptime(wavfile.stem, "%Y%m%d_%H%M%S")

kk = audio_metadata.load(wavfile)["tags"].comment


if Path(datfile).is_file():

    with open(datfile) as dat, wave.open(wavfile_str) as wav:
        segments = json.load(dat)[1:]
        frames = wav.getnframes()
        sampleRate = wav.getframerate()
        data = np.frombuffer(wav.readframes(frames), dtype=np.int16)

    cnt = 1
    tmpdir = Path(destination / wavfile.parts[-3] / wavfile.parts[-2])
    safe_makedir(tmpdir)  # create output directory

    for seg in segments:
        species = seg[4][0]["species"]
        filename = tmpdir / (
            str(
                wavfile.parts[-2]
                + "-"
                + species
                + "-"
                + wavfile.with_suffix("").parts[-1]
                + "-"
                + str(cnt)
                + ".wav"
            )
        )
        cnt += 1

        if not subset:
            save_bout(data, filename, seg, sampleRate, threshold=None)

        elif subset == species:  # select segments with this label
            s = int(seg[0] * sampleRate)
            e = int(seg[1] * sampleRate)
            temp = data[s:e]

            # JSON dictionary to go with .wav file
            # TODO continue populating the JSON and save it; then add bandpass filter

            seg_datetime = datetime + dt.timedelta(seconds=seg[0])
            meta = audio_metadata.load(wavfile)

            s = "Part 1. Part 2. Part 3 then more text"
            re.search(r"Part 1\.(.*?)Part 3", s).group(1)

            tags = audio_metadata.load(wavfile)["tags"].comment[0]
            audiomoth = re.search(r"AudioMoth.(.*?) at gain", tags).group(1)

            json_dict = {}
            json_dict["species"] = species
            json_dict["nestbox"] = wavfile.parts[-2]
            json_dict["recorder"] = audiomoth
            json_dict["recordist"] = "Nilo Merino Recalde"
            json_dict["source_datetime"] = str(datetime)
            json_dict["datetime"] = str(seg_datetime)
            json_dict["date"] = str(seg_datetime.date())
            json_dict["time"] = str(seg_datetime.time())
            json_dict["samplerate_hz"] = sampleRate
            json_dict["length_s"] = len(temp) / sampleRate
            json_dict["upper_freq"] = 
            json_dict["lower_freq"] = 
            json_dict["bit_depth"] = meta["streaminfo"].bit_depth
            json_dict["tech_comment"] = tags
            json_dict["source_location"] = wavfile.as_posix()
            json_dict["wav_location"] = filename.as_posix()


            if not threshold:
                wavio.write(
                    str(filename),
                    temp.astype("int16"),
                    sampleRate,
                    scale="dtype-limits",
                    sampwidth=2,
                )

            elif max(temp) > threshold:
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


#####

# def save_bout(data, filename, seg, sampleRate, threshold=None):


def segment_bouts(wavfile, destination, subset=None, **kwargs):
    """Saves segmented bouts (from AviaNZ) to individual .wav files

    Args:

        wavfile (PosixPath): Path to file
        destination (PosixPath): Destination folder. /year/individual is added.
        subset (str, optional): Subset to save, e.g "GRETI_HQ" only. Defaults to None.
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

            def save_bout(data, filename, seg, sampleRate, threshold=None):

                s = int(seg[0] * sampleRate)
                e = int(seg[1] * sampleRate)
                temp = data[s:e]

                if not threshold:
                    wavio.write(
                        str(filename),
                        temp.astype("int16"),
                        sampleRate,
                        scale="dtype-limits",
                        sampwidth=2,
                    )

                elif max(temp) > threshold:
                    wavio.write(
                        str(filename),
                        temp.astype("int16"),
                        sampleRate,
                        scale="dtype-limits",
                        sampwidth=2,
                    )

            if not subset:
                save_bout(data, filename, seg, sampleRate, **kwargs)

            elif subset == seg[4][0]["species"]:  # select songs with this label
                save_bout(data, filename, seg, sampleRate, **kwargs)

    else:
        print(
            """No .data file exists for this .wav
        There might be files with no segmentation information or
        you might have included an unwanted directory"""
        )


def batch_segment_bouts(origin, destination, subset=None, **kwargs):
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
                segment_bouts(wavfile, destination, subset=subset, **kwargs)


path = Path("/home/nilomr/projects/0.0_great-tit-song/test")
origin = path / "raw" / "2020"
destination = path / "interim"


batch_segment_bouts(origin, destination, subset="GRETI_HQ", threshold=5000)


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

