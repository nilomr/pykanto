import glob
import gzip
import json
import os
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from xml.etree import ElementTree

import numpy as np
import pandas as pd
from pykanto.utils.compute import timing, with_pbar
from pykanto.utils.io import read_json
from pykanto.utils.paths import get_file_paths
from pykanto.utils.types import SegmentAnnotation
from tqdm import tqdm

# ──── METADATA FILE PARSERS ────────────────────────────────────────────────────


def parse_sonic_visualiser_xml(xml_filepath: Path) -> SegmentAnnotation:
    """
    Parses an xml annotation file generated by Sonic Visualiser and returns
    a SegmentAnnotation.

    Note:
        The individual ID string = folder name + file name + segment index.

    Args:
        xml_filepath (Path): Path to xml file.

    Returns:
        SegmentAnnotation: Object with relevant metadata.
    """
    # Parse xml
    root = ElementTree.parse(xml_filepath).getroot()

    # Data to extract from xml
    fields = ["frame", "duration", "value", "extent"]

    # Retrieve segment information
    seq_list = []
    for _, segment in enumerate(root.findall("data/dataset/point")):
        # Extract relevant information from xml file
        seg_info = [int(float(segment.get(value))) for value in fields]
        label: List[str] = [str(segment.get("label"))]
        seq_list.append(seg_info + label)

    # Organise in a dictionary
    starts, durations, lower_freqs, freq_extents, labels = zip(*seq_list)

    sa = SegmentAnnotation(
        ID=str(xml_filepath.parent.name),
        # ID=[f"{parname}_{filename}_{i}" for i in range(len(starts))],
        start_times=list(starts),
        durations=list(durations),
        end_times=(np.asarray(starts) + np.asarray(durations)).tolist(),
        lower_freq=list(lower_freqs),
        upper_freq=(
            np.asarray(lower_freqs) + np.asarray(freq_extents)
        ).tolist(),
        label=list(labels),
        annotation_file=xml_filepath,
    )

    return sa


# ──── OTHERS ───────────────────────────────────────────────────────────────────


def open_gzip(file: Path) -> Tuple[Dict[str, Any], Dict[str, List[int]], float]:
    """
    Reads syllable segmentation generated
    with `Chipper <https://github.com/CreanzaLab/chipper>`_.

    Args:
        file (Path): Path to the .gzip file.

    Returns:
        Tuple[Dict[str, Any], Dict[str, List[int]], float]: Tuple containing
        two dictionaries (the first contains chipper parameters, the second
        has two keys ['Onsets', 'Offsets']) and a parameter 'timeAxisConversion'.
    """
    with gzip.open(file, "rb") as f:
        data = f.read()
    song_data = pickle.loads(data, encoding="utf-8")

    return song_data[0], song_data[1], song_data[3]["timeAxisConversion"]


@timing
def chipper_units_to_json(
    directory: Path,
    n_fft: int = 1024,
    overlap: int = 1010,
    pad: int = 150,
    window_offset: bool = True,
    overwrite_json: bool = False,
    pbar: bool = True,
):
    """
    Reads audio unit segmentation metadata from .gzip files output by Chipper and appends them to pykanto .JSON metadata files.

    Args:
        directory (Path): _description_
        n_fft (int, optional): _description_. Defaults to 1024.
        overlap (int, optional): _description_. Defaults to 1010.
        pad (int, optional): _description_. Defaults to 150.
        window_offset (bool, optional): _description_. Defaults to True.
        overwrite_json (bool, optional): _description_. Defaults to False.
        pbar (bool, optional): _description_. Defaults to True.

    Raises:
        FileExistsError: _description_
    """

    woffset: int = n_fft // 2 if window_offset else 0

    jsons, gzips = [
        get_file_paths(directory, ext)
        for ext in ([".json", ".JSON"], [".gzip", ".GZIP"])
    ]

    jsons = {path.stem: path for path in jsons}
    gzips = {path.stem.replace("SegSyllsOutput_", ""): path for path in gzips}

    if len([gzip for gzip in gzips if gzip in jsons]) == 0:
        raise KeyError("No JSON and GZIP file names match")

    for gz_name, gz_path in with_pbar(
        gzips.items(),
        desc="Adding unit onset/offset information "
        "from .gzip to .json files",
        disable=False if pbar else True,
    ):

        if gz_name in jsons:

            jsondict = read_json(jsons[gz_name])
            if "onsets" in jsondict and not overwrite_json:
                raise FileExistsError(
                    "Json files already contain unit onset/offset times."
                    "Set `overwrite_json = True` if you want "
                    "to overwrite them."
                )

            sr = jsondict["sample_rate"]
            gzip_onoff = open_gzip(gz_path)[1]
            on, off = np.array(gzip_onoff["Onsets"]), np.array(
                gzip_onoff["Offsets"]
            )

            jsondict["onsets"], jsondict["offsets"] = [
                (((arr - pad) * (n_fft - overlap) + woffset) / sr).tolist()
                for arr in (on, off)
            ]

            with open(jsons[gz_name].as_posix(), "w") as f:
                json.dump(jsondict, f, indent=2)
