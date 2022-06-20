# ─── DESCRIPTION ──────────────────────────────────────────────────────────────

"""
A collection of functions and methods used to create directories and write data
to disk.
"""

# ─── DEPENDENCIES ─────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import os
import os.path
import shutil
import sys
import tarfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

import numpy as np
import ujson
from tqdm import tqdm

if TYPE_CHECKING:
    from pykanto.dataset import KantoData

from pykanto.utils.compute import with_pbar

# ──── FUNCTIONS ───────────────────────────────────────────────────────────────


def makedir(DIR: Path, return_path: bool = True) -> Path | None:
    """
    Make a safely nested directory. Returns the Path object by default. Modified
    from code by Tim Sainburg (`source <https://shorturl.at/douA0>`_).

    Args:
        DIR (Path): Path to be created. return_path (bool, optional): Whether to
        return the path. Defaults to True.

    Raises:
        TypeError: Wrong argument type to 'DIR'

    Returns:
        Path: Path to file or directory.
    """

    if not isinstance(DIR, Path):
        raise TypeError(f"Wrong argument type passed to 'DIR': {DIR}")

    # If this is a file
    if len(DIR.suffix) > 0:
        DIR.parent.mkdir(parents=True, exist_ok=True)
    else:
        try:
            DIR.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            if e.errno != 17:
                print("Error:", e)
    if return_path:
        return DIR


def copy_xml_files(file_list: List[Path], dest_dir: Path) -> None:
    """
    Copies a list of files to `dest_dir / file.parent.name / file.name`

    Args:
        file_list (List[Path]): List of files to be copied.
        dest_dir (Path): Path to destination folder, will create it
            if doesn't exist.
    """
    file: Path
    for file in tqdm(
        file_list, desc="Copying files", leave=True, file=sys.stdout
    ):
        dest_file: Path = dest_dir / file.parent.name / file.name
        makedir(dest_file)
        shutil.copyfile(file, dest_file, follow_symlinks=True)
    print(f"Done copying {len(file_list)} files to {dest_dir}")


def save_json(json_object: Dict, json_loc: Path) -> None:
    """
    Saves a .json file using ujson.

    Args:
        json_loc (Path): Path to json file.

    Returns:
        Dict: Json file as a dictionary.
    """
    with open(json_loc, "w", encoding="utf-8") as f:
        ujson.dump(json_object, f, ensure_ascii=False, indent=4)


# Save segmentation and label data to jsons:
def save_to_jsons(dataset: KantoData) -> None:
    """
    Appends new metadata generated in pykanto to the original json metadata
    files that were used to create a `KantoData` dataset. These usually include
    things like type labels and unit oset/offsets.

    Args:
        dataset (KantoData): Dataset object.
    """
    # need to convert numpy to list before can dump to json:
    df = dataset.data.copy()

    # Convert any numpy arrays to lists, pathlib.Paths to str
    arraycols, pathcols = [
        [col for col in df.columns if isinstance(df[col][0], coltype)]
        for coltype in [np.ndarray, Path]
    ]
    df[arraycols] = df[arraycols].applymap(lambda x: x.tolist())

    # Convert any pathlib.Paths to str
    arraycols = [col for col in df.columns if isinstance(df[col][0], Path)]
    df[pathcols] = df[pathcols].applymap(lambda x: x.as_posix())

    # Parallelisable part
    # TODO: refactor and parallelise with ray
    for jsonfile in with_pbar(
        dataset.DIRS.JSON_LIST,
        desc="Adding new metadata to .json files "
        f" in {dataset.DIRS.JSON_LIST[0].parent}",
    ):
        idx = jsonfile.stem

        if idx in dataset.data.index:
            dfdict = df.loc[idx].to_dict()
            with open(jsonfile, "r") as f:
                data = json.load(f)
                for key in dfdict:
                    if key not in data:
                        data[key] = dfdict[key]

            # create randomly named temporary file to avoid
            # interference with other thread/asynchronous request
            tempfile = os.path.join(
                os.path.dirname(jsonfile), str(uuid.uuid4())
            )
            with open(tempfile, "w") as f:
                json.dump(data, f, indent=4)

            # rename temporary file replacing old file
            os.replace(tempfile, jsonfile)


class NumpyEncoder(json.JSONEncoder):
    """
    Stores a numpy.ndarray or any nested-list composition as JSON.
    Source: karlB on `Stack Overflow <https://stackoverflow.com/a/47626762>`_.

    Extends the json.JSONEncoder class.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def make_tarfile(source_dir: Path, output_filename: Path) -> None:
    """
    Makes a tarfile from a given directory.
    Source: George V. Reilly on
    `stack overflow <https://stackoverflow.com/a/17081026>`_.

    Args:
        source_dir (Path): Directory to tar
        output_filename (Path): Name of output file (e.g. file.tar.gz).
    """
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
