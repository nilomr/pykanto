# ─── DESCRIPTION ──────────────────────────────────────────────────────────────

"""
Functions to read external files -e.g. JSON- efficiently.
"""

# ──── IMPORTS ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import os
import os.path
import pickle
import shutil
import sys
import tarfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, ItemsView, List
from matplotlib import cm

import numpy as np
import ray
import ujson
from PIL import Image
from tqdm import tqdm

from pykanto.utils.compute import (
    calc_chunks,
    flatten_list,
    get_chunks,
    print_parallel_info,
    to_iterator,
    with_pbar,
)

if TYPE_CHECKING:
    from pykanto.dataset import KantoData
    from pykanto.utils.paths import ProjDirs


# ─── FUNCTIONS ────────────────────────────────────────────────────────────────


def _relink_kantodata(dataset_dir: Path, path: Path) -> Path:
    index = path.parts.index("spectrograms")
    return dataset_dir.parent.joinpath(*path.parts[index:])


def _relink_df(dataset_dir: Path, x: Any) -> None | Any:
    if isinstance(x, Path):
        return _relink_kantodata(dataset_dir, x)
    else:
        return x


def _path_is_file(x) -> bool | None:
    if isinstance(x, Path):
        return x.is_file()


def load_dataset(
    dataset_dir: Path, DIRS: ProjDirs, relink_data: bool = True
) -> KantoData:
    """
    Load an existing dataset, fixing any broken links to data using a new
    ProjDirs object.

    Args:
        dataset_dir (Path): Path to the dataset file (\*.db)
        DIRS (ProjDirs): New project directories
        relink_data (bool, optional): Whether to make update dataset paths.
            Defaults to True.

    Returns:
        KantoData: The dataset
    """
    # Load
    dataset = pickle.load(open(dataset_dir, "rb"))

    if relink_data:

        # Fix DIRS
        dataset.DIRS = DIRS
        setattr(dataset.DIRS, "DATASET", dataset_dir)
        dataset.DIRS.SPECTROGRAMS = _relink_kantodata(
            dataset_dir, dataset.DIRS.SPECTROGRAMS
        )

        # Update all paths
        dataset.files
        pathcols = [
            col
            for col in dataset.files.columns
            if any(isinstance(x, Path) for x in dataset.files[col])
        ]
        for col in pathcols:
            dataset.files[col] = dataset.files[col].apply(
                lambda x: _relink_df(dataset_dir, x)
            )

        # Check data are reachable
        exist = np.concatenate(
            [
                dataset.files[col].apply(lambda x: _path_is_file(x)).values
                for col in pathcols
            ]
        )
        n_noexist, n_none = [
            np.count_nonzero(exist == v) for v in [False, None]
        ]
        if n_noexist > 0 and n_noexist != len(exist) - n_none:
            raise FileNotFoundError(
                "Could not reconnect all data in the dataset "
                f"({n_noexist}/{len(exist)} failed)."
            )
        elif n_noexist == len(exist) - n_none:
            raise FileNotFoundError(
                "Could not reconnect any data in the dataset."
            )
    return dataset


def read_json(json_loc: Path) -> Dict:
    """
    Reads a .json file using ujson.

    Args:
        json_loc (Path): Path to json file.

    Returns:
        Dict: Json file as a dictionary.
    """
    with open(json_loc) as f:
        return ujson.load(f)


def _get_json(file):
    """
    Reads and returns a .json file with a new 'noise' key with value True if the
    json file has a 'label' key with value 'NOISE'.
    """
    jf = read_json(file)
    try:
        jf["noise"] = True if jf["label"] == "NOISE" else False
    except:
        jf["label"] = False
    return jf


@ray.remote
def _get_json_r(files: List[Path]) -> List[Dict[str, Any]]:
    return [_get_json(file) for file in files]


def _get_json_parallel(
    lst: List[Path], verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Parallel implementation of
    :func:`~pykanto.utils.io._get_json`.
    """
    # Calculate and make chunks
    n = len(lst)
    chunk_info = calc_chunks(n, verbose=verbose)
    chunk_length, n_chunks = chunk_info[3], chunk_info[2]
    chunks = get_chunks(lst, chunk_length)
    print_parallel_info(n, "JSON files", n_chunks, chunk_length)

    # Distribute with ray
    obj_ids = [_get_json_r.remote(i) for i in chunks]
    pbar = {"desc": "Loading JSON files", "total": n_chunks}
    jsons = [obj_id for obj_id in with_pbar(to_iterator(obj_ids), **pbar)]

    # Flatten and return
    return flatten_list(jsons)


# ──── WRITE ────────────────────────────────────────────────────────────────────


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
        dataset._jsonfiles,
        desc="Adding new metadata to .json files "
        f" in {dataset._jsonfiles[0].parent}",
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
    Source: `karlB on Stack Overflow <https://stackoverflow.com/a/47626762>`_.

    Extends the json.JSONEncoder class.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def make_tarfile(source_dir: Path, output_filename: Path) -> None:
    """
    Makes a tarfile from a given directory.
    Source: ` George V. Reilly on stack overflow <https://stackoverflow.com/a/17081026>`_.

    Args:
        source_dir (Path): Directory to tar
        output_filename (Path): Name of output file (e.g. file.tar.gz).
    """
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def get_unit_spectrograms(dataset: KantoData, ID: str) -> Dict[str, np.ndarray]:
    """
    Retrieves unit (e.g. individual notes) spectrograms for a grouping ID in a
        dataset.

    Args:
        dataset (KantoData): Dataset to use.
        ID (str): Which id to use (present in an ID column in the dataset)

    Returns:
        Dict[str, np.ndarray]: A dictionary of spectrograms, keyed by
            vocalisation index.
    Example:
        >>> units = get_unit_spectrograms(dataset, "BIGBIRD")
        >>> last_note = units["BIGBIRD_0"][-1]
    """
    units = pickle.load(
        open(
            dataset.files.query("ID==@ID")[
                "average_units" if dataset.parameters.song_level else "units"
            ][0],
            "rb",
        )
    )
    return units


def save_songs(folder: Path, specs: List[Path]) -> None:
    """
    Save song spectrograms as .jpg images to folder.

    Args:
        folder (Path): Path to destination folder.
        specs (List[Path]): List of spectrogram paths.
    """
    folder.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        img = np.load(spec)
        img *= 255.0 / (img + img.min()).max()
        img = np.invert(np.flipud(np.floor(img).astype(int))) + 256
        img = Image.fromarray(np.uint8(cm.magma(img) * 255)).convert("RGB")
        img.save(folder / f"{spec.stem}.jpg")


def save_subset(
    train_dir: Path,
    test_dir: Path,
    dname: str,
    to_export: ItemsView[str, List[Path]],
) -> None:
    """
    Save train and test subsets of dataset to disk as .jpg images (in folders
    correspoding to class labels).

    Args:
        train_dir (Path): Destination folder for training data.
        test_dir (Path): Destination folder for test data.
        dname (str): Name of subset, one of "train" or "test".
        to_export (ItemsView[str, List[Path]]): Subset of dataset to export.
    """
    for song_class, specs in with_pbar(to_export, total=len(to_export)):
        folder = (train_dir if dname == "train" else test_dir) / song_class
        save_songs(folder, specs)
