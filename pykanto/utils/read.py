# ─── DESCRIPTION ──────────────────────────────────────────────────────────────

"""
Functions to read external files -e.g. JSON- efficiently.
"""

# ──── IMPORTS ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List
import numpy as np

import ray
import ujson as json

if TYPE_CHECKING:
    from pykanto.dataset import KantoData
    from pykanto.utils.paths import ProjDirs

from pykanto.utils.compute import (
    calc_chunks,
    flatten_list,
    get_chunks,
    print_parallel_info,
    to_iterator,
    with_pbar,
)

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
        dataset_dir (Path): Path to the dataset file (*.db)
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
        return json.load(f)


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
    :func:`~pykanto.utils.read._get_json`.
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
