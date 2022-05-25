# ─── DESCRIPTION ──────────────────────────────────────────────────────────────

"""
Functions to read external files -e.g. JSON- efficiently.
"""

# ──── IMPORTS ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

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


def load_dataset(
    dataset_dir: Path, DIRS: ProjDirs, relink_data: bool = True
) -> KantoData:
    """
    Load an existing dataset. NOTE: temporaty fix.

    Args:
        dataset_dir (Path): Path to the dataset file (*.db)
        DIRS (ProjDirs): New project directories
        relink_data (bool, optional): Whether to make update dataset paths.
            Defaults to True.

    Raises:
        FileNotFoundError: _description_

    Returns:
        KantoData: _description_
    """

    def relink_kantodata(dataset_location: Path, path: Path):
        index = path.parts.index("spectrograms")
        return dataset_location.parent.joinpath(*path.parts[index:])

    dataset = pickle.load(open(dataset_dir, "rb"))
    if relink_data:

        # Update ProjDirs section
        for k, v in dataset.DIRS.__dict__.items():
            if k in DIRS.__dict__:
                setattr(dataset.DIRS, k, getattr(DIRS, k))

        # Update dataset location
        setattr(dataset.DIRS, "DATASET", dataset_dir)

        if not dataset.vocs["spectrogram_loc"][0].is_file():
            dataset.vocs["spectrogram_loc"] = dataset.vocs[
                "spectrogram_loc"
            ].apply(lambda x: relink_kantodata(dataset_dir, x))
        if not dataset.vocs["spectrogram_loc"][0].is_file():
            raise FileNotFoundError("Failed to reconnect spectrogram data")

        for k, v in dataset.DIRS.__dict__.items():
            if k in [
                "SPECTROGRAMS",
                "UNITS",
                "UNIT_LABELS",
                "AVG_UNITS",
                "VOCALISATION_LABELS",
            ]:
                if isinstance(v, Path):
                    dataset.DIRS.__dict__[k] = relink_kantodata(dataset_dir, v)
                elif isinstance(v, list):
                    dataset.DIRS.__dict__[k] = [
                        relink_kantodata(dataset_dir, path) for path in v
                    ]
                elif isinstance(v, dict):
                    for k1, v1 in v.items():  # Level 1
                        if isinstance(v1, Path):
                            dataset.DIRS.__dict__[k][k1] = relink_kantodata(
                                dataset_dir, v1
                            )
                        elif isinstance(v1, dict):
                            for k2, v2 in v1.items():
                                dataset.DIRS.__dict__[k][k1][
                                    k2
                                ] = relink_kantodata(dataset_dir, v2)
                        elif k1 == "already_checked":
                            continue
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
    Returns a json file with a 'label' field,
    with value 'NOISE' if this field was present and
    its value was 'NOISE' and 'VOCALISATION' in all other cases.
    """
    jf = read_json(file)
    try:
        jf["label"] = jf["label"] if jf["label"] == "NOISE" else "VOCALISATION"
    except:
        jf["label"] = "VOCALISATION"
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
