
# ─── DESCRIPTION ──────────────────────────────────────────────────────────────

"""
Functions to read external files -e.g. JSON- efficiently.
"""

# ──── IMPORTS ─────────────────────────────────────────────────────────────────

from pathlib import Path
from typing import Any, Dict, List
import ray
import ujson as json
from pykanto.utils.compute import (calc_chunks, flatten_list, get_chunks,
                                   print_parallel_info, to_iterator, tqdmm)

# ─── FUNCTIONS ────────────────────────────────────────────────────────────────


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
        jf["label"] = (jf["label"] if jf["label"] ==
                       'NOISE' else 'VOCALISATION')
    except:
        jf["label"] = 'VOCALISATION'
    return jf


@ray.remote
def _get_json_r(files: List[Path]) -> List[Dict[str, Any]]:
    return [_get_json(file) for file in files]


def _get_json_parallel(lst: List[Path],
                       verbose: bool = False
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
    print_parallel_info(n, 'JSON files', n_chunks, chunk_length)

    # Distribute with ray
    obj_ids = [_get_json_r.remote(i) for i in chunks]
    pbar = {'desc': "Loading JSON files", 'total': n_chunks}
    jsons = [obj_id for obj_id in tqdmm(to_iterator(obj_ids), **pbar)]

    # Flatten and return
    return flatten_list(jsons)
