# ─── DESCRIPTION ──────────────────────────────────────────────────────────────

"""
A collection of functions and decorators used to manage common operations (e.g.,
parallelisation, timing).
"""

# ─── DEPENDENCIES ─────────────────────────────────────────────────────────────

from __future__ import annotations

import gc
import itertools
import sys
from collections import ChainMap
from functools import wraps
from time import time
from typing import Any, Dict, Iterable, Iterator, List, TYPE_CHECKING

import numpy as np
import pandas as pd
import psutil
import ray
from pykanto.utils.types import Chunkinfo
from tqdm.auto import tqdm


# ──── FUNCTIONS ───────────────────────────────────────────────────────────────


def to_iterator(obj_ids, breaks: bool = True):
    """
    Use ray paralell with a progress bar.
    Modified from https://git.io/JMv6r.
    """
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        if breaks:
            yield ray.get(done[0])
        else:
            try:
                yield ray.get(done[0])
            except Exception as e:
                print(f"{done[0]} failed: {e}")
        del done[0]
        gc.collect()


def with_pbar(
    iterable: Iterable[Any], desc: str | None = None, **kwargs
) -> tqdm:
    """
    Returns a custom progress bar. This is just a wrapper around tqdm.

    Args:
        iterable (Iterable[Any]): Object to iterate on.
        desc (str, optional): Description of what's going on. Defaults to None.

    Returns:
        tqdm: progress bar.
    """
    return tqdm(
        iterable, desc=desc, leave=True, position=0, file=sys.stdout, **kwargs
    )


def print_dict(dictionary: Dict) -> str:
    """
    Pretty print a class __dict__ attribute.

    Args:
        dictionary (Dict): __dict__ attribute
            containing an object's writable attributes.

    Returns:
        str: Dictionary contents in a legible way.
    """
    items = []
    for k, v in dictionary.items():
        if isinstance(v, dict):
            v1 = dict([list(v.items())[0]])
            if isinstance([list(v1.items())[0][1]][0], dict):
                v1 = {
                    k: f"{dict([list(v.items())[0]])} + {len(v)-1} other entries"
                    for k, v in v1.items()
                }
            items.append(f"{k}: {v1} + {len(v)-1} other entries.")
        elif isinstance(v, pd.DataFrame):
            items.append(f"{k}: pd.DataFrame with {len(v)} entries")
        elif isinstance(v, (list, np.ndarray)):
            items.append(f"{k}: {v[:1]} + {len(v)-1} other entries.")
        else:
            items.append(f"{k}: {v}")
    nl = "\n"
    return f"{nl}{nl}Items held:{nl}{nl}{nl.join(items)}"


def timing(f):
    """
    Custom timer decorator. Prints time info unless used within a KantoData
    where parameters.verbose = False.
    """

    @wraps(f)
    def wrap(*args, **kwargs):
        start = time()
        output = f(*args, **kwargs)
        end = time()
        from pykanto.dataset import KantoData

        verbose = (
            args[0].parameters.verbose
            if isinstance(args[0], KantoData)
            else True
        )
        if verbose:
            print(f"Function '{f.__name__}' took {end-start:2.4f} sec.")
        return output

    return wrap


def calc_chunks(
    len_iterable: int,
    factor: int = 2,
    n_workers: float | None = None,
    verbose: bool = False,
) -> Chunkinfo:
    """
    Calculate chunk size to optimise parallel computing.
    Adapted from https://stackoverflow.com/a/54032744.

    returns ['n_workers', 'len_iterable', 'n_chunks',
                      'chunksize', 'last_chunk']

    """
    if not n_workers:
        try:
            n_workers = len(psutil.Process().cpu_affinity())
        except:
            # cpu_affinity doesn't work on macOS nilomr/pykanto/#18
            # TODO: #21 @nilomr optionally pass n_workers from methods using calc_chunks
            n_workers = psutil.cpu_count(logical=False)

    chunksize, extra = divmod(len_iterable, n_workers * factor)
    if extra:
        chunksize += 1
    # `+ (len_iterable % chunksize > 0)` exploits that `True == 1`
    n_chunks = len_iterable // chunksize + (len_iterable % chunksize > 0)
    # exploit `0 == False`
    last_chunk = len_iterable % chunksize or chunksize

    chunks = Chunkinfo(n_workers, len_iterable, n_chunks, chunksize, last_chunk)
    if verbose:
        print(chunks)
    return chunks


def get_chunks(lst: List[Any], n: int) -> Iterator[Any]:
    """
    Yields successive n-sized chunks from list.
    Last chunk will be shorter if len(lst) % n != 0.

    Args:
        lst (List[Any]): List to return chunks from.
        n (int): Number of chunks

    Yields:
        Iterator[Any]: n-sized chunks.
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def flatten_list(lst: List[Any]) -> List[Any]:
    """
    Flattens a list using chain.
    """
    return list(itertools.chain(*lst))


def dictlist_to_dict(dictlist: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Builds a dictionary from a list of dictionaries.

    Args:
        dictlist (List[Dict[str, Any]]): List of dictionaries.
    Returns:
        Dict[str, Any]: Dictionary with items == list elements.
    """
    return dict(ChainMap(*dictlist))


def print_parallel_info(
    n: int, iterable_name: str, n_chunks: int, chunk_length: int
) -> None:
    """
    Prints information about a parallel process for the user.

    Args:
        n (int): Total length of iterable.
        iterable_name (str): Description of iterable.
        n_chunks (int): Number of chunks that will be used.
        chunk_length (int): Length of each chunk.
    """
    print(
        f"Found {n} {iterable_name}. "
        f"They will be processed in {n_chunks} "
        f"chunks of length {chunk_length}."
    )
