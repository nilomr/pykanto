# ─── DESCRIPTION ──────────────────────────────────────────────────────────────

"""Perform dimensionality reduction and clustering."""

# ─── DEPENDENCIES ─────────────────────────────────────────────────────────────

from __future__ import annotations

import itertools
import pickle
import warnings
from logging import warn
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import pandas as pd
import psutil
import ray
import umap
from hdbscan import HDBSCAN
from pykanto.signal.spectrogram import pad_spectrogram
from pykanto.utils.compute import (
    calc_chunks,
    flatten_list,
    get_chunks,
    print_parallel_info,
    to_iterator,
    with_pbar,
)
from umap import UMAP

if TYPE_CHECKING:
    from pykanto.dataset import KantoData
try:
    from cuml import UMAP as cumlUMAP
except:
    warnings.warn(
        "The cuML library is not not available: defaulting to "
        "slower CPU UMAP implementation",
        ImportWarning,
    )
    _has_cuml = False
else:
    _has_cuml = True


# ──── FUNCTIONS ───────────────────────────────────────────────────────────────


def umap_reduce(
    data: np.ndarray,
    n_neighbors: int = 15,
    n_components: int = 2,
    min_dist: float = 0.1,
    verbose: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, umap.UMAP]:
    """
    Uniform Manifold Approximation and Projection.
    Uses either the cuml GPU-accelerated version or the 'regular' umap version.
    See the documentation of either for valid kwargs.

    Args:
        data (array-like, shape = (n_samples, n_features)): Data to reduce.
        n_neighbors (int, optional): [description]. Defaults to 15.
        n_components (int, optional): [description]. Defaults to 2.
        min_dist (float, optional): [description]. Defaults to 0.1.
        kwargs: Passed to umap.UMAP or cuml.umap.UMAP.

    Returns:
        Tuple[np.ndarray, umap.UMAP]: Embedding coordinates
        and UMAP reducer.
    """
    if _has_cuml:
        reducer = cumlUMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            **kwargs,
        )
        embedding = reducer.fit_transform(data)
    else:
        if verbose:
            warnings.warn(
                "The cuML library is not not available: defaulting to "
                "slower CPU UMAP implementation."
            )
        reducer = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            **kwargs,
        )
        embedding = reducer.fit_transform(data)
    return embedding, reducer


def hdbscan_cluster(
    embedding: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: None | int = None,
    **kwargs,
) -> HDBSCAN:
    """
    Perform HDBSCAN clustering from vector array or distance matrix.
    Convenience wrapper. See the
    `HDBSCAN* docs <https://hdbscan.readthedocs.io/en/latest/parameter_selection.html>`_.

    Args:
        embedding (np.ndarray): Data to cluster. See hdbscan documentation
            for more.
        min_cluster_size (int, optional): Minimum number of samples to
            consider a cluster. Defaults to 5.
        min_samples (int, optional): Controls how 'conservative' clustering is.
            Larger values = more points will be declared as noise.
            Defaults to None.
        kwargs: Passed to HDBSCAN.

    Returns:
        HDBSCAN: HDBSCAN object. Labels are at `self.labels_`.
    """
    if min_cluster_size < 2:
        warnings.warn("`min_cluster_size` too small, setting it to 2")
        min_cluster_size = 2
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
        **kwargs,
    )
    clusterer.fit(embedding)
    return clusterer


def reduce_and_cluster(
    dataset: KantoData, ID: str, song_level: bool = False, min_sample: int = 10
) -> pd.DataFrame | None:
    # TODO: pass UMAP and HDBSCAN params!
    """
    Args:
        dataset (KantoData): Data to be used.
        ID (str): Grouping factor.
        song_level (bool, optional): Whether to use the average of all units in
            each vocalisation instead of all units. Defaults to False.
        min_sample (int, optional): Minimum number of vocalisations or units.
            Defaults to 10.

    Returns:
        pd.DataFrame | None: Dataframe with columns ['vocalisation_key', 'ID',
            'idx', 'umap_x', 'umap_y', 'auto_class'] or None if sample size
            is too small
    """

    # Retrieve units or averaged units for one individual
    units = pickle.load(
        open(
            dataset.files.query("ID==@ID")[
                "average_units" if song_level else "units"
            ][0],
            "rb",
        )
    )

    units_keys = (
        [key for key in units]
        if song_level
        else [
            key + f"_{i}"
            for key, vals in units.items()
            for i in range(len(vals))
        ]
    )

    # Check if sample size is sufficient
    if len(units_keys) < min_sample:
        warnings.warn(f"Insufficient sample size for {ID}. " "Returning None.")
        return

    # Pad if necessary # REVIEW
    if song_level:
        if not {unit.shape[1] for unit in units.values()} == 1:
            max_frames = max([unit.shape[1] for unit in units.values()])
            units = {
                key: pad_spectrogram(spec, max_frames)
                for key, spec in units.items()
            }
    else:
        if not {unit.shape[1] for ls in units.values() for unit in ls} == 1:
            max_frames = max(
                [unit.shape[1] for ls in units.values() for unit in ls]
            )
            units = {
                key: [pad_spectrogram(spec, max_frames) for spec in ls]
                for key, ls in units.items()
            }

    # Flatten units
    if song_level:
        flat_units = [unit.flatten() for unit in units.values()]
    else:
        unitkeys = list(
            itertools.chain.from_iterable(
                [[key] * len(value) for key, value in units.items()]
            )
        )
        flat_units = np.array(
            [unit.flatten() for ls in units.values() for unit in ls]
        )

    # Run UMAP
    embedding, _ = umap_reduce(
        flat_units, n_neighbors=25, min_dist=0.2, n_components=2
    )

    # Cluster using HDBSCAN
    # smallest cluster size allowed
    clusterer = hdbscan_cluster(
        embedding, min_cluster_size=int(len(flat_units) * 0.02), min_samples=10
    )

    # Put together in a dataframe
    cluster_df = pd.DataFrame(units_keys, columns=["index"])
    cluster_df.set_index("index", inplace=True)
    if song_level is False:
        cluster_df["vocalisation_key"] = unitkeys
        cluster_df["ID"] = ID
        cluster_df["idx"] = [idx[-1] for idx in cluster_df.index]
    cluster_df["umap_x"] = embedding[:, 0]
    cluster_df["umap_y"] = embedding[:, 1]
    cluster_df["auto_class"] = list(clusterer.labels_)
    cluster_df["auto_class"] = cluster_df["auto_class"].astype(str)

    return cluster_df


def reduce_and_cluster_parallel(
    dataset: KantoData, min_sample: int = 10, num_cpus: float | None = None
) -> pd.DataFrame | None:
    """
    Parallel implementation of
    :func:`~pykanto.signal.cluster.reduce_and_cluster`.
    """
    song_level = dataset.parameters.song_level
    IDS = set(dataset.files["ID"])
    # IDS = dataset.DIRS.AVG_UNITS if song_level else dataset.DIRS.UNITS

    # Calculate and make chunks
    n = len(IDS)
    if not n:
        raise KeyError(
            "No sound file keys were passed to " "reduce_and_cluster."
        )
    chunk_info = calc_chunks(
        n, n_workers=num_cpus, verbose=dataset.parameters.verbose
    )
    chunk_length, n_chunks = chunk_info[3], chunk_info[2]
    chunks = get_chunks(list(IDS), chunk_length)
    print_parallel_info(n, "individual IDs", n_chunks, chunk_length)

    # Copy dataset to local object store
    dataset_ref = ray.put(dataset)

    # Distribute with ray
    @ray.remote(
        num_cpus=num_cpus, num_gpus=1 / psutil.cpu_count() if _has_cuml else 0
    )  # type: ignore
    def _reduce_and_cluster_r(
        dataset: KantoData,
        IDS: List[str],
        song_level: bool = False,
        min_sample: int = 10,
    ) -> List[pd.DataFrame | None]:
        return [
            reduce_and_cluster(
                dataset, ID, song_level=song_level, min_sample=min_sample
            )
            for ID in IDS
        ]

    obj_ids = [
        _reduce_and_cluster_r.remote(
            dataset_ref, i, song_level=song_level, min_sample=min_sample
        )
        for i in chunks
    ]
    pbar = {
        "desc": "Projecting and clustering vocalisations",
        "total": n_chunks,
    }
    dfls = [obj_id for obj_id in with_pbar(to_iterator(obj_ids), **pbar)]

    # Minimally check output
    try:
        df = pd.concat(flatten_list(dfls))
        return df
    except ValueError as e:
        if str(e) == "All objects passed were None":
            raise TypeError(
                f"{str(e)}. Possible reasons include:"
                "\n1: Sample sizes were insufficient for each and "
                "every individual in the dataset."
                "\n2: Inadequate parametrisation."
            )
