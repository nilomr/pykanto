# ─── DESCRIPTION ─────────────────────────────────────────────────────────────

"""
Code to generate embeddable data to be used in the interactive song labelling
web application.
"""

# ──── IMPORTS ──────────────────────────────────────────────────────────────────
from __future__ import annotations

import base64
import itertools
import pickle
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import ray
from bokeh.models.sources import ColumnDataSource
from PIL import Image, ImageEnhance, ImageOps
from pykanto.signal.spectrogram import (
    cut_or_pad_spectrogram,
    retrieve_spectrogram,
)
from pykanto.utils.compute import (
    calc_chunks,
    get_chunks,
    print_parallel_info,
    to_iterator,
    with_pbar,
)
from pykanto.utils.io import makedir

if TYPE_CHECKING:
    from pykanto.dataset import KantoData

# ──── FUNCTIONS ────────────────────────────────────────────────────────────────


def embeddable_image(
    data: np.ndarray, invert: bool = False, background: int = 41
) -> str:
    """
    Save a base 64 png from a np.ndarray.
    Source: `Leland McInnes, 2018
    <https://umap-learn.readthedocs.io/en/latest/basic_usage.html>`_.

    Args:
        data (np.ndarray): Image to embed.
        invert (bool, optional): Whether to invert image. Defaults to True.
        background (int, optional): RGB grey value. Defaults to 41 (same as app).

    Returns:
        str: A decoded png image.
    """
    img_data = np.flip(
        np.interp(data, (data.min(), data.max()), (0, 255)).astype(np.uint8)
    )
    image = Image.fromarray(img_data, mode="L").resize((64, 64), Image.BICUBIC)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    image = Image.fromarray(
        np.where(np.array(image) < background, background, np.array(image))
    )
    if invert:
        image = ImageOps.invert(image)
    buffer = BytesIO()
    image.save(buffer, format="png")
    for_encoding = buffer.getvalue()
    return "data:image/png;base64," + base64.b64encode(for_encoding).decode()


def prepare_datasource(
    dataset: KantoData,
    ID: str,
    spec_length: int = 500,
    song_level: bool = False,
) -> Tuple[str, Path]:
    """
    Prepare and save data source for the interactibe labelling application.

    Args:
        dataset (KantoData): Source dataset.
        ID (str): ID to process.
        spec_length (int, optional): Desired spectrogram lenght, in frames.
            Defaults to 500.
        song_level (bool, optional): Whether to use all units per
            vocalisation or their average. Defaults to False.

    Returns:
        Tuple[str, Path]: A tuple with ID and path to saved data source.
    """

    # Get a subset of the main dataset for this individual
    if song_level:
        df = dataset.data.query("ID==@ID")[
            ["ID", "auto_class", "umap_x", "umap_y"]
        ].copy()

        df["spectrogram"] = dataset.files.loc[df.index, "spectrogram"]

        spectrograms = [
            retrieve_spectrogram(spec_loc) for spec_loc in df["spectrogram"]
        ]
    else:
        df = dataset.units.query("ID==@ID")[
            ["ID", "auto_class", "umap_x", "umap_y"]
        ].copy()

        units = pickle.load(
            open(dataset.files.query("ID==@ID")["units"][0], "rb")
        )
        spectrograms = list(itertools.chain.from_iterable(units.values()))

    # Preprocess spectrograms
    spectrograms = [
        cut_or_pad_spectrogram(spec, spec_length) for spec in spectrograms
    ]
    df["spectrogram"] = list(map(embeddable_image, spectrograms))

    out_dir = (
        dataset.DIRS.SPECTROGRAMS
        / "app_data"
        / (f"{ID}_app_data.p" if song_level else f"{ID}_app_unit_data.p")
    )
    makedir(out_dir)
    pickle.dump(df, open(out_dir, "wb"))
    return (ID, out_dir)


def prepare_datasource_parallel(
    dataset: KantoData,
    spec_length: float | None = None,
    song_level: bool = False,
    num_cpus: float | None = None,
) -> List[List[Tuple[str, Path]]]:
    """
    Prepare and save data sources for the interactibe labelling application (parallel).

    Args:
        dataset (KantoData): Source dataset.
        spec_length (float | None, optional): . Defaults to None.
        song_level (bool, optional): _description_. Defaults to False.
        num_cpus (float | None, optional): N cpus to use. Defaults to None.

        dataset (KantoData): Source dataset.
        spec_length (float | None, optional): Desired spectrogram lenght, in
            frames. Defaults to 500.
        song_level (bool, optional): Whether to use all units per vocalisation
            or their average. Defaults to False.
        num_cpus (float | None, optional):
            N cpus to use. Defaults to None.

    Returns:
        List[List[Tuple[str, Path]]]: A list of lists of tuples
            with ID and path to saved data source.
    """

    # Set or get spectrogram length (affects width of unit/voc preview)
    # NOTE: this is set to maxlen=50%len if using units, 4*maxlen if using
    # entire vocalisations. This might not be adequate in all cases.
    if not spec_length:
        max_l = max(
            [i for array in dataset.data.unit_durations.values for i in array]
        )
        spec_length = (max_l + 0.7 * max_l) if not song_level else 6 * max_l

    spec_length_frames = int(
        np.floor(
            spec_length
            * dataset.parameters.sr
            / dataset.parameters.hop_length_ms
        )
    )
    # Prepare dataframes with spectrogram pngs
    # Calculate and make chunks
    datatype = "data" if song_level else "units"
    IDS = np.unique(
        getattr(dataset, datatype).dropna(subset=["auto_class"])["ID"]
    )
    n = len(IDS)
    chunk_info = calc_chunks(
        n, n_workers=num_cpus, verbose=dataset.parameters.verbose
    )
    chunk_length, n_chunks = chunk_info[3], chunk_info[2]
    chunks = get_chunks(list(IDS), chunk_length)
    print_parallel_info(n, "individual IDs", n_chunks, chunk_length)

    # Distribute with ray
    dataset_ref = ray.put(dataset)

    @ray.remote(num_cpus=num_cpus)
    def prepare_datasource_r(
        dataset: KantoData,
        IDS: List[str],
        spec_length: int = 500,
        song_level: bool = False,
    ):
        return [
            prepare_datasource(
                dataset,
                ID,
                spec_length=spec_length,
                song_level=song_level,
            )
            for ID in IDS
        ]

    obj_ids = [
        prepare_datasource_r.remote(
            dataset_ref,
            i,
            spec_length=spec_length_frames,
            song_level=song_level,
        )
        for i in chunks
    ]
    pbar = {
        "desc": "Prepare interactive visualisation",
        "total": n_chunks,
    }
    dt = [obj_id for obj_id in with_pbar(to_iterator(obj_ids), **pbar)]
    return dt


def load_app_data(
    dataset: KantoData, datatype: str, ID: str
) -> ColumnDataSource:
    """
    Load saved data source to use in interactive labelling app.

    Args:
        dataset (KantoData): Source dataset.
        datatype (str): Type of data to use (one of 'voc_app_data',
            'unit_app_data')
        ID (str): ID to process.

    Returns:
        ColumnDataSource: Data ready to plot.
    """
    df_loc = dataset.files.query("ID==@ID")[datatype][0]
    df = pickle.load(open(df_loc, "rb"))
    source = ColumnDataSource(
        df[["ID", "auto_class", "umap_x", "umap_y", "spectrogram"]]
    )
    return source
