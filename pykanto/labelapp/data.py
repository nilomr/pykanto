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
from typing import TYPE_CHECKING, Tuple

import numpy as np
from bokeh.models.sources import ColumnDataSource
from PIL import Image, ImageEnhance, ImageOps
from pykanto.signal.spectrogram import (cut_or_pad_spectrogram,
                                        retrieve_spectrogram)
from pykanto.utils.write import makedir

if TYPE_CHECKING:
    from pykanto.dataset import SongDataset

# ──── FUNCTIONS ────────────────────────────────────────────────────────────────


def embeddable_image(
        data: np.ndarray,
        invert: bool = False,
        background: int = 41) -> str:
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
    img_data = np.rot90(np.interp(
        data, (data.min(),
               data.max()),
        (0, 255)).astype(
        np.uint8), 2)
    image = Image.fromarray(img_data, mode='L').resize((64, 64), Image.BICUBIC)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    image = Image.fromarray(
        np.where(np.array(image) < background, background, np.array(image)))
    if invert:
        image = ImageOps.invert(image)
    buffer = BytesIO()
    image.save(buffer, format='png')
    for_encoding = buffer.getvalue()
    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()


def prepare_datasource(
        dataset: SongDataset, individual: str, spec_length: int = 500,
        song_level: bool = False) -> Tuple[str, Path]:
    """
    Prepare and save data source for the interactibe labelling application.

    Args:
        dataset (SongDataset): Source dataset.
        individual (str): ID to process.
        spec_length (int, optional): Desired spectrogram lenght, in frames. 
            Defaults to 500.
        song_level (bool, optional): Whether to use all units per 
            vocalisation or their average. Defaults to False.

    Returns:
        Tuple[str, Path]: A tuple with ID and path to saved data source.
    """

    # Get a subset of the main dataset for this individual
    if song_level:
        df = dataset.vocs[dataset.vocs['ID'] == individual][[
            'ID', 'auto_type_label', 'umap_x', 'umap_y', 'spectrogram_loc']].copy()
        spectrograms = [retrieve_spectrogram(spec_loc)
                        for spec_loc in df['spectrogram_loc']]
    else:
        df = dataset.units[dataset.units['ID'] == individual][[
            'ID', 'auto_type_label', 'umap_x', 'umap_y']].copy()
        units = pickle.load(open(dataset.DIRS.UNITS[individual], "rb"))
        spectrograms = list(itertools.chain.from_iterable(units.values()))

    # Preprocess spectrograms
    spectrograms = [cut_or_pad_spectrogram(
        spec, spec_length) for spec in spectrograms]
    df['spectrogram'] = list(map(embeddable_image, spectrograms))

    out_dir = dataset.DIRS.SPECTROGRAMS / 'bk_data' / (
        f'{individual}_bk_data.p'
        if song_level else f'{individual}_bk_unit_data.p')
    makedir(out_dir)
    pickle.dump(df, open(out_dir, "wb"))
    return (individual, out_dir)


def load_bk_data(
        dataset: SongDataset, dataloc: str, individual: str) -> ColumnDataSource:
    """
    Load saved data source to use in interctive labelling app.

    Args:
        dataset (SongDataset): Source dataset.
        dataloc (str): Type of data to use (one of 'vocalisation_labels', 
            'unit_labels')
        individual (str): ID to process.

    Returns:
        ColumnDataSource: Data ready to plot.
    """
    df_loc = getattr(dataset.DIRS, dataloc.upper())['predatasource'][individual]
    df = pickle.load(open(df_loc, "rb"))
    source = ColumnDataSource(
        df[['ID', 'auto_type_label', 'umap_x', 'umap_y', 'spectrogram']])
    return source
