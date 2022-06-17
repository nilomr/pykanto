#%% imports

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Tuple

import librosa.display
import numpy as np
import pkg_resources
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
from matplotlib.collections import PathCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pykanto.dataset import KantoData
from pykanto.parameters import Parameters
from pykanto.plot import segmentation
from pykanto.signal.spectrogram import (
    cut_or_pad_spectrogram,
    retrieve_spectrogram,
)
from pykanto.utils.paths import ProjDirs
from pykanto.utils.read import load_dataset

#%% Setup
DATASET_ID = "GREAT_TIT"
DATA_PATH = Path(pkg_resources.resource_filename("pykanto", "data"))
PROJECT = Path(DATA_PATH).parent
RAW_DATA = DATA_PATH / "segmented" / "great_tit"
DIRS = ProjDirs(PROJECT, RAW_DATA, mkdir=True)

params = Parameters(dereverb=True, verbose=False)
dataset = KantoData(
    DATASET_ID,
    DIRS,
    parameters=params,
    overwrite_dataset=True,
    overwrite_data=True,
    random_subset=10,
)
out_dir = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
dataset = load_dataset(out_dir)
dataset.segment_into_units()


key = dataset.data.index[0]

key = dataset.data["length_s"].idxmax()
# melspectrogram(
#     dataset.files.at[key, "spectrogram"],
#     parameters=dataset.parameters,
#     title=Path(key).stem,
#     cmap = "bone",
#     max_lenght = None,
#     colour_bar=False
# )
#%%

key = dataset.data["length_s"].idxmin()

nparray_or_dir = dataset.files.at[key, "spectrogram"]
parameters = dataset.parameters
title = Path(key).stem
cmap = "bone"
max_lenght = None
colour_bar = True  # Chenge default to true


def melspectrogram(
    nparray_or_dir: Path | np.ndarray,
    parameters: None | Parameters = None,
    title: None | str = None,
    cmap: str = "bone",
    max_lenght: None | float = None,
    colour_bar=True,
) -> Axes:

    if isinstance(nparray_or_dir, np.ndarray):
        mel_spectrogram = nparray_or_dir
    elif isinstance(nparray_or_dir, Path):
        mel_spectrogram = retrieve_spectrogram(nparray_or_dir)
    else:
        raise TypeError("nparray_or_dir must be of type Path or np.ndarray")
    if parameters is None:
        warnings.warn(
            "You need to provide a Parameters object; "
            "setting defaults which will likely be inadequate."
        )
        parameters = Parameters()

    # Shorten spectrogram if needed
    if max_lenght:
        max_len_frames = math.floor(
            max_lenght * parameters.sr / parameters.hop_length
        )
        if max_len_frames > mel_spectrogram.shape[1]:
            max_len_frames = mel_spectrogram.shape[1]
            warnings.warn(
                f"{max_lenght=} is longer than the spectrogram, "
                "setting max_lenght to the length of the spectrogram"
            )
        mel_spectrogram = cut_or_pad_spectrogram(
            mel_spectrogram, max_len_frames
        )

    # Fig settings
    back_colour = "white"
    text_colour = "#636363"
    shape = mel_spectrogram.shape[::-1]
    figsize = tuple([x / 50 for x in shape])

    # Plot spectrogram proper
    fig, ax = plt.subplots(figsize=figsize, facecolor=back_colour)
    spec_im = librosa.display.specshow(
        mel_spectrogram,
        x_axis="time",
        y_axis="mel",
        hop_length=parameters.hop_length,
        fmin=parameters.lowcut,
        fmax=parameters.highcut,
        sr=parameters.sr,
        cmap=cmap,
        ax=ax,
    )
    # Set limits
    spec_im.set_clim(np.min(mel_spectrogram) - 10, np.max(mel_spectrogram))

    # Set background in case spectrogram doesnt reach 0
    ax.set_facecolor(get_cmap(cmap)(0))

    # Ticks and labels
    ax0, ax1 = (0, ax.get_xlim()[1])
    ax.set_xticks([l for l in np.arange(ax0, ax1, 0.4)])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%1.1f"))
    ax.tick_params(axis="both", which="both", length=0, colors=text_colour)
    plt.tick_params(
        axis="both", which="major", labelsize=12, pad=15, colors=text_colour
    )
    ax.set_xlabel("Time (s)", fontsize=15, labelpad=12, color=text_colour)
    ax.set_ylabel("Frequency (Hz)", fontsize=15, labelpad=12, color=text_colour)
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(False)

    # Colour bar
    if colour_bar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        cbar = plt.colorbar(spec_im, cax=cax)
        cbar.ax.tick_params(size=0, labelsize=9, pad=10, colors=text_colour)
        cbar.outline.set_visible(False)
        cax.set_title("# of contacts")
        cbar.ax.set_ylabel("dB", color=text_colour, size=12, labelpad=12)
        cbar.ax.yaxis.set_label_position("right")

    # Title
    if isinstance(title, str):
        ax.set_title(title, color=text_colour, fontsize=15, pad=12)
    elif title is not None:
        warnings.warn(f"Title must be of type str or None")
    # plt.show() #REVIEW
    return ax


#
#%% Segmentation plot

# dataset: KantoData,
# key: str = None,
spectrogram: bool | np.ndarray = False
onsets_offsets: bool | Tuple[np.ndarray, np.ndarray] = False

#
# def segmentation(
#     dataset: KantoData,
#     key: str = None,
#     spectrogram: bool | np.ndarray = False,
#     onsets_offsets: bool | Tuple[np.ndarray, np.ndarray] = False,
#     **kwargs,
# ) -> None:
"""
Plots a vocalisation and overlays the results
of the segmentation process.

Args:
    dataset (KantoData): A KantoData object.
    key (str, optional): Vocalisation key. Defaults to None.
    spectrogram (bool | np.ndarray, optional): [description].
        Defaults to False.
    onsets_offsets (bool | Tuple[np.ndarray, np.ndarray], optional):
        Tuple containing arrays with unit onsets and offsets. Defaults to False.
    kwargs: Keyword arguments to be passed to
            :func:`~pykanto.plot.melspectrogram`
"""
params = dataset.parameters
if not isinstance(spectrogram, np.ndarray):
    spectrogram = retrieve_spectrogram(dataset.files.at[key, "spectrogram"])
    onsets_offsets = [dataset.data.at[key, i] for i in ["onsets", "offsets"]]

ax = melspectrogram(spectrogram, parameters=params, title=key if key else "")

# Add unit onsets and offsets
ylmin, ylmax = ax.get_ylim()
ysize = (ylmax - ylmin) * 0.05
ymin = ylmax - ysize
patches = []
for onset, offset in zip(*onsets_offsets):
    ax.axvline(onset, color="#FFFFFF", ls="-", lw=0.5, alpha=0.5)
    ax.axvline(offset, color="#FFFFFF", ls="-", lw=0.5, alpha=0.5)
    patches.append(
        Rectangle(
            xy=(onset, ylmin), width=offset - onset, height=(ylmax - ylmin)
        )
    )
collection = PatchCollection(patches, color="white", alpha=0.07)
ax.add_collection(collection)
plt.show()


segmentation(dataset, key)
