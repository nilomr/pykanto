
# ─── DESCRIPTION ──────────────────────────────────────────────────────────────

"""
Classes and methods to store and modify pykanto parameters.
"""

# ──── IMPORTS ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.cm import get_cmap
from matplotlib import gridspec
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numba.core.decorators import njit
from pykanto.signal.spectrogram import cut_or_pad_spectrogram, retrieve_spectrogram
from pykanto.parameters import Parameters

if TYPE_CHECKING:
    from pykanto.dataset import KantoData


# ──── FUNCTIONS ───────────────────────────────────────────────────────────────

def sns_histoplot(data, nbins=100) -> None:
    """
    Plots a histogram using seaborn
    """

    ticks = [1, 10, 100, 1000, 10000]
    sns.set_style("dark")
    ax = sns.displot(data, bins=nbins)
    ax.set(yscale='log')
    ax.set(yticks=ticks, yticklabels=ticks)
    plt.show()


def melspectrogram(
        nparray_or_dir: Path | np.ndarray,
        parameters: None | Parameters = None,
        title: None | str = None,
        cmap: str = 'bone',
        max_lenght: None | float = None,
        colour_bar=False) -> None:

    if isinstance(nparray_or_dir, np.ndarray):
        mel_spectrogram = nparray_or_dir
    elif isinstance(nparray_or_dir, Path):
        mel_spectrogram = retrieve_spectrogram(nparray_or_dir)
    else:
        raise TypeError('nparray_or_dir must be of type Path or np.ndarray')
    if parameters is None:
        warnings.warn('You need to provide a Parameters object; '
                      'setting defaults which will likely be inadequate.')
        parameters = Parameters()

    # Shorten spectrogram if needed
    if max_lenght:
        max_len_frames = math.floor(
            max_lenght * parameters.sr / parameters.hop_length)
        if max_len_frames > mel_spectrogram.shape[1]:
            max_len_frames = mel_spectrogram.shape[1]
            warnings.warn(f"{max_lenght=} is longer than the spectrogram, "
                          "setting max_lenght to the length of the spectrogram")
        mel_spectrogram = cut_or_pad_spectrogram(
            mel_spectrogram, max_len_frames)

    # Fig settings
    back_colour = '#2F2F2F'
    text_colour = '#c2c2c2'
    shape = mel_spectrogram.shape[::-1]
    figsize = tuple([x/50 for x in shape])

    # Plot spectrogram proper
    fig, ax = plt.subplots(figsize=figsize, facecolor=back_colour)
    spec_im = librosa.display.specshow(
        mel_spectrogram, x_axis="time", y_axis="mel",
        hop_length=parameters.hop_length, fmin=parameters.lowcut,
        fmax=parameters.highcut, sr=parameters.sr, cmap=cmap, ax=ax)

    # Set background in case spectrogram doesnt reach 0
    ax.set_facecolor(get_cmap(cmap)(0))

    # Ticks and labels
    xlims = ax.get_xlim()
    ax.set_xticks([min(xlims),  max(xlims)*0.25, max(xlims)
                  * 0.5, max(xlims)*0.75, max(xlims)])
    ax.xaxis.set_major_formatter(FormatStrFormatter('%1.1f'))
    ax.tick_params(axis=u'both', which=u'both', length=0, colors=text_colour)
    plt.tick_params(axis='both', which='major',
                    labelsize=12, pad=15, colors=text_colour)
    ax.set_xlabel('Time (s)', fontsize=15, labelpad=12, color=text_colour)
    ax.set_ylabel('Frequency (Hz)', fontsize=15, labelpad=12, color=text_colour)
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)

    # Colour bar
    if colour_bar:
        cbaxes = inset_axes(
            ax, width=0.2, height="100%", loc="lower left",
            bbox_to_anchor=(1, 0, 1, 1),
            bbox_transform=ax.transAxes, borderpad=0,)
        cbar = plt.colorbar(mappable=spec_im, cax=cbaxes,
                            orientation="vertical", pad=0)
        clims = cbar.ax.get_xlim()
        cbar.set_ticks([int(x)
                        for x in
                        [max(clims),
                        min(clims) / 2, math.floor(min(clims))]])
        cbar.ax.tick_params(size=0, labelsize=9, pad=10, colors=text_colour)
        cbar.outline.set_visible(False)

    # Title
    if isinstance(title, str):
        ax.set_title(title, color=text_colour, fontsize=15, pad=12)
    elif title is not None:
        warnings.warn(f'Title must be of type str or None')
    # plt.show() #REVIEW
    return ax


def segmentation(
        dataset: KantoData, key: str = None,
        spectrogram: bool | np.ndarray = False,
        onsets_offsets: bool | Tuple[np.ndarray, np.ndarray] = False,
        **kwargs) -> None:
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
        spectrogram = retrieve_spectrogram(dataset.vocs.at
                                           [key, 'spectrogram_loc'])
        onsets_offsets = [dataset.vocs.at[key, i]
                          for i in ['onsets', 'offsets']]

    ax = melspectrogram(spectrogram, parameters=params,
                        title=key if key else '', **kwargs)

    # Add unit onsets and offsets
    ylmin, ylmax = ax.get_ylim()
    ysize = (ylmax - ylmin) * 0.05
    ymin = ylmax - ysize
    patches = []
    for onset, offset in zip(*onsets_offsets):
        # ax.axvline(onset, color="#FFFFFF", ls="-", lw=0.5, alpha=0.3)
        # ax.axvline(offset, color="#FFFFFF", ls="-", lw=0.5, alpha=0.3)
        patches.append(Rectangle(xy=(onset, ylmin),
                                 width=offset - onset, height=(ylmax - ylmin)))
    collection = PatchCollection(patches, color="white", alpha=0.1)
    ax.add_collection(collection)
    plt.show()


def mspaced_mask(N: int, M: int) -> List[int]:
    """
    Returns a mask of length N to select M 
    indices as regularly spaced as possible.

    Args:
        N (int): Lenght of list
        M (int): Number of indices to return

    Returns:
        List[int]: A binary mask
    """
    if M > N/2:
        cut = np.zeros(N, dtype=int)
        q, r = divmod(N, N-M)
        indices = [q*i + min(i, r) for i in range(N-M)]
        cut[indices] = True
    else:
        cut = np.ones(N, dtype=int)
        q, r = divmod(N, M)
        indices = [q*i + min(i, r) for i in range(M)]
        cut[indices] = False
    return cut


@njit
def rand_jitter(arr, jitter: float = .001):
    stdev = jitter * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def show_spec_centroid_bandwidth(
        dataset: KantoData, centroid: np.ndarray, spec_bw: np.ndarray,
        key: None | str = None, spec: None | np.ndarray = None) -> None:
    """
    Plots spectral centroids and bandwiths over a mel spectrogram.
    You can either provide a key string for a vocalisation or its
    mel spectrogram directly.

    Args:
        dataset (KantoData): Dataset object with your data.
        centroid (np.ndarray): Array of centroids.
        spec_bw (np.ndarray): Array of badwidths.
        key (None | str = None): Key of a vocalisation. Defaults to None.
        spec (spec: None | np.ndarray): Mel spectrogram. Defaults to None.
    """

    if not key and not isinstance(spec, np.ndarray):
        KeyError('You need to provide either a key or a spectrogram')

    if not isinstance(spec, np.ndarray):
        spec = retrieve_spectrogram(dataset.vocs.at
                                    [key, 'spectrogram_loc'])

    times = (np.array(range(spec.shape[1]))
             * dataset.parameters.hop_length / dataset.parameters.sr)

    ax = melspectrogram(spec, parameters=dataset.parameters, title=key)
    ax.fill_between(
        times, np.maximum(0, centroid - spec_bw),
        np.minimum(centroid + spec_bw,
                   dataset.parameters.sr / 2),
        alpha=0.5, label='Bandwidth')
    ax.plot(times, centroid, label='Spectral centroid', color='w')
    ax.legend(loc='upper right', frameon=False, labelcolor='w')
    plt.show()


def show_minmax_frequency(
        dataset: KantoData, minfreqs: np.ndarray, maxfreqs: np.ndarray,
        key: None | str = None, spec: None | np.ndarray = None) -> None:
    """
    Plots approximate minimum and maximum frequencies over a mel spectrogram.
    You can either provide a key string for a vocalisation or its
    mel spectrogram directly.

    Args:
        dataset (KantoData): Dataset object with your data.
        rolloff_max (np.ndarray): Array of maximum frequencies.
        rolloff_min (np.ndarray): Array of minimum frequencies.
        key (None | str = None): Key of a vocalisation. Defaults to None.
        spec (spec: None | np.ndarray): Mel spectrogram. Defaults to None.
    """

    if not key and not isinstance(spec, np.ndarray):
        KeyError('You need to provide either a key or a spectrogram')

    if not isinstance(spec, np.ndarray):
        spec = retrieve_spectrogram(dataset.vocs.at
                                    [key, 'spectrogram_loc'])

    times = (np.array(range(spec.shape[1]))
             * dataset.parameters.hop_length / dataset.parameters.sr)

    ax = melspectrogram(spec, parameters=dataset.parameters, title=key)
    ax.plot(times, maxfreqs, label='Max frequency (roll = 0.95)')
    ax.plot(times, minfreqs, label='Min frequency (roll = 0.1)')
    ax.legend(loc='upper right', frameon=False, labelcolor='w')
    plt.show()
