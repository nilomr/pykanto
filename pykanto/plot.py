# ─── DESCRIPTION ──────────────────────────────────────────────────────────────

"""
A collection of functions to plot spectrograms. annotations and other output.
"""

# ──── IMPORTS ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba.core.decorators import njit

from pykanto.parameters import Parameters
from pykanto.signal.spectrogram import (
    cut_or_pad_spectrogram,
    retrieve_spectrogram,
)

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
    ax.set(yscale="log")
    ax.set(yticks=ticks, yticklabels=ticks)
    plt.show()


def melspectrogram(
    nparray_or_dir: Path | np.ndarray,
    parameters: None | Parameters = None,
    title: None | str = None,
    cmap: str = "bone",
    max_lenght: None | float = None,
    colour_bar=True,
) -> Axes:
    """
    Plots a melspectrogram from a numpy array or path to a numpy array.

    Args:
        nparray_or_dir (Path | np.ndarray): Spectrogram array or path to a
            stored numpy array.
        parameters (None | Parameters, optional): Parameters used to
            calculate the spectrogram. Defaults to None.
        title (None | str, optional): Title for plot. Defaults to None.
        cmap (str, optional): Matplotlib colour palette to use.
            Defaults to "bone".
        max_lenght (None | float, optional): Maximum length of the
            spectrogram beyond which it will be center-cropped for plotting.
            Defaults to None.
        colour_bar (bool, optional): Wheter to include a colour bar
            legend for the amplitude. Defaults to True.

    Returns:
        Axes: A matplotlib.axes.Axes instance
    """

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
        cbar.ax.set_ylabel("dB", color=text_colour, size=12, labelpad=12)
        cbar.ax.yaxis.set_label_position("right")

    # Title
    if isinstance(title, str):
        ax.set_title(title, color=text_colour, fontsize=15, pad=12)
    elif title is not None:
        warnings.warn(f"Title must be of type str or None")
    # plt.show() #REVIEW
    return ax


def segmentation(
    dataset: KantoData,
    key: str | None = None,
    spectrogram: bool | np.ndarray = False,
    onsets_offsets: bool | Tuple[np.ndarray, np.ndarray] = False,
    **kwargs,
) -> None:
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
        onsets_offsets = [
            dataset.data.at[key, i] for i in ["onsets", "offsets"]
        ]

    title = kwargs.pop("title") if "title" in kwargs else (key if key else "")
    max_lenght = kwargs.pop("max_lenght") if "max_lenght" in kwargs else None

    if max_lenght:
        max_len_frames = math.floor(max_lenght * params.sr / params.hop_length)
        if max_len_frames > spectrogram.shape[1]:
            max_len_frames = spectrogram.shape[1]
            warnings.warn(
                f"{max_lenght=} is longer than the spectrogram, "
                "setting max_lenght to the length of the spectrogram"
            )
        spectrogram = spectrogram[:, :max_len_frames]
        onsets_offsets = (
            onsets_offsets[0][onsets_offsets[0] <= max_lenght],
            onsets_offsets[1][onsets_offsets[1] <= max_lenght],
        )

    ax = melspectrogram(spectrogram, parameters=params, title=title, **kwargs)

    # Add unit onsets and offsets
    ylmin, ylmax = ax.get_ylim()
    ysize = (ylmax - ylmin) * 0.05
    ymin = ylmax - ysize
    patches = []
    for onset, offset in zip(*onsets_offsets):
        ax.axvline(onset, color="#FFFFFF", ls="-", lw=0.7, alpha=0.7)
        ax.axvline(offset, color="#FFFFFF", ls="-", lw=0.7, alpha=0.7)
        patches.append(
            Rectangle(
                xy=(onset, ylmin), width=offset - onset, height=(ylmax - ylmin)
            )
        )
    collection = PatchCollection(patches, color="white", alpha=0.07)
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
    if M > N / 2:
        cut = np.zeros(N, dtype=int)
        q, r = divmod(N, N - M)
        indices = [q * i + min(i, r) for i in range(N - M)]
        cut[indices] = True
    else:
        cut = np.ones(N, dtype=int)
        q, r = divmod(N, M)
        indices = [q * i + min(i, r) for i in range(M)]
        cut[indices] = False
    return cut


@njit
def rand_jitter(arr: np.ndarray, jitter: float = 0.001) -> np.ndarray:
    """
    Adds random jitter to an array.

    Args:
        arr (np.ndarray): Array to jitter.
        jitter (float, optional): Jitter factor. Defaults to 0.001.

    Returns:
        np.ndarray: Jittered array.
    """
    stdev = jitter * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def show_spec_centroid_bandwidth(
    dataset: KantoData,
    centroid: np.ndarray,
    spec_bw: np.ndarray,
    key: None | str = None,
    spec: None | np.ndarray = None,
) -> None:
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
        KeyError("You need to provide either a key or a spectrogram")

    if not isinstance(spec, np.ndarray):
        spec = retrieve_spectrogram(dataset.files.at[key, "spectrogram"])

    times = (
        np.array(range(spec.shape[1]))
        * dataset.parameters.hop_length
        / dataset.parameters.sr
    )

    ax = melspectrogram(spec, parameters=dataset.parameters, title=key)
    ax.fill_between(
        times,
        np.maximum(0, centroid - spec_bw),
        np.minimum(centroid + spec_bw, dataset.parameters.sr / 2),
        alpha=0.5,
        label="Bandwidth",
    )
    ax.plot(times, centroid, label="Spectral centroid", color="w")
    ax.legend(loc="upper right", frameon=False, labelcolor="w")
    plt.show()


def show_minmax_frequency(
    dataset: KantoData,
    minfreqs: np.ndarray,
    maxfreqs: np.ndarray,
    roll_percents: list[float, float],
    key: None | str = None,
    spec: None | np.ndarray = None,
) -> None:
    """
    Plots approximate minimum and maximum frequencies over a mel spectrogram.
    You can either provide a key string for a vocalisation or its
    mel spectrogram directly.

    Args:
        dataset (KantoData): Dataset object with your data.
        maxfreqs (np.ndarray): Array of maximum frequencies.
        minfreqs (np.ndarray): Array of minimum frequencies.
        roll_percents (list[float, float]): Percentage of energy
            contained in bins.
        key (None | str = None): Key of a vocalisation. Defaults to None.
        spec (spec: None | np.ndarray): Mel spectrogram. Defaults to None.
    """

    if not key and not isinstance(spec, np.ndarray):
        KeyError("You need to provide either a key or a spectrogram")

    if not isinstance(spec, np.ndarray):
        spec = retrieve_spectrogram(dataset.files.at[key, "spectrogram"])

    times = (
        np.array(range(spec.shape[1]))
        * dataset.parameters.hop_length
        / dataset.parameters.sr
    )

    ax = melspectrogram(spec, parameters=dataset.parameters, title=key)
    ax.plot(times, maxfreqs, label=f"Max frequency (roll = {roll_percents[0]})")
    ax.plot(times, minfreqs, label=f"Min frequency (roll ={roll_percents[1]})")
    ax.legend(loc="upper right", frameon=False, labelcolor="w")
    plt.show()


def build_plot_summary(
    dataset: KantoData, nbins: int = 50, variable: str = "frequency"
) -> None:
    """
    Plots a histogram + kernel densiyy estimate of the frequency
    distribution of vocalisation duration and frequencies.

    Note:
        Durations and frequencies come from bounding boxes,
        not vocalisations. This function, along with
        :func:`~pykanto.dataset.show_extreme_songs`, is useful to spot
        any outliers, and to quickly explore the full range of data.

    Args:
        dataset (KantoData): Dataset to use.
        nbins (int, optional): Number of bins in histogram. Defaults to 50.
        variable (str, optional): One of 'frequency', 'duration',
            'sample_size', 'all'. Defaults to 'frequency'.

    Raises:
        ValueError: `variable` must be one of
            ['frequency', 'duration', 'sample_size', 'all']
    """

    if variable not in ["frequency", "duration", "sample_size", "all"]:
        raise ValueError(
            "`variable` must be one of ['frequency', 'duration', 'sample_size', 'all']"
        )

    # Plot size and general aesthetics
    sns.set(font_scale=1.5, rc={"axes.facecolor": "#ededed"}, style="dark")
    fig, axes = plt.subplots(
        figsize=(18 if variable == "all" else 6, 6),
        ncols=3 if variable == "all" else 1,
    )

    # Build frequency or duration plots
    for i, var in enumerate(["frequency", "duration"]):
        if var != variable and variable != "all":
            continue

        if var == "frequency":
            data = {
                "upper_freq": dataset.data["upper_freq"],
                "lower_freq": dataset.data["lower_freq"],
            }
        else:
            data = {"song_duration": dataset.data["length_s"]}

        sns.histplot(
            data,
            bins=nbins,
            kde=True,
            palette=["#107794", "#d97102"]
            if var == "frequency"
            else ["#107794"],
            legend=False,
            ax=axes if variable != "all" else axes[i],
            linewidth=0.2,
            log_scale=True if var == "duration" else False,
            line_kws=dict(linewidth=5, alpha=0.7),
        )

        if var == "duration":
            (axes[i] if variable == "all" else axes).xaxis.set_major_formatter(
                mpl.ticker.ScalarFormatter()
            )

        if var == "frequency":
            (axes[i] if variable == "all" else axes).legend(
                labels=["Min", "Max"],
                loc=2,
                bbox_to_anchor=(0.70, 1),
                borderaxespad=0,
                frameon=False,
            )
        (
            (axes[i] if variable == "all" else axes).set(
                xlabel="Frequency (Hz)"
                if var == "frequency"
                else "Duration (s)",
                ylabel="Count",
            )
        )

    # Build sample size plot
    if variable in ["sample_size", "all"]:
        individual_ss = dataset.data["ID"].value_counts()
        data = pd.DataFrame(individual_ss).rename(columns={"ID": "n"})
        data["ID"] = individual_ss.index
        sns.histplot(
            data=data,
            palette=["#107794"],
            bins=nbins,
            ax=axes if variable != "all" else axes[2],
            alpha=0.6,
            legend=False,
        )
        (axes[2] if variable == "all" else axes).set(
            xlabel=f"Sample size (total: {len(dataset.data)})",
            ylabel="Count",
        )
        # Reduce xtick density
        nlabs = len((axes[2] if variable == "all" else axes).get_xticklabels())
        mid = math.trunc(nlabs / 2)
        for i, label in enumerate(
            (axes[2] if variable == "all" else axes).get_xticklabels()
        ):
            if i not in [0, mid, nlabs - 1]:
                label.set_visible(False)

    # Common axes
    if variable == "all":
        for ax in axes:
            ax.yaxis.labelpad = 15
            ax.xaxis.labelpad = 15
    else:
        axes.yaxis.labelpad = 15
        axes.xaxis.labelpad = 15

    fig.tight_layout()
    plt.show()
