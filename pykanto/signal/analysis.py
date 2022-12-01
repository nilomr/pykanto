# ─── DESCRIPTION ──────────────────────────────────────────────────────────────

"""
Basic audio feature calculations (spectral centroids, peak frequencies, etc.)
"""

# ──── IMPORTS ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, List, Tuple

import librosa
import numpy as np
from pykanto.plot import show_minmax_frequency, show_spec_centroid_bandwidth
from pykanto.signal.filter import mels_to_hzs
from pykanto.signal.spectrogram import retrieve_spectrogram

if TYPE_CHECKING:
    from pykanto.dataset import KantoData

# ──── FUNCTIONS ───────────────────────────────────────────────────────────────


def get_peak_freqs(
    dataset: KantoData,
    spectrograms: np.ndarray,
    melscale: bool = True,
    threshold: float = 0.3,
) -> np.ndarray:
    """
    Return the peak frequencies of each spectrogram in an array of spectrograms.

    Args:
        dataset (KantoData): Vocalisation dataset.
        spectrograms (np.ndarray): Array of np.ndarray spectrograms.
        melscale (bool, optional): Are the spectrograms in the mel scale? Defaults to True.
        threshold (float, optional): Threshold for peak detection. Defaults to 0.3.

    Returns:
        np.ndarray: Array with peak frequencies.
    """

    minfreq = dataset.parameters.lowcut
    min_db = -dataset.parameters.top_dB

    if melscale:
        hz_freq = mels_to_hzs(dataset)
        result = np.array(
            [
                hz_freq[np.argmax(np.max(w, axis=1))]
                if (max(np.max(w, axis=1)) > min_db * (1 - threshold))
                else -1
                for w in spectrograms
            ]
        )

        return result

    else:
        return np.array(
            [minfreq + np.argmax(np.max(w, axis=1)) for w in spectrograms]
        )
        # REVIEW did not test for melscale = False


def spec_centroid_bandwidth(
    dataset: KantoData,
    key: None | str = None,
    spec: None | np.ndarray = None,
    plot: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate a vocalisation's spectral centroid and bandwidth from a mel
    spectrogram. You can either provide a key string for a vocalisation or its
    mel spectrogram directly.

    Args:
        dataset (KantoData): Dataset object with your data.
        key (None | str = None): Key of a vocalisation. Defaults to None.
        spec (spec: None | np.ndarray): Mel spectrogram. Defaults to None.
        plot (bool, optional): Whether to show the result. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple with the centroids
            and bandwidths.
    """

    if not key and not isinstance(spec, np.ndarray):
        raise KeyError("You need to provide either a key or a spectrogram")
    if not isinstance(spec, np.ndarray):
        spec = retrieve_spectrogram(dataset.files.at[key, "spectrogram"])

    offset = 0
    if np.min(spec) < 0:
        offset = abs(np.min(spec))
    centroid = librosa.feature.spectral_centroid(
        S=spec + offset, freq=mels_to_hzs(dataset)
    )[0]
    spec_bw = librosa.feature.spectral_bandwidth(
        S=spec + offset, freq=mels_to_hzs(dataset)
    )[0]
    centroid[centroid <= dataset.parameters.lowcut] = np.nan
    spec_bw[spec_bw <= dataset.parameters.lowcut] = np.nan

    if plot:
        show_spec_centroid_bandwidth(
            dataset, centroid, spec_bw, key=key, spec=spec
        )

    return centroid, spec_bw


def get_mean_sd_mfcc(S: np.ndarray, n_mfcc: int) -> np.ndarray:
    """
    Extract the mean and SD of n Mel-frequency cepstral coefficients (MFCCs)
    calculated ffrom a log-power Mel spectrogram.

    Args:
        S (np.ndarray): A log-power Mel spectrogram.
        n_mfcc (int): Number of coefficients to return.

    Returns:
        np.ndarray: Array containing mean and std of each coefficient (len =
        n_mfcc*2).
    """
    mfcc = librosa.feature.mfcc(S=S, n_mfcc=n_mfcc)
    mean_sd = np.hstack((np.mean(mfcc, axis=1), np.std(mfcc, axis=1)))
    return mean_sd


def approximate_minmax_frequency(
    dataset: KantoData,
    key: None | str = None,
    spec: None | np.ndarray = None,
    roll_percents: list[float] = [0.95, 0.1],
    plot: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate approximate minimum and maximum frequencies from a mel
    spectrogram. You can either provide a key string for a vocalisation or its
    mel spectrogram directly.

    Args:
        dataset (KantoData): Dataset object with your data.
        key (None | str = None): Key of a vocalisation. Defaults to None.
        spec (spec: None | np.ndarray): Mel spectrogram. Defaults to None.
        roll_percents (list[float, float], optional): Percentage of energy
            contained in bin. Defaults to [0.95, 0.1].
        plot (bool, optional): Whether to show the result. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple with the approximate minimum and
            maximum frequencies, in this order.
    """

    if not key and not isinstance(spec, np.ndarray):
        raise KeyError("You need to provide either a key or a spectrogram")
    if not isinstance(spec, np.ndarray):
        spec = retrieve_spectrogram(dataset.files.at[key, "spectrogram"])

    offset = 0
    if np.min(spec) < 0:
        offset = abs(np.min(spec))

    maxfreqs, minfreqs = [
        librosa.feature.spectral_rolloff(
            S=spec + offset,
            sr=dataset.parameters.sr,
            roll_percent=p,
            freq=mels_to_hzs(dataset),
        )[0]
        for p in roll_percents
    ]

    maxfreqs[maxfreqs <= dataset.parameters.lowcut] = np.nan
    minfreqs[minfreqs <= dataset.parameters.lowcut] = np.nan

    if plot:
        show_minmax_frequency(
            dataset, maxfreqs, minfreqs, roll_percents, key=key, spec=spec
        )

    return minfreqs, maxfreqs
