# ─── DESCRIPTION ──────────────────────────────────────────────────────────────

"""
Functions and methods to perform spectrogram filtering, dereverberating,
bandpassing, etc.
"""

# ──── IMPORTS ─────────────────────────────────────────────────────────────────

from __future__ import annotations
from typing import TYPE_CHECKING, Tuple

import librosa
import numpy as np
from numba.core.decorators import njit
from scipy.ndimage import gaussian_filter

if TYPE_CHECKING:
    from pykanto.dataset import KantoData

# ──── FUNCTIONS ───────────────────────────────────────────────────────────────


def dereverberate(
    spectrogram: np.ndarray,
    echo_range: int = 100,
    echo_reduction: float = 0.5,
    hop_length: int = 128,
    sr: int = 22050,
) -> np.ndarray:
    """
    Reduce echo in a spectrogram by subtracting a delayed version of itself.
    Based in JS code from
    `Robert Lachlan <https://rflachlan.github.io/Luscinia/>`_.

    Args:
        spectrogram (np.ndarray): Data to dereverberate.
        echo_range (int, optional): How many frames to dereverb.
            Defaults to 100.
        echo_reduction (float, optional): How much reduction to perform.
            Defaults to 0.5.
        hop_length (int, optional): Hop length used to create the spectrogram.
            Defaults to 128.
        sr (int, optional): Sampling ratio. Defaults to 22050.

    Returns:
        np.ndarray: De-echoed spectrogram.
    """

    hop_length_ms = int(hop_length / sr * 1000)
    echo_range = int(echo_range / hop_length_ms)

    nbins = len(spectrogram[0])
    mindb = np.min(spectrogram)

    if echo_range > nbins:
        echo_range = nbins

    newspec = []

    for row in spectrogram:
        newrow = []
        for colindex, amplitude in enumerate(row):
            anterior = row[(colindex - echo_range) : colindex]
            if colindex < echo_range:
                posterior = row[colindex : (colindex + echo_range)]
                newval = amplitude - echo_reduction * (
                    max(posterior) - amplitude
                )
                newrow.append(newval if newval > mindb else mindb)
            elif (len(anterior) > 0) and (max(anterior) > amplitude):
                newval = amplitude - echo_reduction * (
                    max(anterior) - amplitude
                )
                newrow.append(newval if newval > mindb else mindb)
            else:
                newrow.append(amplitude)
        newspec.append(newrow)

    return np.asarray(newspec)


dereverberate_jit = njit(dereverberate)


def get_norm_spectral_envelope(
    mel_spectrogram: np.ndarray, mindb: int, kernel_size: int = 5
) -> np.ndarray:
    """
    Returns a spectral envelope of sorts - useful to quickly characterise a
    song's frequency distribution. Minmax rescaled to [0,1]

    Args:
        mel_spectrogram (np.ndarray): Source spectrogram
        mindb (int): Min value to threshold.
        kernel_size (int, optional): That. Defaults to 10.

    Returns:
        np.ndarray: [description]
    """

    spec = norm(normalise(mel_spectrogram, min_level_db=-mindb))
    spec = spec - np.median(spec, axis=1).reshape((len(spec), 1))
    spec[spec < 0] = 0
    amp_envelope = np.max(spec, axis=1) * np.sqrt(np.mean(spec, axis=1))

    kernel = np.ones(kernel_size) / kernel_size
    data_convolved = np.convolve(amp_envelope, kernel, mode="same")

    return norm(data_convolved)


def gaussian_blur(array: np.ndarray, gauss_sigma: int = 3, max: int = 0):
    """
    Gaussian blur a 2d numpy array, rescale to desired range to account for loss.

    Args:
        array (np.ndarray): An array to be blurred
        gauss_sigma (int, optional): Standard deviation for Gaussian kernel. Defaults to 3.
        max (int, optional): Max value in returned array. Defaults to 0.

    Returns:
        np.ndarray: Blurred and interpolated array
    """
    array = gaussian_filter(array, sigma=gauss_sigma)
    return np.interp(array, (array.min(), array.max()), (array.min(), max))
    # NOTE: can change 0 to mel_spectrogram.max() if needed


# ────────────────────────────────────────────────────────────────────────────────

# Minor helper functions here


class kernels:
    """
    Class holding kernels for use in filtering.
    """

    erosion_kern = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])

    dilation_kern = np.array(
        [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]
    )


# write function to normalise a numpy array to [0,1]:


@njit
def norm(x: np.ndarray) -> np.ndarray:
    """
    Normalise a numpy array to [0,1].

    Args:
        x (np.ndarray): Array to normalise.

    Returns:
        np.ndarray: Normalised array.
    """
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def normalise(S: np.ndarray, min_level_db: int) -> np.ndarray:
    """
    Normalise a numpy array to [0,1] and clip to min_level_db.

    Args:
        S (np.ndarray): Array to normalise.
        min_level_db (int): Threshold, in relative dB.

    Returns:
        np.ndarray: Normalised and clipped array.
    """
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)


def hz_to_mel_lib(hz: int, minmax_freq: Tuple[int, int], parameters):
    """Convert hz to mel frequencies

    Args:
        hz (int): [description]
        minmax_freq (Tuple): [description]
        parameters ([type]): [description]

    Returns:
        [type]: [description]
    """
    freqs = librosa.core.mel_frequencies(
        fmin=minmax_freq[0], fmax=minmax_freq[1], n_mels=parameters.num_mel_bins
    )
    return np.argmin(abs(freqs - hz))


def mel_to_hz(mel_bin: int, dataset: KantoData) -> int:
    """
    Returns the original frequency from a mel bin index.
    Requires a KantoData object with set parameters.

    Args:
        mel_bin (int): Mel bin to convert
        dataset (KantoData): KantoData object

    Returns:
        int: Approximate original frequency in hertzs
    """
    freqs = librosa.core.mel_frequencies(
        fmin=dataset.parameters.lowcut,
        fmax=dataset.parameters.highcut,
        n_mels=dataset.parameters.num_mel_bins,
    )
    return int(freqs[mel_bin])


def mels_to_hzs(dataset: KantoData) -> np.ndarray[int]:
    """
    Returns the original frequencies from the mel bins used in a dataset.
    Requires a KantoData object with set parameters.

    Args:
        dataset (KantoData): KantoData object

    Returns:
        np.ndarray[int]: Approximate original frequencies in hertzs
    """
    freqs = librosa.core.mel_frequencies(
        fmin=dataset.parameters.lowcut,
        fmax=dataset.parameters.highcut,
        n_mels=dataset.parameters.num_mel_bins,
    )
    return freqs.astype(int)
