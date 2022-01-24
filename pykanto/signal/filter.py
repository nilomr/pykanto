
from __future__ import annotations
from typing import TYPE_CHECKING, Tuple

import librosa
import numpy as np
from numba.core.decorators import njit
from scipy.ndimage import gaussian_filter

if TYPE_CHECKING:
    from pykanto.dataset import SongDataset


def dereverberate(spectrogram: np.ndarray, echo_range: int = 100,
                  echo_reduction: float = 0.5,
                  hop_length: int = 128, sr: int = 22050) -> np.ndarray:

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
            anterior = row[(colindex - echo_range): colindex]
            if colindex < echo_range:
                posterior = row[colindex: (colindex + echo_range)]
                newval = amplitude - echo_reduction * \
                    (max(posterior) - amplitude)
                newrow.append(newval if newval > mindb else mindb)
            elif (len(anterior) > 0) and (max(anterior) > amplitude):
                newval = amplitude - echo_reduction * \
                    (max(anterior) - amplitude)
                newrow.append(newval if newval > mindb else mindb)
            else:
                newrow.append(amplitude)
        newspec.append(newrow)

    return np.asarray(newspec)


dereverberate_jit = njit(dereverberate)


def get_norm_spectral_envelope(
        mel_spectrogram: np.ndarray,
        mindb: int, kernel_size: int = 5) -> np.ndarray:
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

    spec = norm(normalise(mel_spectrogram, min_level_db=- mindb))
    spec = spec - np.median(spec, axis=1).reshape((len(spec), 1))
    spec[spec < 0] = 0
    amp_envelope = np.max(spec, axis=1) * np.sqrt(np.mean(spec, axis=1))

    kernel = np.ones(kernel_size) / kernel_size
    data_convolved = np.convolve(amp_envelope, kernel, mode='same')

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

class kernels():
    erosion_kern = np.array(
        [[0, 0, 0],
         [1, 1, 1],
         [0, 0, 0]])

    dilation_kern = np.array(
        [[0, 1, 0],
         [0, 1, 0],
         [0, 1, 0],
         [0, 1, 0],
         [0, 1, 0]])


@njit
def norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def normalise(S, min_level_db):
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
    freqs = librosa.core.mel_frequencies(fmin=minmax_freq[0],
                                         fmax=minmax_freq[1],
                                         n_mels=parameters.num_mel_bins)
    return np.argmin(abs(freqs - hz))


def mel_to_hz(mel_bin: int, dataset: SongDataset) -> int:
    """
    Returns the original frequency from a mel bin index.
    Requires a SongDataset object with set parameters.

    Args:
        mel_bin (int): Mel bin to convert
        dataset (SongDataset): SongDataset object

    Returns:
        int: Approximate original frequency in hertzs
    """
    freqs = librosa.core.mel_frequencies(fmin=dataset.parameters.lowcut,
                                         fmax=dataset.parameters.highcut,
                                         n_mels=dataset.parameters.num_mel_bins)
    return int(freqs[mel_bin])


def mels_to_hzs(dataset: SongDataset) -> np.ndarray[int]:
    """
    Returns the original frequencies from the mel bins used in a dataset.
    Requires a SongDataset object with set parameters.

    Args:
        dataset (SongDataset): SongDataset object

    Returns:
        np.ndarray[int]: Approximate original frequencies in hertzs
    """
    freqs = librosa.core.mel_frequencies(fmin=dataset.parameters.lowcut,
                                         fmax=dataset.parameters.highcut,
                                         n_mels=dataset.parameters.num_mel_bins)
    return freqs.astype(int)


def get_peak_freqs(dataset: SongDataset,
                   spectrograms: np.ndarray,
                   melscale: bool = True,
                   threshold: float = 0.3):

    minfreq = dataset.parameters.lowcut
    min_db = - dataset.parameters.top_dB

    if melscale:
        hz_freq = mels_to_hzs(dataset)
        result = np.array(
            [hz_freq[np.argmax(np.max(w, axis=1))]
             if(max(np.max(w, axis=1)) > min_db * (1 - threshold)) else -1
             for w in spectrograms])

        return result

    else:
        return np.array([minfreq + np.argmax(np.max(w, axis=1))
                         for w in spectrograms])
        # REVIEW did not test for melscale = False