# ──── DESCRIPTION ──────────────────────────────────────────────────────────────

"""
A collection of functions used to create and manipulate spectrograms.
"""

# ──── IMPORTS ──────────────────────────────────────────────────────────────────

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Tuple

import librosa
import numba
import numpy as np
import ray
from numba import njit
from pykanto.signal.filter import dereverberate, hz_to_mel_lib, norm
from pykanto.utils.compute import (
    calc_chunks,
    dictlist_to_dict,
    flatten_list,
    get_chunks,
    print_parallel_info,
    to_iterator,
    with_pbar,
)
from pykanto.utils.io import makedir

if TYPE_CHECKING:
    from pykanto.dataset import KantoData

# ──── FUNCTIONS ────────────────────────────────────────────────────────────────


def save_melspectrogram(
    dataset: KantoData,
    key: str,
    dereverb: bool = True,
    bandpass: bool = True,
) -> Dict[str, Path]:
    """
    Computes and saves a melspectrogram as a numpy array
    using dataset parameters.

    Args:
        dataset (KantoData): A KantoData object.
        key (str): Reference of wav file to open.
        dereverb (bool, optional): Whether to apply dereverberation to
            the spectrogram. Defaults to True.
        bandpass (bool, optional): Whether to bandpass the spectrogram using
            the minimum and maximum frequencies of the audio segment's
            bounding box. Defaults to True.

    Returns:
        Tuple[str, Path]: Key and location of each spectrogram.
    """

    # Locate song metadata
    d_dict = dataset.metadata[key]
    file = d_dict["wav_file"]

    # load audio file with librosa
    dat, _ = librosa.load(file, sr=dataset.parameters.sr)

    # Compute mel spec
    spec = librosa.feature.melspectrogram(
        y=dat,
        sr=dataset.parameters.sr,
        n_fft=dataset.parameters.fft_size,
        win_length=dataset.parameters.window_length,
        hop_length=dataset.parameters.hop_length,
        n_mels=dataset.parameters.num_mel_bins,
        fmin=dataset.parameters.lowcut,
        fmax=dataset.parameters.highcut,
    )

    # To dB
    spec = librosa.amplitude_to_db(
        S=spec, top_db=dataset.parameters.top_dB, ref=np.max
    )

    if bandpass:
        # Mask anything outside frequency bounding box
        spec = _mask_melspec(dataset, d_dict, spec)

    if dereverb:
        spec = dereverberate(
            spec,
            echo_range=100,
            echo_reduction=0.6,
            hop_length=dataset.parameters.hop_length,
            sr=dataset.parameters.sr,
        )

    # Save spec
    file = Path(file)
    ID = dataset.metadata[key]["ID"]
    spec_out_dir: Path = dataset.DIRS.SPECTROGRAMS / (Path(file).stem + ".npy")
    makedir(spec_out_dir)
    np.save(spec_out_dir, spec)

    return {Path(file).stem: spec_out_dir}


@ray.remote
def _save_melspectrogram_r(
    dataset: KantoData, keys: List[str], **kwargs
) -> List[Dict[str, Path]]:
    """
    Helper of :func:`~pykanto.signal.spectrogram._save_melspectrogram_parallel`.
    """
    return [save_melspectrogram(dataset, key, **kwargs) for key in keys]


def _save_melspectrogram_parallel(
    dataset: KantoData, keys: List[str], **kwargs
) -> Dict[str, Path]:
    """
    Parallel implementation of
    :func:`~pykanto.signal.spectrogram.save_melspectrogram`.
    """

    # Calculate and make chunks
    n = len(keys)
    if not n:
        raise KeyError(
            "No sound file keys were passed to " "save_melspectrogram."
        )
    chunk_info = calc_chunks(n, verbose=dataset.parameters.verbose)
    chunk_length, n_chunks = chunk_info[3], chunk_info[2]
    chunks = get_chunks(keys, chunk_length)
    if dataset.parameters.verbose:
        print_parallel_info(n, "new audio files", n_chunks, chunk_length)

    # Copy dataset to local object store
    dataset_ref = ray.put(dataset)

    # Distribute with ray
    obj_ids = [
        _save_melspectrogram_r.remote(dataset_ref, i, **kwargs) for i in chunks
    ]
    pbar = {"desc": "Preparing spectrograms", "total": n_chunks}
    spec_locs = [obj_id for obj_id in with_pbar(to_iterator(obj_ids), **pbar)]

    # Flatten and return
    return dictlist_to_dict(flatten_list(spec_locs))


def retrieve_spectrogram(nparray_dir: Path) -> np.ndarray:
    """
    Loads an spectrogram that was saved as a numpy array.

    Args:
        nparray_dir (Path): Path to the numpy array.

    Returns:
        np.ndarray: Spectrogram.
    """
    return np.load(nparray_dir)


def _mask_melspec(
    dataset: KantoData, d_dict: Dict[str, Any], mel_spectrogram: np.ndarray
) -> np.ndarray:
    """
    Private method. Frequency-mask 'bandpass' a melspectrogram using
    frequency bounds contained in a vocalisation's metadata.

    Args:
        dataset (KantoData): Dataset with parameters object.
        d_dict (Dict[str, Any]): Vocalisation metadata as a dictionary.
        mel_spectrogram (np.ndarray): Melspectrogram to mask.

    Returns:
        np.ndarray: Masked spectrogram.
    """

    freq_lims_mel = [
        hz_to_mel_lib(
            hz,
            (dataset.parameters.lowcut, dataset.parameters.highcut),
            dataset.parameters,
        )
        for hz in (d_dict["lower_freq"], d_dict["upper_freq"])
    ]

    mel_spectrogram[: freq_lims_mel[0]] = -dataset.parameters.top_dB
    mel_spectrogram[freq_lims_mel[1] : -1] = -dataset.parameters.top_dB

    return mel_spectrogram


@njit
def pad_spectrogram(spectrogram: np.ndarray, pad_length: int) -> np.ndarray:
    """
    Centre pads a spectrogram to a given length.

    Args:
        spectrogram (np.ndarray): Spectrogram to pad.
        pad_length (int): Full length of padded spectrogram

    Returns:
        np.ndarray: Padded spectrogram
    """
    spec_shape = np.shape(spectrogram)
    excess_needed = pad_length - spec_shape[1]
    pad_left = int(np.floor(float(excess_needed) / 2))
    pad_right = int(np.ceil(float(excess_needed) / 2))
    padded_spec = np.full((spec_shape[0], pad_length), np.min(spectrogram))
    padded_spec[:, pad_left : pad_length - pad_right] = spectrogram
    return padded_spec


@njit
def crop_spectrogram(
    spectrogram: np.ndarray, crop_x: int = 0, crop_y: int = 0
) -> np.ndarray:
    """
    Centre crops an spectrogram to given dimensions.

    Args:
        spectrogram (np.ndarray): Spectrogram to crop.
        crop_x (int, optional): Final x length, > 0. Defaults to 0 (no crop).
        crop_y (int, optional): Final y length, > 0. Defaults to 0 (no crop).

    Returns:
        np.ndarray: Cropped spectrogram
    """
    y, x = spectrogram.shape
    crop_y = y if crop_y == 0 else crop_y
    start_x = x // 2 - crop_x // 2
    start_y = y // 2 - crop_y // 2
    return spectrogram[start_y : start_y + crop_y, start_x : start_x + crop_x]


@njit
def cut_or_pad_spectrogram(spectrogram: np.ndarray, length: int) -> np.ndarray:
    """
    Cut or pad a spectrogram to be a given length.

    Args:
        spectrogram (np.ndarray): Spectrogram to cut or pad.
        length (int): Final desired lenght, in frames

    Returns:
        np.ndarray: Cut or padded spectrogram.
    """
    if spectrogram.shape[1] > length:
        return crop_spectrogram(spectrogram, crop_x=length)
    elif spectrogram.shape[1] < length:
        return pad_spectrogram(spectrogram, length)
    else:
        return spectrogram


@njit
def get_unit_spectrograms(
    spectrogram: np.ndarray,
    onsets: np.ndarray,
    offsets: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Get an array containing spectrograms for every unit in a given song.

    Args:
        spectrogram (np.ndarray): Spectrogram for a single song.
        onsets (np.ndarray): Unit onsets, in seconds.
        offsets (np.ndarray): Unit offsets, in seconds.
        sr (int): Sampling rate, in Hz
        hop_length (int): Hop length, in frames.

    Returns:
        np.ndarray: An array of arrays, one per unit.
    """

    units = []
    # Seconds to frames
    onsets = onsets * sr / hop_length
    offsets = offsets * sr / hop_length
    for on, off in zip(onsets.astype(np.int32), offsets.astype(np.int32)):
        unit = spectrogram[:, on:off]
        units.append(unit)

    return units


def get_vocalisation_units(
    dataset: KantoData, key: str, song_level: bool = False
) -> Dict[str, np.ndarray | List[np.ndarray]]:
    """
    Returns spectrogram representations of the units present in a vocalisation
    (e.g. in a song) or their average.

    Args:
        dataset (KantoData): A KantoData object.
        key (str): Single vocalisation locator (key).
        song_level (bool, optional): Whether to return average of all units.
            Defaults to False.

    Returns:
        Dict[str, np.ndarray | List[np.ndarray]]: Dictionary with key and
            average of all its units if `song_level` = True, padded to
            maximum duration. If `song_level` = False returns a Dict with
            key and a list of unit spectrograms, without padding.
    """

    # Get spectrogram and segmentation information
    if "onsets" not in dataset.data.columns:
        raise KeyError(
            "'onsets':  This vocalisation has not yet been segmented."
        )
    spectrogram = retrieve_spectrogram(dataset.files.at[key, "spectrogram"])
    onsets, offsets = [dataset.data.at[key, i] for i in ["onsets", "offsets"]]

    # Get spectrogram for each unit
    unit_spectrograms = get_unit_spectrograms(
        spectrogram,
        onsets,
        offsets,
        sr=dataset.parameters.sr,
        hop_length=dataset.parameters.hop_length,
    )

    if song_level:
        # Pad unit spectrograms to max duration and return mean
        max_frames = max([unit.shape[1] for unit in unit_spectrograms])
        padded_units = [
            pad_spectrogram(unit, max_frames) for unit in unit_spectrograms
        ]
        avg_unit = norm(np.mean(padded_units, axis=0))
        return {key: avg_unit}
    else:
        return {key: unit_spectrograms}


def get_indv_units(
    dataset: KantoData,
    keys: List[str],
    ID: str,
    pad: bool = True,
    song_level: bool = False,
) -> Dict[str, Path]:
    """
    Returns a spectrogram representations of the units or the average of
    the units present in the vocalisations of an ID in the dataset.
    Saves the data as pickled dictionary, returns its location.

    Args:
        dataset (KantoData): Source dataset
        keys (List[str]): List of keys belonging to an ID
        ID (str): ID ID
        pad (bool, optional): Whether to pad spectrograms to the maximum lenght.
            Defaults to True.
        song_level (bool, optional): Whether to return the average of all units.
            Defaults to False.
    Returns:
        Dict[str, Path]: ID and location of its pickled dictionary.
    """

    units = [
        get_vocalisation_units(dataset, key, song_level=song_level)
        for key in keys
    ]
    units = dictlist_to_dict(units)

    if pad:
        if song_level:
            max_frames = max([unit.shape[1] for unit in units.values()])
            units = {
                key: pad_spectrogram(spec, max_frames)
                for key, spec in units.items()
            }
        else:
            max_frames = max(
                [unit.shape[1] for ls in units.values() for unit in ls]
            )
            units = {
                key: [pad_spectrogram(spec, max_frames) for spec in ls]
                for key, ls in units.items()
            }

    # Save dataset
    out_dir = dataset.DIRS.SPECTROGRAMS / (
        f"{ID}_avg_units.p" if song_level else f"{ID}_units.p"
    )
    makedir(out_dir)
    pickle.dump(units, open(out_dir, "wb"))
    return {ID: out_dir}


def get_indv_units_parallel(
    dataset: KantoData,
    pad: bool = True,
    song_level: bool = False,
    num_cpus: float | None = None,
) -> Dict[str, Dict[str, Path]]:
    """
    Parallel implementation of
    :func:`~pykanto.signal.spectrogram.get_indv_units`.
    """
    indv_dict = {
        indv: dataset.data[dataset.data["ID"] == indv].index
        for indv in set(dataset.data["ID"])
    }

    # Calculate and make chunks
    n = len(indv_dict)
    if not n:
        raise KeyError("No file keys were passed to " "get_indv_units.")
    chunk_info = calc_chunks(
        n, n_workers=num_cpus, verbose=dataset.parameters.verbose
    )
    chunk_length, n_chunks = chunk_info[3], chunk_info[2]
    chunkeys = get_chunks(list(indv_dict), chunk_length)
    chunks = [
        {k: v for k, v in indv_dict.items() if k in chunk} for chunk in chunkeys
    ]
    if dataset.parameters.verbose:
        print_parallel_info(n, "new audio files", n_chunks, chunk_length)

    # Copy dataset to local object store
    dataset_ref = ray.put(dataset)

    # Distribute with ray
    @ray.remote(num_cpus=num_cpus)
    def _get_indv_units_r(*args, **kwargs):
        return get_indv_units(*args, **kwargs)

    obj_ids = []
    for chunk in chunks:
        for ID, keys in chunk.items():
            obj_ids.append(
                _get_indv_units_r.remote(
                    dataset_ref, keys, ID, pad=pad, song_level=song_level
                )
            )

    pbar = {
        "desc": "Calculating and saving unit spectrograms",
        "total": n_chunks,
    }
    dic_locs = [obj_id for obj_id in with_pbar(to_iterator(obj_ids), **pbar)]

    # Flatten and return as a dictionary
    return dictlist_to_dict(dic_locs)


@njit
def window(spectrogram: np.ndarray, wlength: int) -> Iterator[np.ndarray]:
    """
    Extract windows of length 'wlength' from a spectrogram. Jitted.

    Args:
        spectrogram (np.ndarray): Spectrogram to window.
        wlength (int): Desired window length.

    Yields:
        Iterator[np.ndarray]: A single window.
    """

    y = spectrogram.shape[1]
    for j in range(y):
        ymin = j
        ymax = j + wlength if j + wlength <= y else y
        if ymax == y:
            break
        yield spectrogram[:, ymin:ymax]


@njit
def extract_windows(
    spectrograms: numba.typed.List[np.ndarray], wlength: int
) -> Tuple[numba.typed.List[np.ndarray], List[int]]:
    """
    Extract windows from multiple spectrograms. Jitted.

    Args:
        spectrograms (numba.typed.List[np.ndarray]): Spectrogram to window.
        wlength (int): Desired window length.

    Returns:
        Tuple[numba.typed.List[np.ndarray], List[int]]: Contains a list with the
        resulting windows and a list with the window counts per spectrogram.
    """
    windows = numba.typed.List()
    n_windows = []
    for spec in spectrograms:
        for i, w in enumerate(window(spec, wlength)):
            windows.append(w)
        n_windows.append(i + 1)
    return windows, n_windows


@njit
def flatten_spectrograms(
    windows: numba.typed.List[np.ndarray],
) -> numba.typed.List[np.ndarray]:
    """
    Return a numba typed list containing the 2d array collapsed into one
    dimension. Jitted.

    Args:
        windows (numba.typed.List[np.ndarray]): List of 2d spectrograms.

    Returns:
        numba.typed.List[np.ndarray]: The same list, now containing 1d
        spectrograms.
    """

    return numba.typed.List([w.flatten() for w in windows])
