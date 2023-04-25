# ─── DESCRIPTION ──────────────────────────────────────────────────────────────

"""Segment audio files and find vocalisation units in spectrograms."""

# ─── DEPENDENCIES ─────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Tuple
from xml.etree import ElementTree

import attr
import audio_metadata as audiometa
import librosa
import librosa.display
import numba
import numpy as np
import psutil
import ray
import soundfile as sf
from pykanto.signal.filter import (
    dereverberate,
    dereverberate_jit,
    gaussian_blur,
    kernels,
    norm,
    normalise,
)
from pykanto.signal.spectrogram import retrieve_spectrogram
from pykanto.utils.compute import (
    calc_chunks,
    flatten_list,
    get_chunks,
    print_parallel_info,
    timing,
    to_iterator,
    with_pbar,
)
from pykanto.utils.custom import parse_sonic_visualiser_xml
from pykanto.utils.io import makedir
from pykanto.utils.types import (
    Annotation,
    AudioAnnotation,
    Metadata,
    SegmentAnnotation,
)
from scipy import ndimage
from skimage.exposure import equalize_hist
from skimage.filters.rank import median
from skimage.morphology import dilation, disk, erosion
from skimage.util import img_as_ubyte

if TYPE_CHECKING:
    from pykanto.dataset import KantoData

from pykanto.utils.paths import ProjDirs, get_file_paths

# ──── DIVIDING RAW FILES INTO SEGMENTS ─────────────────────────────────────────


# ──── CLASSES AND CLASS METHODS ────


class ReadWav:
    """
    Reads a wav file and its metadata.

    Note:
        You can extend this class to read in metadata from the wav file that is
        specific to your research, e.g. the recorder device ID, or time
        information.

    Examples:
        TODO
    """

    def __init__(self, wav_dir: Path) -> None:
        self.wav_dir = wav_dir
        """Location of wav file."""
        self._load_wav()
        self._load_metadata()

    def _load_wav(self) -> None:
        """
        Opens a wav sound file.

        Raises:
            ValueError: The file is not 'seekable'.
        """
        wavfile = sf.SoundFile(str(self.wav_dir))

        if not wavfile.seekable():
            raise ValueError("Cannot seek through this file", self.wav_dir)

        self.wavfile = wavfile
        self.nframes = self.wavfile.seek(0, sf.SEEK_END)

    def _load_metadata(self) -> None:
        """
        Loads available metadata from wavfile; builds a AudioAnnotation object.
        """

        self.all_metadata = audiometa.load(self.wav_dir)
        """All available metadata for this audio clip."""

        self.metadata = AudioAnnotation(
            sample_rate=self.wavfile.samplerate,
            bit_rate=self.all_metadata["streaminfo"].bitrate,
            length_s=self.nframes / self.wavfile.samplerate,
            source_wav=self.wav_dir,
        )
        """Selected metadata to be used later"""

    def get_wav(self) -> sf.SoundFile:
        """
        Returns the wavfile.

        Returns:
            sf.SoundFile: Seekable wavfile.
        """
        return self.wavfile

    def get_metadata(self) -> AudioAnnotation:
        """
        Returns metadata attached to wavfile as an AudioAnnotation object.

        Returns:
            AudioAnnotation: Wavfile metadata.
        """
        return self.metadata

    def as_dict(self) -> Dict[str, Any]:
        """
        Returns metadata attached to wavfile as a dictionary.

        Returns:
            Dict[str, Any]: Wavfile metadata.
        """
        out = self.get_metadata()
        if isinstance(out, dict):
            return out
        else:
            return out.__dict__


class SegmentMetadata:
    """
    Consolidates segment metadata in a single Metadata object,
    which can then be saved as a standard .JSON file.

    You can extend this class to incorporate other metadata fields
    specific to your research (see the docs).

    """

    def __init__(
        self,
        metadata: Annotation,
        audio_section: np.ndarray,
        i: int,
        sr: int,
        wav_out: Path,
    ) -> None:
        """
        Consolidates segment metadata in a single Metadata object,
        which can then be saved as a standard .JSON file.

        Args:
            name (str): Segment identifier.
            metadata (Annotation): An object containing relevant metadata.
            audio_section (np.ndarray): Array containing segment audio data
                (to extract min/max amplitude).
            i (int): Segment index.
            sr (int): Sample rate.
            wav_out (Path): Path to segment wav file.

        Returns: None

        """

        self.all_metadata = metadata
        """Attribute containing all available metadata"""

        self.index: int = i
        """Index of 'focal' segment"""

        self._build_metadata(metadata, audio_section, i, sr, wav_out)

    def _build_metadata(
        self,
        metadata: Annotation,
        audio_section: np.ndarray,
        i: int,
        sr: int,
        wav_out: Path,
    ) -> None:
        """
        Consolidates segment metadata in a single dictionary,
        which can then be saved as a standard .JSON file.

        Args:
            metadata (Dict[str, Any]): A dictionary containing pertinent metadata.
            audio_section (np.ndarray): Array containing segment audio data
                (to extract min/max amplitude).
            i (int): Segment index.
            sr (int): Sample rate.
            wav_out (Path): Path to segment wav file.

        Returns: None

        """

        self.metadata = Metadata(
            ID=metadata.ID,
            label=metadata.label[i],
            sample_rate=sr,
            start=metadata.start_times[i],
            end=metadata.end_times[i],
            length_s=len(audio_section) / sr,
            lower_freq=metadata.lower_freq[i],
            upper_freq=metadata.upper_freq[i],
            max_amplitude=float(max(audio_section)),
            min_amplitude=float(min(audio_section)),
            bit_rate=metadata.bit_rate,
            source_wav=metadata.source_wav.as_posix(),
            annotation_file=metadata.annotation_file.as_posix(),
            wav_file=wav_out.as_posix(),
        )

        self.index = i

    def get_metadata(self) -> Metadata:
        """
        Get Metadata object.

        Returns:
            Metadata: Single-segment metadata.
        """
        return self.metadata

    def as_dict(self) -> Dict[str, Any]:
        """
        Returns Metadata object as a dictionary.

        Returns:
            Dict[str, Any]: Wavfile metadata.
        """
        out = self.get_metadata()
        if isinstance(out, dict):
            return out
        else:
            return out.__dict__


# ──── FUNCTIONS ────


def segment_file(
    wav_dir: Path,
    metadata_dir: Path,
    wav_outdir: Path,
    json_outdir: Path,
    resample: int | None = 22050,
    parser_func: Callable[
        [Path], SegmentAnnotation
    ] = parse_sonic_visualiser_xml,
    **kwargs,
):
    """
    Segments and saves audio segmets and their metadata from a single audio
    file, based on annotations provided in a separate 'metadata' file.

    Args:
        wav_dir (Path): Where is the wav file to be segmented?
        metadata_dir (Path): Where is the file containing its segmentation
            metadata?
        wav_outdir (Path): Where to save the resulting wav segments.
        json_outdir (Path): Where to save the resulting json metadata files.
        resample (int | None, optional): Whether to resample audio, and to what
            sample ratio. Defaults to 22050.
        parser_func (Callable[[Path], dict[str, Any]], optional):
            Function to parse your metadata format. Defaults to parse_sonic_visualiser_xml.
        **kwargs: Keyword arguments passed to
            :func:`~pykanto.signal.segment.segment_is_valid`.
    """

    # Read audio and metadata
    wav_object = ReadWav(wav_dir)
    wavfile, audio_metadata = wav_object.get_wav(), wav_object.as_dict()
    metadata = Annotation(
        **{**attr.asdict(parser_func(metadata_dir)), **audio_metadata}
    )
    # Then save segments
    save_segments(
        metadata, wavfile, wav_outdir, json_outdir, resample=resample, **kwargs
    )


def save_segments(
    metadata: Annotation,
    wavfile: sf.SoundFile,
    wav_outdir: Path,
    json_outdir: Path,
    resample: int | None = 22050,
    **kwargs,
) -> None:
    """
    Save segments present in a single wav file to new separate files along with
    their metadata.

    Args:
        metadata (Annotation): Annotation and file metadata for this wav file.
        wavfile (SoundFile): Seekable wav file.
        wav_outdir (Path): Where to save the resulting segmented wav files.
        json_outdir (Path): Where to save the resulting json metadata files.
        resample (int | None, optional): Whether to resample audio, and to what
            sample ratio. Defaults to 22050.
        **kwargs: Keyword arguments passed to
            :func:`~pykanto.signal.segment.segment_is_valid`.
    """

    n_segments = len(metadata.start_times)

    for i in range(n_segments):
        # Get segment frames
        wavfile.seek(metadata.start_times[i])
        audio_section: np.ndarray = wavfile.read(metadata.durations[i])

        # Collapse to mono if not already the case
        if len(audio_section.shape) == 2:
            audio_section: np.ndarray = librosa.to_mono(
                np.swapaxes(audio_section, 0, 1)
            )

        # Filter segments not matching inclusion criteria
        if not segment_is_valid(
            metadata,
            float(max(audio_section)),
            i,
            integer_format=str(wavfile.subtype),
            **kwargs,
        ):
            continue

        # Resample if necessary
        sr = metadata.sample_rate

        if resample:
            audio_section: np.ndarray = librosa.resample(
                y=audio_section,
                orig_sr=sr,
                target_sr=resample,
                res_type="kaiser_fast",
            )
            sr = resample

        # Both to disk under name:
        name: str = f"{metadata.ID}_{metadata.source_wav.stem}_{metadata.start_times[i]}"

        # Save .wav
        wav_out = wav_outdir / f"{name}.wav"
        sf.write(wav_out.as_posix(), audio_section, sr)

        # Save metadata .JSON
        segment_metadata = SegmentMetadata(
            metadata, audio_section, i, sr, wav_out
        ).as_dict()
        json_out = json_outdir / f"{name}.JSON"
        with open(json_out.as_posix(), "w") as f:
            print(json.dumps(segment_metadata, indent=2), file=f)


def segment_is_valid(
    metadata: Annotation,
    max_amplitude: float,
    i: int,
    integer_format: str = "PCM_16",
    min_duration: float = 0.01,
    min_freqrange: int = 10,
    min_amplitude: int = 0,
    labels_to_ignore: List[str] = ["NO", "NOISE"],
) -> bool:
    """
    Checks whether a segment of index i within a dictionary is a valid segment.

    Args:
        metadata (Annotation): Annotation object for a wav file.
        i (int): Segment index.
        min_duration (float, optional): Minimum duration of segment to
            consider valid, in seconds. Defaults to 0.01.
        min_freqrange (int, optional): Minimum frequency range of segment to
            consider valid, in Hertzs. Defaults to 10.
        labels_to_ignore (List[str], optional): Exclude any segments with these
            labels. Defaults to ["NO", "NOISE"].

    Returns:
        bool: Is this a valid segment?
    """

    min_frames = min_duration * metadata.sample_rate
    if integer_format == "PCM_16":
        scale = 32767
    elif integer_format == "PCM_25":
        scale = 8388607
    else:
        raise NotImplementedError(
            f"Integer format '{integer_format}'' not supported"
        )

    max_amplitude = max_amplitude * scale

    if (
        (metadata.durations[i] < min_frames)
        or (metadata.upper_freq[i] - metadata.lower_freq[i] < min_freqrange)
        or (metadata.label[i] in labels_to_ignore)
        or (max_amplitude < min_amplitude)
    ):
        return False
    else:
        return True


def segment_files(
    datapaths: List[Tuple[Path, Path]],
    wav_outdir: Path,
    json_outdir: Path,
    resample: int | None = 22050,
    parser_func: Callable[
        [Path], SegmentAnnotation
    ] = parse_sonic_visualiser_xml,
    pbar: bool = True,
    **kwargs,
) -> None:
    """
    Finds and saves audio segments and their metadata.
    Parallel version in :func:`~pykanto.signal.segment.segment_files_parallel`.
    Works well with large files (only reads one chunk at a time).

    Args:
        datapaths (List[Tuple[Path, Path]]): List of tuples with pairs of paths
            to raw data files and their annotation metadata files.
        wav_outdir (Path): Location where to save generated wav files.
        json_outdir (Path): Location where to save generated json metadata files.
        resample (int | None, optional): Whether to resample audio.
            Defaults to 22050.
        parser_func (Callable[[Path], dict[str, Any]], optional):
            Function to parse your metadata format.
            Defaults to parse_sonic_visualiser_xml.
        pbar (bool, optional): Wheter to print progress bar. Defaults to True.
        **kwargs: Keyword arguments passed to
            :func:`~pykanto.signal.segment.segment_is_valid`
    """
    if len(datapaths) == 0:
        raise IndexError("List must contain at least one tuple.", datapaths)
    elif isinstance(datapaths, tuple):
        datapaths = [datapaths]

    for wav_dir, metadata_dir in with_pbar(
        datapaths,
        desc="Finding and saving audio segments and their metadata",
        disable=False if pbar else True,
    ):
        try:
            segment_file(
                wav_dir,
                metadata_dir,
                wav_outdir,
                json_outdir,
                resample=resample,
                parser_func=parser_func,
                **kwargs,
            )
        except RuntimeError as e:
            print(f"Failed to export {wav_dir}: ", e)


@timing
def segment_files_parallel(
    datapaths: List[Tuple[Path, Path]],
    dirs: ProjDirs,
    resample: int | None = 22050,
    parser_func: Callable[
        [Path], SegmentAnnotation
    ] = parse_sonic_visualiser_xml,
    num_cpus: float | None = None,
    verbose: bool = True,
    **kwargs,
) -> None:
    """
    Finds and saves audio segments and their metadata.
    Parallel version of :func:`~pykanto.signal.segment.segment_files`.
    Works well with large files (only reads one chunk at a time).

    Note:
        Creates ["WAV", "JSON"] output subfolders in data/segmented/dataset.

    Args:
        datapaths (List[Tuple[Path, Path]]): List of tuples with pairs of paths
            to raw data files and their annotation metadata files.
        dirs (ProjDirs): Project directory structure.
        resample (int | None, optional): Whether to resample audio.
            Defaults to 22050.
        parser_func (Callable[[Path], SegmentAnnotation], optional):
            Function to parse your metadata format.
            Defaults to parse_sonic_visualiser_xml.
        num_cpus (float | None, optional): Number of cpus to use for parallel
            computing. Defaults to None (all available).
        verbose (bool, optional): Defaults to True
        **kwargs: Keyword arguments passed to
            :func:`~pykanto.signal.segment.segment_is_valid`
    """

    # Make sure output folders exists
    wav_outdir, json_outdir = [
        makedir(dirs.SEGMENTED / ext) for ext in ["WAV", "JSON"]
    ]

    # Calculate and make chunks
    n = len(datapaths)
    if not n:
        raise KeyError(
            "No file keys were passed to " "segment_song_into_units."
        )
    chunk_length, n_chunks = map(
        calc_chunks(n, verbose=verbose).__getitem__, [3, 2]
    )
    chunks = get_chunks(datapaths, chunk_length)
    print_parallel_info(n, "files", n_chunks, chunk_length)

    # Distribute with ray

    @ray.remote(num_cpus=num_cpus)
    def segment_files_r(*args, **kwargs):
        return segment_files(*args, **kwargs)

    obj_ids = [
        segment_files_r.remote(
            paths,
            wav_outdir,
            json_outdir,
            resample=resample,
            parser_func=parser_func,
            pbar=False,
            **kwargs,
        )
        for paths in chunks
    ]
    pbar = {
        "desc": "Finding and saving audio segments and their metadata",
        "total": n_chunks,
    }
    [obj_id for obj_id in with_pbar(to_iterator(obj_ids), **pbar)]


def get_segment_info(
    RAW_DATA_DIR: Path,
    min_duration: float,
    min_freqrange: int,
    ignore_labels: List[str] = ["FIRST", "first"],
) -> Dict[str, List[float]]:
    """
    Get a summary of all segments present in a directory. Works for .xml files
        output by Sonic Visualiser.

    Args:
        RAW_DATA_DIR (Path): Folder to check, normally DATA_DIR / "raw" / YEAR
        min_duration (float): Minimum duration for a segment to be
            considered (in seconds)
        min_freqrange (int): Minimum frequency range for a segment to be
            considered (in hertz)
        ignore_labels (List[str], optional): Ignore segments with these labels.
            Defaults to ["FIRST", "first"].
    Returns:
        Dict[str, List[float]]: Lists of segment durations, in seconds
    """

    # TODO: Make it work with any file type (by passing a custom parser
    # function)

    XML_LIST = get_file_paths(RAW_DATA_DIR, [".xml"])
    cnt = 0
    noise_cnt = 0
    signal_cnt = 0
    noise_lengths: List[float] = []
    signal_lengths: List[float] = []

    for XML_FILEDIR in XML_LIST:
        root = ElementTree.parse(XML_FILEDIR).getroot()
        sr = int(root.findall("data/model")[0].get("sampleRate"))
        min_frames = min_duration * sr

        # iterate over segments and save them (+ metadata)
        for segment in root.findall("data/dataset/point"):
            seg_nframes = float(segment.get("duration"))
            # Ignore very short segments
            if seg_nframes < min_frames:
                continue
            # Also ignore segments that have very narroy bandwidth
            if float(segment.get("extent")) < min_freqrange:
                continue
            # Ignore first segments
            if segment.get("label") in ignore_labels:
                continue
            else:
                cnt += 1
                if segment.get("label") in ["NOISE", "noise"]:
                    noise_cnt += 1
                    noise_lengths.append(seg_nframes / sr)
                else:
                    signal_cnt += 1
                    signal_lengths.append(seg_nframes / sr)

    print(
        f"There are {cnt} segments in {RAW_DATA_DIR}, of which {signal_cnt} are "
        f"songs and {noise_cnt} are noise samples. Returning a dictionary "
        "containing lists of segment durations."
    )

    return {"signal_lengths": signal_lengths, "noise_lengths": noise_lengths}


# ──── SEGMENTING UNITS PRESENT IN A SEGMENT ────────────────────────────────────


def find_units(
    dataset: KantoData, spectrogram: np.ndarray
) -> Tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Segment a given spectrogram array into its units. For convenience,
    parameters are defined in a KantoData class instance (class Parameters).
    Based on Tim Sainburg's
    `vocalseg <https://github.com/timsainb/vocalization-segmentation/>`_ code.


    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple(onsets, offsets)
        None: Did not find any units matching given criteria.
    """

    params = dataset.parameters
    envelope_is_good = False
    params.hop_length_ms = params.sr / params.hop_length

    # Loop through thresholds, lowest first
    for min_level_dB in np.arange(
        -params.top_dB, params.max_dB, params.dB_delta
    ):
        # Threshold spectrogram
        spec = norm(normalise(spectrogram, min_level_db=min_level_dB))
        spec = spec - np.median(spec, axis=1).reshape((len(spec), 1))
        spec[spec < 0] = 0

        # Calculate and normalise the amplitude envelope
        envelope = np.max(spec, axis=0) * np.sqrt(np.mean(spec, axis=0))
        envelope = envelope / np.max(envelope)

        # Get onsets and offsets (sound and silence)
        onsets, offsets = (
            onsets_offsets(envelope > params.silence_threshold)
            / params.hop_length_ms
        )
        onsets_sil, offsets_sil = (
            onsets_offsets(envelope <= params.silence_threshold)
            / params.hop_length_ms
        )

        # Check results and return or continue
        if len(onsets_sil) > 0:
            # Get longest silences and vocalizations
            max_silence_len = np.max(offsets_sil - onsets_sil)
            max_unit_len = np.max(offsets - onsets)
            # Can this be considered a bout?
            if (
                max_silence_len > params.min_silence_length
                and max_unit_len < params.max_unit_length
            ):
                envelope_is_good = True
                break

    if not envelope_is_good:
        return None, None  # REVIEW
    else:
        # threshold out short syllables
        length_mask = (offsets - onsets) >= params.min_unit_length
        return onsets[length_mask], offsets[length_mask]


def onsets_offsets(signal: np.ndarray) -> np.ndarray:
    """
    Labels features in array as insets and offsets.
    Based on Tim Sainburg's
    `vocalseg <https://github.com/timsainb/vocalization-segmentation/>`_.

    Args:
        signal (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    units, nunits = ndimage.label(signal)
    if nunits == 0:
        return np.array([[0], [0]])
    onsets, offsets = np.array(
        [
            np.where(units == unit)[0][np.array([0, -1])] + np.array([0, 1])
            for unit in np.unique(units)
            if unit != 0
        ]
    ).T
    return np.array([onsets, offsets])


def segment_song_into_units(
    dataset: KantoData, key: str
) -> Tuple[str, np.ndarray, np.ndarray] | None:
    """
    Find amplitude-differentiable units in a given vocalisation after applying a
    series of morphological transformations to reduce noise.

    Args:
        dataset (KantoData): Datset to use.
        key (str): _description_

    Returns:
        Tuple[str, np.ndarray, np.ndarray] | None: _description_
    """

    mel_spectrogram = retrieve_spectrogram(dataset.files.at[key, "spectrogram"])

    # TODO@nilomr #9 Jitted version of dereverberate() now causes ray workers to crash
    # dereverberate_jit = numba.njit(dereverberate)

    mel_spectrogram_d = dereverberate(
        mel_spectrogram,
        echo_range=100,
        echo_reduction=3,
        hop_length=dataset.parameters.hop_length,
        sr=dataset.parameters.sr,
    )
    mel_spectrogram_d = img_as_ubyte(norm(mel_spectrogram_d))

    img_eq = equalize_hist(mel_spectrogram)
    img_med = median(img_as_ubyte(img_eq), disk(2))
    img_eroded = erosion(img_med, kernels.erosion_kern)
    img_dilated = dilation(img_eroded, kernels.dilation_kern)
    img_dilated = dilation(img_dilated, kernels.erosion_kern)

    img_norm = equalize_hist(img_dilated)

    img_inv = np.interp(
        img_norm,
        (img_norm.min(), img_norm.max()),
        (-dataset.parameters.top_dB, 0),
    )
    img_gauss = gaussian_blur(img_inv.astype(float), 3)

    img_gauss_d = dereverberate(
        img_gauss,
        echo_range=100,
        echo_reduction=1,
        hop_length=dataset.parameters.hop_length,
        sr=dataset.parameters.sr,
    )

    onsets, offsets = find_units(dataset, img_gauss_d)
    if onsets is None or offsets is None:
        warnings.warn(
            f"No units found in {key}. "
            "This segment will be dropped from the dataset."
        )
        return None
    return key, onsets, offsets


def segment_song_into_units_parallel(
    dataset: KantoData, keys: Iterable[str], **kwargs
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """See save_melspectrogram"""

    # Calculate and make chunks
    n = len(keys)
    if not n:
        raise KeyError(
            "No file keys were passed to " "segment_song_into_units."
        )
    chunk_info = calc_chunks(n, verbose=dataset.parameters.verbose)
    chunk_length, n_chunks = chunk_info[3], chunk_info[2]
    chunks = get_chunks(keys, chunk_length)
    if dataset.parameters.verbose:
        print_parallel_info(n, "vocalisations", n_chunks, chunk_length)

    # Distribute with ray
    @ray.remote(num_cpus=dataset.parameters.num_cpus, num_gpus=0)
    def _segment_song_into_units_r(dataset, keys, **kwargs):
        return [segment_song_into_units(dataset, key, **kwargs) for key in keys]

    # Copy dataset to local object store
    dataset_ref = ray.put(dataset)

    obj_ids = [
        _segment_song_into_units_r.remote(dataset_ref, i, **kwargs)
        for i in chunks
    ]
    pbar = {"desc": "Finding units in vocalisations", "total": n_chunks}
    units = [obj_id for obj_id in with_pbar(to_iterator(obj_ids), **pbar)]

    # Flatten and return
    return flatten_list(units)


def drop_zero_len_units(
    dataset: KantoData, onsets: np.ndarray, offsets: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Removes onset/offset pairs which (under this dataset's spectrogram parameter
    combination) would result in a unit of length zero.

    Args:
        dataset (KantoData): KantoData instance containing parameters.
        onsets (np.ndarray): In seconds
        offsets (np.ndarray): In seconds

    Returns:
        Tuple[np.ndarray, np.ndarray]: Remaining onsets and offsets
    """

    durations_s = offsets - onsets
    mindur_frames = np.floor(
        durations_s * dataset.parameters.sr / dataset.parameters.hop_length
    )

    mask = np.ones(onsets.size, dtype=bool)
    mask[np.where(mindur_frames == 0)] = False

    return onsets[mask], offsets[mask]
