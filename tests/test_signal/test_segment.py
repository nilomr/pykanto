# ─── DEPENDENCIES ─────────────────────────────────────────────────────────────

from pathlib import Path

import numpy as np
import pkg_resources
import pytest
import soundfile as sf
from pykanto.signal.segment import (
    ReadWav,
    SegmentMetadata,
    segment_files_parallel,
    segment_is_valid,
)
from pykanto.utils.compute import flatten_list
from pykanto.utils.custom import parse_sonic_visualiser_xml
from pykanto.utils.paths import ProjDirs, get_file_paths, get_wavs_w_annotation
from pykanto.utils.types import Annotation, AudioAnnotation, Metadata

# ──── SETTINGS ────────────────────────────────────────────────────────────────

DATASET_ID = "STORM-PETREL"

# ──── FIXTURES ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def DIRS():
    DATA_PATH = Path(pkg_resources.resource_filename("pykanto", "data"))
    PROJECT = Path(DATA_PATH).parent
    RAW_DATA = DATA_PATH / "raw" / DATASET_ID
    DIRS = ProjDirs(PROJECT, RAW_DATA, DATASET_ID, mkdir=True)
    return DIRS


@pytest.fixture()
def datapaths(DIRS):
    annotation_paths = get_file_paths(DIRS.RAW_DATA, [".xml"])
    wav_filepaths = get_file_paths(DIRS.RAW_DATA, [".wav"])
    datapaths = get_wavs_w_annotation(wav_filepaths, annotation_paths)
    return datapaths


@pytest.fixture()
def metadata():
    return Annotation(
        ID="Big_Bird",
        start_times=[1, 3, 5],
        durations=[48000, 48000, 48000],
        end_times=[2, 4, 6],
        lower_freq=[2000, 3000, 3000],
        upper_freq=[5000, 4000, 6000],
        label=["A", "A", "B"],
        annotation_file=Path("root") / "ann.xml",
        sample_rate=44100,
        bit_rate=352,
        length_s=3.2,
        source_wav=Path("root") / "ann.wav",
    )


@pytest.fixture()
def audio_section():
    return np.random.uniform(-1, 1, 44100)


@pytest.fixture()
def files_to_segment(DIRS):
    # Get files to segment and segment them
    wav_filepaths, xml_filepaths = [
        get_file_paths(DIRS.RAW_DATA, [ext]) for ext in [".wav", ".xml"]
    ]
    files_to_segment = get_wavs_w_annotation(wav_filepaths, xml_filepaths)
    return files_to_segment


# ──── TESTS ────────────────────────────────────────────────────────────────────


def test_ReadWav(datapaths):
    ReadWav_inst = ReadWav(datapaths[0][0])
    assert isinstance(ReadWav_inst.get_wav(), sf.SoundFile)
    assert isinstance(ReadWav_inst.get_metadata(), AudioAnnotation)
    assert set(AudioAnnotation.__annotations__) <= ReadWav_inst.as_dict().keys()


def test_SegmentMetadata(metadata, audio_section):
    segment = SegmentMetadata(
        metadata, audio_section, 2, 48000, Path("out") / "test.wav"
    )
    assert isinstance(segment.get_metadata(), Metadata)
    assert set(Metadata.__annotations__) <= segment.as_dict().keys()


@pytest.mark.parametrize(
    (
        "max_amplitude, i, integer_format, min_duration, min_freqrange, "
        "min_amplitude, labels_to_ignore, expected"
    ),
    [
        (6000.0, 0, "PCM_16", 1, 500, 5000, ["A"], False),
        (6000.0, 1, "PCM_16", 1, 500, 5000, ["A"], False),
        (6000.0, 1, "PCM_16", 1, 500, 5000, ["B"], True),
        (6000.0, 1, "PCM_16", 2, 500, 5000, ["B"], False),
        (6000.0, 2, "PCM_16", 1, 500, 5000, [" "], True),
        (6000.0, 2, "PCM_16", 1, 4000, 5000, [" "], False),
        (3000.0, 1, "PCM_16", 1, 500, 5000, ["B"], True),
    ],
)
def test_segment_is_valid(
    metadata,
    max_amplitude,
    i,
    integer_format,
    min_duration,
    min_freqrange,
    min_amplitude,
    labels_to_ignore,
    expected,
):
    assert (
        segment_is_valid(
            metadata,
            max_amplitude,
            i,
            integer_format,
            min_duration,
            min_freqrange,
            min_amplitude,
            labels_to_ignore,
        )
        == expected
    )


def test_segment_files_parallel(files_to_segment, DIRS):
    segment_files_parallel(
        files_to_segment,
        DIRS,
        resample=22050,
        parser_func=parse_sonic_visualiser_xml,
        min_duration=0.1,
        min_freqrange=100,
        labels_to_ignore=["NOISE"],
    )

    outfiles = [
        get_file_paths(DIRS.SEGMENTED, [ext]) for ext in [".wav", ".JSON"]
    ]
    outfiles = flatten_list(outfiles)
    assert all([f.stat().st_size for f in outfiles]) > 0
