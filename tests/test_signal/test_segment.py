# ─── DEPENDENCIES ─────────────────────────────────────────────────────────────

from pathlib import Path

import numpy as np
import pkg_resources
import pytest
import soundfile as sf
from pykanto.signal.segment import ReadWav, SegmentMetadata, segment_is_valid
from pykanto.utils.paths import ProjDirs, get_file_paths, get_wavs_w_annotation
from pykanto.utils.types import Annotation, AudioAnnotation, Metadata

# ──── FIXTURES ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def DIRS():
    DATA_PATH = Path(pkg_resources.resource_filename("pykanto", "data"))
    PROJECT = Path(DATA_PATH).parent
    RAW_DATA = DATA_PATH / "raw"
    DIRS = ProjDirs(PROJECT, RAW_DATA, DATASET_ID="test", mkdir=True)
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


# min_duration: float = .5
# min_freqrange: int = 200
# resample: int = 22050
# labels_to_ignore: List[str] = ["FIRST", "first"]

# # Make sure output folders exists
# wav_outdir = makedir(DIRS.SEGMENTED / "WAV")
# json_outdir = makedir(DIRS.SEGMENTED / "JSON")

# class TestMeta:
#     def __init__(self):

#         self.metadata = Annotation(
#             ID=['segment_1', 'segment_2', 'segment_3'],
#             start_times=[1, 3, 5],
#             durations=[1, 1, 1],
#             end_times=[2, 4, 6],
#             lower_freq=[2000, 3000, 3000],
#             upper_freq=[5000, 4000, 6000],
#             label=['A', 'A', 'B'],
#             annotation_file=Path('root')/"ann.xml",
#             sample_rate=48000,
#             bit_rate=352,
#             length_s=3.2,
#             source_wav=Path('root')/"ann.wav"
#         )

#     def return_metadata(self):
#         return self.metadata

# # Extend the Annotation class to include your new attributes
# # E.g., a datetime attribute of type str.

# @attr.s
# class CustomAnnotation(Annotation):
#     datetime: str = attr.ib(validator=validators.instance_of(str))

# testmeta().return_metadata()

# class CustomTestMeta(TestMeta):
#     def return_metadata(self):
#         newkeys = {'datetime': self.metadata.datetime}
#         annotation = CustomAnnotation(**{**self.metadata.__dict__, **newkeys})
#         return annotation

# TestMeta = CustomTestMeta
