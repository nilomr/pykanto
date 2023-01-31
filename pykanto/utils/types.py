# ─── DESCRIPTION ──────────────────────────────────────────────────────────────

"""
Pykanto typing module: provides a) custom type hints not already covered in other
modules and b) classes whose main purpose is data storage and/or type checking.
"""

# ─── DEPENDENCIES ─────────────────────────────────────────────────────────────
from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from typing import Any, List, Protocol

import attr
from attr import validators

# ──── DEFINITIONS ──────────────────────────────────────────────────────────────


def is_list_of_int(instance, attribute, f):
    """
    Validator for attr.s decorator.
    """
    if not isinstance(f, list) or not all(isinstance(x, int) for x in f):
        raise TypeError(
            f"Attrtibute '{attribute.name}' must be of type: List[int]"
        )


def is_list_of_str(instance, attribute, f):
    """
    Validator for attr.s decorator.
    """
    if not isinstance(f, list) or not all(isinstance(x, str) for x in f):
        raise TypeError(
            f"Attrtibute '{attribute.name}' must be of type: List[str]"
        )


def f_exists(instance, attribute, f: Path):
    """
    File exists validator for attr.s decorator.
    """
    if not f.exists():
        raise FileNotFoundError(f)


class AttrProto(Protocol):
    """Inherit to silence attr related errors in Pyrigh"""

    def __init__(self, **kwargs: Any) -> None:
        ...


@attr.s
class ValidDirs(AttrProto):
    """
    Type check user input before instantiating main ProjDirs class.
    """

    PROJECT: Path = attr.ib(validator=[validators.instance_of(Path), f_exists])
    RAW_DATA: Path = attr.ib(validator=[validators.instance_of(Path), f_exists])
    DATASET_ID: str = attr.ib(validator=validators.instance_of(str))


@attr.s
class SegmentAnnotation(AttrProto):
    """
    Type-checks and stores annotations necessary to segment regions of interest
    present in an audio file (e.g. songs or song bouts).
    """

    ID: str = attr.ib(validator=validators.instance_of(str))
    start_times: List[int] = attr.ib(validator=is_list_of_int)
    durations: List[int] = attr.ib(validator=is_list_of_int)
    end_times: List[int] = attr.ib(validator=is_list_of_int)
    lower_freq: List[int] = attr.ib(validator=is_list_of_int)
    upper_freq: List[int] = attr.ib(validator=is_list_of_int)
    label: List[str] = attr.ib(validator=is_list_of_str)
    annotation_file: Path = attr.ib(validator=validators.instance_of(Path))


@attr.s
class AudioAnnotation(AttrProto):
    """
    Type-checks and stores audio metadata.
    """

    sample_rate: int = attr.ib(validator=validators.instance_of(int))
    bit_rate: int = attr.ib(validator=validators.instance_of(int))
    length_s: float = attr.ib(validator=validators.instance_of(float))
    source_wav: Path = attr.ib(validator=validators.instance_of(Path))


@attr.s
class Annotation(SegmentAnnotation, AudioAnnotation):
    """
    Combines segment annotations and audio metadata.
    """

    pass


@attr.s
class Metadata(AudioAnnotation):
    """
    Type-checks and stores metadata for ONE audio segment
    (e.g. a song or song bout). Type checks ensure that
    instances of this class are JSON-serializable as a dictionary.
    """

    ID: str = attr.ib(validator=validators.instance_of(str))
    label: str = attr.ib(validator=validators.instance_of(str))
    start: int = attr.ib(validator=validators.instance_of(int))
    end: int = attr.ib(validator=validators.instance_of(int))
    lower_freq: int = attr.ib(validator=validators.instance_of(int))
    upper_freq: int = attr.ib(validator=validators.instance_of(int))
    max_amplitude: float = attr.ib(validator=validators.instance_of(float))
    min_amplitude: float = attr.ib(validator=validators.instance_of(float))
    source_wav: str = attr.ib(validator=validators.instance_of(str))
    annotation_file: str = attr.ib(validator=validators.instance_of(str))
    wav_file: str = attr.ib(validator=validators.instance_of(str))


Chunkinfo_ = namedtuple(
    "Chunkinfo",
    ["n_workers", "len_iterable", "n_chunks", "chunksize", "last_chunk"],
)


class Chunkinfo(Chunkinfo_):
    """
    Stores information about a chunked iterable.
    """

    n_workers: int | float
    len_iterable: int
    n_chunks: float
    chunksize: float
    last_chunk: float
