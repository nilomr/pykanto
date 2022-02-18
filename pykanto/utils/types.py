# ─── DESCRIPTION ──────────────────────────────────────────────────────────────

"""
Pykanto typing module: provides custom type hints not already covered in other
modules.
"""

# ─── DEPENDENCIES ─────────────────────────────────────────────────────────────

from pathlib import Path
from typing import List, TypedDict


# ──── DEFINITIONS ──────────────────────────────────────────────────────────────


class AnnotationDict(TypedDict, total=False):
    start_times: List[int]
    durations: List[int]
    end_times: List[int]
    lower_freq: List[int]
    upper_freq: List[int]
    freq_extent: List[int]
    label: List[str]
    file: Path
    sample_rate: int
    bit_rate: int
    source_file: Path


class AudioMetadataDict(TypedDict):
    sample_rate: int
    bit_rate: int
    source_file: Path


class MetadataDict(TypedDict):
    ID: str
    label: str
    sample_rate: int
    length_s: float
    lower_freq: int
    upper_freq: int
    max_amplitude: float
    min_amplitude: float
    bit_rate:  int
    source_file: str
    wav_file: str
