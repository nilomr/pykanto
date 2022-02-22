# ─── DEPENDENCIES ─────────────────────────────────────────────────────────────

from attr import validators
import attr
from pathlib import Path
from typing import List

import git
import pytest
import soundfile as sf
from pykanto.signal.segment import ReadWav
from pykanto.utils.custom import parse_sonic_visualiser_xml
from pykanto.utils.paths import ProjDirs, get_file_paths, get_wavs_w_annotation
from pykanto.utils.types import SegmentAnnotation


# ──── FIXTURES ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def DIRS():
    PROJECT = Path(
        git.Repo('.', search_parent_directories=True).working_tree_dir)
    RAW_DATA = PROJECT / 'pykanto' / 'data' / 'raw'
    DIRS = ProjDirs(PROJECT / 'pykanto', RAW_DATA, mkdir=True)
    return DIRS


@pytest.fixture()
def xml_file(DIRS):
    xml_file = get_file_paths(DIRS.RAW_DATA, ['.xml'])[0]
    return xml_file

# ──── TESTS ────────────────────────────────────────────────────────────────────


def test_parse_sonic_visualiser_xml(xml_file):
    parsed = parse_sonic_visualiser_xml(xml_file)
    assert isinstance(parsed, SegmentAnnotation)
