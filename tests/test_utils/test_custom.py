# ─── DEPENDENCIES ─────────────────────────────────────────────────────────────

from pathlib import Path

import pkg_resources
import pytest
from pykanto.utils.custom import (
    chipper_units_to_json,
    parse_sonic_visualiser_xml,
)
from pykanto.utils.paths import ProjDirs, get_file_paths
from pykanto.utils.read import read_json
from pykanto.utils.types import SegmentAnnotation

# ──── FIXTURES ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def DIRS():
    DATASET_ID = "STORM-PETREL"
    DATA_PATH = Path(pkg_resources.resource_filename("pykanto", "data"))
    PROJECT = Path(DATA_PATH).parent
    RAW_DATA = DATA_PATH / "raw" / DATASET_ID
    DIRS = ProjDirs(PROJECT, RAW_DATA, DATASET_ID, mkdir=True)
    return DIRS


@pytest.fixture()
def xml_file(DIRS):
    xml_file = get_file_paths(DIRS.RAW_DATA, [".xml"])[0]
    return xml_file


# ──── TESTS ────────────────────────────────────────────────────────────────────


def test_parse_sonic_visualiser_xml(xml_file):
    parsed = parse_sonic_visualiser_xml(xml_file)
    assert isinstance(parsed, SegmentAnnotation)


def test_chipper_units_to_json(DIRS):
    chipper_units_to_json(DIRS.SEGMENTED)
    testfiles = get_file_paths(DIRS.SEGMENTED, [".JSON"])
    assert (
        all([True for file in testfiles if "onsets" in read_json(file)]) == True
    )
