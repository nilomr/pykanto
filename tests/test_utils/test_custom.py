# ─── DEPENDENCIES ─────────────────────────────────────────────────────────────

from pathlib import Path
import pkg_resources
import pytest
from pykanto.utils.custom import parse_sonic_visualiser_xml
from pykanto.utils.paths import ProjDirs, get_file_paths
from pykanto.utils.types import SegmentAnnotation


# ──── FIXTURES ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def DIRS():
    DATA_PATH = Path(pkg_resources.resource_filename('pykanto', 'data'))
    PROJECT = Path(DATA_PATH).parent
    RAW_DATA = DATA_PATH / 'raw'
    DIRS = ProjDirs(PROJECT, RAW_DATA, mkdir=True)
    return DIRS


@pytest.fixture()
def xml_file(DIRS):
    xml_file = get_file_paths(DIRS.RAW_DATA, ['.xml'])[0]
    return xml_file

# ──── TESTS ────────────────────────────────────────────────────────────────────


def test_parse_sonic_visualiser_xml(xml_file):
    parsed = parse_sonic_visualiser_xml(xml_file)
    assert isinstance(parsed, SegmentAnnotation)
