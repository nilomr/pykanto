# ──── DESCRIPTION ──────────────────────────────────────────────────────────────

# Test main pipeline, minimal example that doesn't follow ANY testing good
# practices. Necessarily slow - starts ray servers, etc.

# ──── IMPORTS ─────────────────────────────────────────────────────────────────

import shutil
from pathlib import Path

import pkg_resources
import pytest
from pykanto.dataset import KantoData
from pykanto.parameters import Parameters
from pykanto.utils.paths import ProjDirs, pykanto_data
from pykanto.utils.read import load_dataset

# ──── SETTINGS ─────────────────────────────────────────────────────────────────

DATASET_ID = "GREAT_TIT"

# ──── FIXTURES ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def DIRS():
    DIRS = pykanto_data(dataset=DATASET_ID)
    return DIRS


@pytest.fixture()
def dataset_dir(DIRS):
    dataset_dir = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
    return dataset_dir


@pytest.fixture()
def new_dataset(DIRS):
    params = Parameters(dereverb=True)
    new_dataset = KantoData(
        DIRS,
        parameters=params,
        overwrite_dataset=True,
        overwrite_data=True,
    )
    return new_dataset


@pytest.fixture()
def dataset(DIRS, dataset_dir):
    return load_dataset(dataset_dir, DIRS)


# ──── TESTS ────────────────────────────────────────────────────────────────────


def test_segment_into_units(new_dataset):
    new_dataset.segment_into_units()
    assert {
        "onsets",
        "offsets",
        "unit_durations",
        "silence_durations",
    }.issubset(new_dataset.data.columns)


@pytest.mark.parametrize(
    "song_level,expected",
    [(True, "average_units"), (False, "units")],
)
def test_get_units(dataset, song_level, expected):
    dataset.parameters.update(song_level=song_level)
    dataset.get_units()
    assert expected in dataset.files.columns
    assert any(
        isinstance(row, Path) for row in getattr(dataset.files, expected)
    )


@pytest.mark.parametrize(
    "song_level,df, ss",
    [(True, "data", 9), (False, "units", 5)],
)
def test_cluster_ids(dataset, song_level, df, ss):
    dataset.parameters.update(song_level=song_level)
    dataset.cluster_ids(min_sample=ss)
    # TODO: test with min_sample > len of one of the IDs
    assert {"umap_x", "auto_class"}.issubset(getattr(dataset, df).columns)


@pytest.mark.parametrize(
    "song_level,colname",
    [(True, "voc_app_data"), (False, "unit_app_data")],
)
def test_prepare_interactive_data(dataset, song_level, colname):
    dataset.parameters.update(song_level=song_level)
    dataset.prepare_interactive_data()
    assert any(isinstance(row, Path) for row in getattr(dataset.files, colname))


def test_move_dataset(DIRS, dataset):
    move_to = DIRS.DATA / f"{DIRS.DATASET_ID}_MOVED"
    shutil.move(str(dataset.DIRS.DATASET.parent), move_to)
    dataset_dir = move_to / f"{DIRS.DATASET_ID}.db"
    new_dataset = load_dataset(dataset_dir, DIRS)
    assert (
        all(
            [
                f.exists()
                for f in new_dataset.files.spectrogram
                if isinstance(f, Path)
            ]
        )
        == True
    )


def test_remove_output(DIRS):
    to_rm = [DIRS.DATA / f"{DIRS.DATASET_ID}_MOVED"]
    for path in to_rm:
        if path.exists():
            shutil.rmtree(str(path))
    assert all([f.exists() for f in to_rm]) == False
