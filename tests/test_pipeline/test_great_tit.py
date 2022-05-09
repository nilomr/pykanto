# ──── DESCRIPTION ──────────────────────────────────────────────────────────────

# Test main pipeline, minimal example that doesn't follow ANY testing good
# practices. Necessarily slow - starts ray servers, etc.

# ──── IMPORTS ─────────────────────────────────────────────────────────────────

import pickle
import shutil
from pathlib import Path

import pkg_resources
import pytest
from pykanto.dataset import KantoData
from pykanto.parameters import Parameters
from pykanto.utils.paths import ProjDirs
from pykanto.utils.read import load_dataset

# ──── SETTINGS ─────────────────────────────────────────────────────────────────

DATASET_ID = "GREAT_TIT"

# ──── TESTS ────────────────────────────────────────────────────────────────────


@pytest.fixture()
def DIRS():
    DATA_PATH = Path(pkg_resources.resource_filename("pykanto", "data"))
    PROJECT = Path(DATA_PATH).parent
    RAW_DATA = DATA_PATH / "segmented" / "great_tit"
    DIRS = ProjDirs(PROJECT, RAW_DATA, mkdir=True)
    return DIRS


@pytest.fixture()
def new_dataset(DIRS):
    params = Parameters(dereverb=True)
    new_dataset = KantoData(
        DATASET_ID,
        DIRS,
        parameters=params,
        overwrite_dataset=True,
        overwrite_data=True,
    )
    return new_dataset


@pytest.fixture()
def dataset(DIRS):
    # Load an existing dataset
    out_dir = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
    dataset = pickle.load(open(out_dir, "rb"))
    return dataset


def test_segment_into_units(new_dataset):
    new_dataset.segment_into_units()
    assert "onsets" in new_dataset.vocs.columns


def test_get_units(dataset):
    for song_level in [True, False]:
        dataset.parameters.update(song_level=song_level)
        dataset.get_units()
        if song_level:
            assert isinstance(list(dataset.DIRS.AVG_UNITS.values())[0], Path)
        else:
            assert isinstance(list(dataset.DIRS.UNITS.values())[0], Path)


def test_cluster_ids(dataset):
    dataset.reload()
    for song_level in [True, False]:
        dataset.parameters.update(song_level=song_level)
        dataset.cluster_ids(min_sample=5)

        if song_level:
            assert "umap_x" in dataset.vocs
            assert "auto_type_label" in dataset.vocs
        else:
            assert hasattr(dataset, "units")
            assert "umap_x" in dataset.units
            assert "auto_type_label" in dataset.units


def test_prepare_interactive_data(dataset):
    dataset.reload()
    for song_level in [True, False]:
        dataset.parameters.update(song_level=song_level)
        dataset.prepare_interactive_data()

        if song_level:
            assert isinstance(
                list(
                    dataset.DIRS.VOCALISATION_LABELS["predatasource"].values()
                )[0],
                Path,
            )
        else:
            assert isinstance(
                list(dataset.DIRS.UNIT_LABELS["predatasource"].values())[0],
                Path,
            )


def test_remove_output(dataset):
    to_rm = [dataset.DIRS.DATASET.parent]
    for path in to_rm:
        if path.exists():
            shutil.rmtree(str(path))
    assert all([f.exists() for f in to_rm]) == False


def greti_data_test_manual():

    DATASET_ID = "GREAT_TIT"
    DATA_PATH = Path(pkg_resources.resource_filename("pykanto", "data"))
    PROJECT = Path(DATA_PATH).parent
    RAW_DATA = DATA_PATH / "segmented" / "great_tit"
    DIRS = ProjDirs(PROJECT, RAW_DATA, mkdir=True)

    params = Parameters(dereverb=True, verbose=False)
    dataset = KantoData(
        DATASET_ID,
        DIRS,
        parameters=params,
        overwrite_dataset=True,
        overwrite_data=True,
    )
    out_dir = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
    dataset = load_dataset(out_dir)
    dataset.segment_into_units()

    for song_level in [True, False]:
        dataset.parameters.update(song_level=song_level)
        dataset.get_units()

    dataset.reload()
    for song_level in [True, False]:
        dataset.parameters.update(song_level=song_level)
        dataset.cluster_ids(min_sample=5)

    for song_level in [True, False]:
        dataset.parameters.update(song_level=song_level)
        dataset.prepare_interactive_data()

    dataset.parameters.update(song_level=True)
    dataset.open_label_app()
