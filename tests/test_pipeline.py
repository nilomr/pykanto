

# ──── DESCRIPTION ──────────────────────────────────────────────────────────────

# Test main pipeline, minimal example that doesn't follow ANY testing good
# practices. Necessarily slow - starts ray servers, etc.

# ──── IMPORTS ──────────────────────────────────────────────────────────────────
from pathlib import Path
import pickle
from attr import has
import git
from pykanto.dataset import SongDataset
from pykanto.parameters import Parameters
from pykanto.utils.paths import ProjDirs
import pytest

# ──── SETTINGS ─────────────────────────────────────────────────────────────────

song_level = True
DATASET_ID = "GREAT_TIT_EXAMPLE"

# ──── TESTS ────────────────────────────────────────────────────────────────────


# dataset = test_segment_into_units(dataset)
# dataset = test_get_units(dataset)
# dataset = test_cluster_individuals(dataset)
# dataset = test_prepare_interactive_data(dataset)

@pytest.fixture()
def DIRS():
    PROJECT_DIR = Path(
        git.Repo('.', search_parent_directories=True).working_tree_dir)
    DIRS = ProjDirs(PROJECT_DIR / 'pykanto', mkdir=True)
    DIRS.append('WAVFILES', DIRS.PROJECT / 'data' / 'great_tit')
    return DIRS


@pytest.fixture()
def new_dataset(DIRS):
    params = Parameters(dereverb=True)
    dataset = SongDataset(DATASET_ID, DIRS, parameters=params,
                          overwrite_dataset=True, overwrite_data=True)
    return dataset


@pytest.fixture()
def dataset(DIRS):
    # Load an existing dataset
    out_dir = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
    dataset = pickle.load(open(out_dir, "rb"))
    return dataset


def test_segment_into_units(new_dataset):
    new_dataset.segment_into_units()
    assert 'onsets' in new_dataset.vocalisations.columns


def test_get_units(dataset):
    for song_level in [True, False]:
        dataset.parameters.update(song_level=song_level)
        dataset.get_units()
        if song_level:
            assert isinstance(
                list(dataset.DIRS.AVG_UNITS.values())[0],
                Path)
        else:
            assert isinstance(list(dataset.DIRS.UNITS.values())[0], Path)


def test_cluster_individuals(dataset):
    dataset.reload()
    for song_level in [True, False]:
        dataset.parameters.update(song_level=song_level)
        dataset.cluster_individuals(min_sample=5)

        if song_level:
            assert 'umap_x' in dataset.vocalisations
            assert 'auto_cluster_label' in dataset.vocalisations
        else:
            assert hasattr(dataset, 'units')
            assert 'umap_x' in dataset.units
            assert 'auto_cluster_label' in dataset.units


def test_prepare_interactive_data(dataset):
    dataset.reload()
    for song_level in [True, False]:
        dataset.parameters.update(song_level=song_level)
        dataset.prepare_interactive_data()

        if song_level:
            assert isinstance(
                list(dataset.DIRS.VOCALISATION_LABELS['predatasource'].values())
                [0],
                Path)
        else:
            assert isinstance(
                list(dataset.DIRS.UNIT_LABELS['predatasource'].values())[0],
                Path)
