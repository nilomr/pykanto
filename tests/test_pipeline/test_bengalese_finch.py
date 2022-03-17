

# ──── DESCRIPTION ──────────────────────────────────────────────────────────────

# Test main pipeline, minimal example that doesn't follow ANY testing good
# practices. Necessarily slow - starts ray servers, etc.
# NOTE: Tests segmentation based in bengalese finch data. Only tests clustering
# of units.


# ──── IMPORTS ──────────────────────────────────────────────────────────────────
import shutil
import pkg_resources
from pathlib import Path
import pickle
from pykanto.dataset import SongDataset
from pykanto.parameters import Parameters
from pykanto.signal.segment import segment_files_parallel
from pykanto.utils.compute import flatten_list
from pykanto.utils.custom import parse_sonic_visualiser_xml
from pykanto.utils.paths import ProjDirs, get_file_paths, get_wavs_w_annotation
import pytest

# ──── SETTINGS ─────────────────────────────────────────────────────────────────

DATASET_ID = "BENGALESE_FINCH"

# ──── TESTS ────────────────────────────────────────────────────────────────────


@pytest.fixture()
def DIRS():
    DATA_PATH = Path(pkg_resources.resource_filename('pykanto', 'data'))
    PROJECT = Path(DATA_PATH).parent
    RAW_DATA = DATA_PATH / 'raw' / DATASET_ID
    DIRS = ProjDirs(PROJECT, RAW_DATA, mkdir=True)
    return DIRS


@pytest.fixture()
def files_to_segment(DIRS):
    # Get files to segment and segment them
    wav_filepaths, xml_filepaths = [get_file_paths(
        DIRS.RAW_DATA, [ext]) for ext in ['.wav', '.xml']]
    files_to_segment = get_wavs_w_annotation(wav_filepaths, xml_filepaths)
    return files_to_segment


@pytest.fixture()
def new_dataset(DIRS):
    params = Parameters(sr=32000, top_dB=125, lowcut=200,
                        highcut=11000, dereverb=True)
    new_dataset = SongDataset(DATASET_ID, DIRS, parameters=params,
                              overwrite_dataset=True, overwrite_data=True)
    return new_dataset


@pytest.fixture()
def dataset(DIRS):
    # Load an existing dataset
    out_dir = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
    dataset = pickle.load(open(out_dir, "rb"))
    return dataset


def test_segment_files_parallel(files_to_segment, DIRS):
    segment_files_parallel(
        files_to_segment,
        DIRS,
        resample=None,
        parser_func=parse_sonic_visualiser_xml,
        min_duration=.5,
        min_freqrange=200,
        labels_to_ignore=["NOISE"]
    )

    outfiles = [get_file_paths(DIRS.SEGMENTED, [ext])
                for ext in ['.wav', '.JSON']]
    outfiles = flatten_list(outfiles)
    assert all([f.stat().st_size for f in outfiles]) > 0


def test_segment_into_units(new_dataset):
    new_dataset.segment_into_units()
    assert 'onsets' in new_dataset.vocs.columns


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


def test_cluster_ids(dataset):
    dataset.reload()
    dataset.parameters.update(song_level=False)
    dataset.cluster_ids(min_sample=5)
    assert hasattr(dataset, 'units')
    assert 'umap_x' in dataset.units
    assert 'auto_cluster_label' in dataset.units


def test_prepare_interactive_data(dataset):
    dataset.reload()
    dataset.parameters.update(song_level=False)
    dataset.prepare_interactive_data()
    assert isinstance(
        list(dataset.DIRS.UNIT_LABELS['predatasource'].values())[0],
        Path)


def test_remove_output(dataset):
    to_rm = [dataset.DIRS.DATASET.parent, dataset.DIRS.SEGMENTED]
    for path in to_rm:
        if path.exists():
            shutil.rmtree(str(path))
    assert all([f.exists() for f in to_rm]) == False


def bf_data_test_manual():
    DATASET_ID = "BENGALESE_FINCH"
    DATA_PATH = Path(pkg_resources.resource_filename('pykanto', 'data'))
    PROJECT = Path(DATA_PATH).parent
    RAW_DATA = DATA_PATH / 'raw' / DATASET_ID
    DIRS = ProjDirs(PROJECT, RAW_DATA, mkdir=True)

    params = Parameters(sr=32000, top_dB=130, lowcut=200,
                        highcut=11000, dereverb=True)
    dataset = SongDataset(DATASET_ID, DIRS, parameters=params,
                          overwrite_dataset=True, overwrite_data=True)

    out_dir = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
    dataset = pickle.load(open(out_dir, "rb"))
    dataset.segment_into_units()

    dataset.parameters.update(song_level=False)
    dataset.get_units()

    dataset.reload()
    dataset.parameters.update(song_level=False)
    dataset.cluster_ids(min_sample=5)

    dataset.parameters.update(song_level=False)
    dataset.prepare_interactive_data()

    dataset.open_label_app()
