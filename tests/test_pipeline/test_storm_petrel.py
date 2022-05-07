# ──── DESCRIPTION ─────────────────────────────────────────────────────────────

# Test main pipeline, minimal example that doesn't follow ANY testing good
# practices. Necessarily slow - starts ray servers, etc.
# NOTE: Tests segmentation based in bengalese finch data. Only tests clustering
# of units.


# ──── IMPORTS ─────────────────────────────────────────────────────────────────

import shutil
import pkg_resources
from pathlib import Path
import pickle
import pytest
from pykanto.dataset import KantoData
from pykanto.parameters import Parameters
from pykanto.signal.segment import segment_files_parallel
from pykanto.utils.compute import flatten_list
from pykanto.utils.custom import (
    parse_sonic_visualiser_xml,
    chipper_units_to_json,
)
from pykanto.utils.paths import ProjDirs, get_file_paths, get_wavs_w_annotation
from pykanto.utils.read import read_json

# ──── SETTINGS ────────────────────────────────────────────────────────────────

DATASET_ID = "STORM-PETREL"

# ──── TESTS ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def DIRS():
    DATA_PATH = Path(pkg_resources.resource_filename("pykanto", "data"))
    PROJECT = Path(DATA_PATH).parent
    RAW_DATA = DATA_PATH / "raw" / DATASET_ID
    DIRS = ProjDirs(PROJECT, RAW_DATA, mkdir=True)
    return DIRS


@pytest.fixture()
def files_to_segment(DIRS):
    # Get files to segment and segment them
    wav_filepaths, xml_filepaths = [
        get_file_paths(DIRS.RAW_DATA, [ext]) for ext in [".wav", ".xml"]
    ]
    files_to_segment = get_wavs_w_annotation(wav_filepaths, xml_filepaths)
    return files_to_segment


@pytest.fixture()
def new_dataset(DIRS):
    params = Parameters(
        window_length=512,
        hop_length=32,
        n_fft=2048,
        num_mel_bins=240,
        sr=22050,
        top_dB=65,
        highcut=10000,
        dereverb=True,
        song_level=True,
    )
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


def test_segment_files_parallel(files_to_segment, DIRS):
    segment_files_parallel(
        files_to_segment,
        DIRS,
        resample=22050,
        parser_func=parse_sonic_visualiser_xml,
        min_duration=0.1,
        min_freqrange=100,
        labels_to_ignore=["NOISE"],
    )

    outfiles = [
        get_file_paths(DIRS.SEGMENTED, [ext]) for ext in [".wav", ".JSON"]
    ]
    outfiles = flatten_list(outfiles)
    assert all([f.stat().st_size for f in outfiles]) > 0


def test_chipper_units_to_json(DIRS):
    chipper_units_to_json(DIRS.SEGMENTED)
    testfiles = get_file_paths(DIRS.SEGMENTED, [".JSON"])
    assert (
        all([True for file in testfiles if "onsets" in read_json(file)]) == True
    )


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
    dataset.parameters.update(song_level=False)
    dataset.cluster_ids(min_sample=5)
    assert hasattr(dataset, "units")
    assert "umap_x" in dataset.units
    assert "auto_type_label" in dataset.units


def test_prepare_interactive_data(dataset):
    dataset.reload()
    dataset.parameters.update(song_level=False)
    dataset.prepare_interactive_data()
    assert isinstance(
        list(dataset.DIRS.UNIT_LABELS["predatasource"].values())[0], Path
    )


def test_remove_output(dataset):
    to_rm = [
        dataset.DIRS.DATASET.parent,
        dataset.DIRS.SEGMENTED / "WAV",
        dataset.DIRS.SEGMENTED / "JSON",
    ]
    for path in to_rm:
        if path.exists():
            shutil.rmtree(str(path))
    assert all([f.exists() for f in to_rm]) == False


def bf_data_test_manual():

    DATASET_ID = "STORM-PETREL"

    DATA_PATH = Path(pkg_resources.resource_filename("pykanto", "data"))
    PROJECT = Path(DATA_PATH).parent
    RAW_DATA = DATA_PATH / "raw" / DATASET_ID
    DIRS = ProjDirs(PROJECT, RAW_DATA, mkdir=True)

    wav_filepaths, xml_filepaths = [
        get_file_paths(DIRS.RAW_DATA, [ext]) for ext in [".wav", ".xml"]
    ]
    files_to_segment = get_wavs_w_annotation(wav_filepaths, xml_filepaths)

    segment_files_parallel(
        files_to_segment,
        DIRS,
        resample=22050,
        parser_func=parse_sonic_visualiser_xml,
        min_duration=0.1,
        min_freqrange=100,
        labels_to_ignore=["NOISE"],
    )

    chipper_units_to_json(DIRS.SEGMENTED)

    params = Parameters(
        window_length=512,
        hop_length=32,
        n_fft=2048,
        num_mel_bins=240,
        sr=22050,
        top_dB=65,
        highcut=10000,
        dereverb=True,
        song_level=True,
    )
    dataset = KantoData(
        DATASET_ID,
        DIRS,
        parameters=params,
        overwrite_dataset=True,
        overwrite_data=True,
    )

    out_dir = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
    dataset = pickle.load(open(out_dir, "rb"))
    dataset.segment_into_units()

    dataset.get_units()
    dataset.reload()
    dataset.cluster_ids(min_sample=5)
    dataset.prepare_interactive_data()
    dataset.open_label_app()

    for voc in dataset.vocs.index[:2]:
        dataset.plot_segments(voc)
