# # ──── DESCRIPTION ──────────────────────────────────────────────────────────────

# # Test main pipeline, minimal example that doesn't follow ANY testing good
# # practices. Necessarily slow - starts ray servers, etc.
# # NOTE: Tests segmentation based in bengalese finch data. Only tests clustering
# # of units.


# import pickle

# # ──── IMPORTS ──────────────────────────────────────────────────────────────────
# import shutil
# from pathlib import Path

# import pkg_resources
# import pytest
# from pykanto.dataset import KantoData
# from pykanto.parameters import Parameters
# from pykanto.signal.segment import segment_files_parallel
# from pykanto.utils.compute import flatten_list
# from pykanto.utils.custom import parse_sonic_visualiser_xml
# from pykanto.utils.paths import ProjDirs, get_file_paths, get_wavs_w_annotation
# from pykanto.utils.read import load_dataset

# # ──── SETTINGS ─────────────────────────────────────────────────────────────────

# DATASET_ID = "BENGALESE_FINCH"

# # ──── TESTS ────────────────────────────────────────────────────────────────────


# @pytest.fixture()
# def DIRS():
#     DATA_PATH = Path(pkg_resources.resource_filename("pykanto", "data"))
#     PROJECT = Path(DATA_PATH).parent
#     RAW_DATA = DATA_PATH / "raw" / DATASET_ID
#     DIRS = ProjDirs(PROJECT, RAW_DATA, DATASET_ID, mkdir=True)
#     return DIRS


# @pytest.fixture()
# def files_to_segment(DIRS):
#     # Get files to segment and segment them
#     wav_filepaths, xml_filepaths = [
#         get_file_paths(DIRS.RAW_DATA, [ext]) for ext in [".wav", ".xml"]
#     ]
#     files_to_segment = get_wavs_w_annotation(wav_filepaths, xml_filepaths)
#     return files_to_segment


# @pytest.fixture()
# def new_dataset(DIRS):
#     params = Parameters(
#         sr=32000,
#         window_length=512,
#         hop_length=128,
#         fft_size=2048,
#         top_dB=120,
#         num_mel_bins=224,
#         lowcut=500,
#         highcut=11000,
#         dereverb=False,
#         silence_threshold=0.1,
#     )
#     new_dataset = KantoData(
#         DIRS,
#         parameters=params,
#         overwrite_dataset=True,
#         overwrite_data=True,
#     )
#     return new_dataset


# @pytest.fixture()
# def dataset(DIRS):
#     # Load an existing dataset
#     out_dir = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
#     dataset = load_dataset(out_dir, DIRS)
#     return dataset


# def test_segment_files_parallel(files_to_segment, DIRS):
#     segment_files_parallel(
#         files_to_segment,
#         DIRS,
#         resample=None,
#         parser_func=parse_sonic_visualiser_xml,
#         min_duration=0.5,
#         min_freqrange=200,
#         labels_to_ignore=["NOISE"],
#     )

#     outfiles = [
#         get_file_paths(DIRS.SEGMENTED, [ext]) for ext in [".wav", ".JSON"]
#     ]
#     outfiles = flatten_list(outfiles)
#     assert all([f.stat().st_size for f in outfiles]) > 0


# def test_segment_into_units(new_dataset):
#     new_dataset.segment_into_units()
#     assert "onsets" in new_dataset.data.columns


# def test_get_units(dataset):
#     for song_level in [True, False]:
#         dataset.parameters.update(song_level=song_level)
#         dataset.get_units()
#         if song_level:
#             assert "average_units" in dataset.files.columns
#             assert any(
#                 isinstance(row, Path) for row in dataset.files.average_units
#             )
#         else:
#             assert "units" in dataset.files.columns
#             assert any(isinstance(row, Path) for row in dataset.files.units)


# def test_cluster_ids(dataset):
#     dataset.parameters.update(song_level=False)
#     dataset.cluster_ids(min_sample=5)
#     assert hasattr(dataset, "units")
#     assert "umap_x" in dataset.units
#     assert "auto_class" in dataset.units


# def test_prepare_interactive_data(dataset):
#     dataset.parameters.update(song_level=False)
#     dataset.prepare_interactive_data()
#     assert any(isinstance(row, Path) for row in dataset.files.unit_app_data)


# def test_remove_output(dataset):
#     to_rm = [dataset.DIRS.DATASET.parent, dataset.DIRS.SEGMENTED]
#     for path in to_rm:
#         if path.exists():
#             shutil.rmtree(str(path))
#     assert all([f.exists() for f in to_rm]) == False


# def bf_data_test_manual():
#     DATA_PATH = Path(pkg_resources.resource_filename("pykanto", "data"))
#     PROJECT = Path(DATA_PATH).parent
#     RAW_DATA = DATA_PATH / "raw" / DATASET_ID
#     DIRS = ProjDirs(PROJECT, RAW_DATA, DATASET_ID, mkdir=True)

#     # Get files to segment and segment them
#     wav_filepaths, xml_filepaths = [
#         get_file_paths(DIRS.RAW_DATA, [ext]) for ext in [".wav", ".xml"]
#     ]
#     files_to_segment = get_wavs_w_annotation(wav_filepaths, xml_filepaths)

#     segment_files_parallel(
#         files_to_segment,
#         DIRS,
#         resample=None,
#         parser_func=parse_sonic_visualiser_xml,
#         min_duration=0.5,
#         min_freqrange=200,
#         labels_to_ignore=["NOISE"],
#     )

#     params = Parameters(
#         sr=32000,
#         window_length=512,
#         hop_length=128,
#         fft_size=2048,
#         top_dB=120,
#         num_mel_bins=224,
#         lowcut=500,
#         highcut=11000,
#         dereverb=False,
#         silence_threshold=0.1,
#     )
#     dataset = KantoData(
#         DIRS,
#         parameters=params,
#         overwrite_dataset=True,
#         overwrite_data=True,
#     )

#     out_dir = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
#     dataset = pickle.load(open(out_dir, "rb"))
#     dataset.segment_into_units()
#     dataset.parameters.update(song_level=False)
#     dataset.get_units()
#     dataset.cluster_ids(min_sample=5)
#     dataset.prepare_interactive_data()
#     dataset.open_label_app()
