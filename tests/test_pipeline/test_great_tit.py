# # ──── DESCRIPTION ──────────────────────────────────────────────────────────────

# # Test main pipeline, minimal example that doesn't follow ANY testing good
# # practices. Necessarily slow - starts ray servers, etc.

# # ──── IMPORTS ─────────────────────────────────────────────────────────────────

# import shutil
# from pathlib import Path

# import pkg_resources
# import pytest
# from pykanto.dataset import KantoData
# from pykanto.parameters import Parameters
# from pykanto.utils.paths import ProjDirs
# from pykanto.utils.read import load_dataset

# # ──── SETTINGS ─────────────────────────────────────────────────────────────────

# DATASET_ID = "GREAT_TIT"

# # ──── TESTS ────────────────────────────────────────────────────────────────────


# @pytest.fixture()
# def DIRS():
#     DATA_PATH = Path(pkg_resources.resource_filename("pykanto", "data"))
#     PROJECT = Path(DATA_PATH).parent
#     RAW_DATA = DATA_PATH / "segmented" / "great_tit"
#     DIRS = ProjDirs(PROJECT, RAW_DATA, DATASET_ID, mkdir=True)
#     return DIRS


# @pytest.fixture()
# def new_dataset(DIRS):
#     params = Parameters(dereverb=True)
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
#     dataset.parameters.update(song_level=True)
#     dataset.cluster_ids(min_sample=5)
#     # TODO: test with min_sample > len of one of the IDs
#     assert "umap_x" in dataset.data
#     assert "auto_class" in dataset.data


# def test_prepare_interactive_data(dataset):
#     dataset.prepare_interactive_data()
#     assert any(isinstance(row, Path) for row in dataset.files.voc_app_data)


# def test_remove_output(dataset):
#     to_rm = [dataset.DIRS.DATASET.parent]
#     for path in to_rm:
#         if path.exists():
#             shutil.rmtree(str(path))
#     assert all([f.exists() for f in to_rm]) == False


# def greti_data_test_manual():

#     DATASET_ID = "GREAT_TIT"
#     DATA_PATH = Path(pkg_resources.resource_filename("pykanto", "data"))
#     PROJECT = Path(DATA_PATH).parent
#     RAW_DATA = DATA_PATH / "segmented" / "great_tit"
#     DIRS = ProjDirs(PROJECT, RAW_DATA, DATASET_ID, mkdir=True)

#     params = Parameters(dereverb=True, verbose=False)
#     dataset = KantoData(
#         DIRS,
#         parameters=params,
#         overwrite_dataset=True,
#         overwrite_data=True,
#     )
#     out_dir = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
#     dataset = load_dataset(out_dir, DIRS)
#     dataset.segment_into_units()

#     for song_level in [True, False]:
#         dataset.parameters.update(song_level=song_level)
#         dataset.get_units()

#     for song_level in [True, False]:
#         dataset.parameters.update(song_level=song_level)
#         dataset.cluster_ids(min_sample=10)

#     for song_level in [True, False]:
#         dataset.parameters.update(song_level=song_level)
#         dataset.prepare_interactive_data()

#     dataset.parameters.update(song_level=False)
#     dataset.open_label_app()
#     dataset = dataset.reload()
