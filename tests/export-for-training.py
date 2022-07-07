#%%
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import cm
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from pykanto.dataset import KantoData
from pykanto.parameters import Parameters
from pykanto.utils.compute import with_pbar
from pykanto.utils.io import load_dataset
from pykanto.utils.paths import ProjDirs, link_project_data

#%%


# ──── # SET PROJECT UP ─────────────────────────────────────────────────────────

PROJECT_ROOT = Path("/home/nilomr/projects/greti-main")
DATA_LOCATION = Path("/media/nilomr/My Passport/SONGDATA/wytham-great-tit")
link_project_data(DATA_LOCATION, PROJECT_ROOT / "data")
RAW_DATASET = "GRETI_2021"
DATASET_ID = "GRETI_2021_TEST"

RAW_DATA = PROJECT_ROOT / "data" / "raw" / RAW_DATASET
DIRS = ProjDirs(PROJECT_ROOT, RAW_DATA, DATASET_ID, mkdir=True)

# ──── TMP CREATE TEST DATASET ──────────────────────────────────────────────────

params = Parameters(
    dereverb=True, verbose=False, subset=(198, 300), song_level=True
)
dataset = KantoData(
    DIRS,
    parameters=params,
    overwrite_dataset=True,
    overwrite_data=True,
)
out_dir = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
dataset = load_dataset(out_dir, DIRS)
dataset.segment_into_units()
dataset.get_units()
dataset.cluster_ids(min_sample=10)
dataset.prepare_interactive_data()


# Load existing dataset
dataset_path = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
dataset = load_dataset(dataset_path, DIRS)
dataset.plot(dataset.data.index[0])


# ──── TMP JOINING TWO DATASETS ─────────────────────────────────────────────────

DATASET_ID_0 = "GRETI_2021_TEST"
DIRS_0 = ProjDirs(PROJECT_ROOT, RAW_DATA, DATASET_ID, mkdir=True)
dataset_path_0 = DIRS_0.DATA / "datasets" / DATASET_ID_0 / f"{DATASET_ID_0}.db"
dataset_0 = load_dataset(dataset_path_0, DIRS_0)

DATASET_ID_1 = "GRETI_2021_TEST_2"
DIRS_1 = ProjDirs(PROJECT_ROOT, RAW_DATA, DATASET_ID_1, mkdir=True)
dataset_path_1 = DIRS_1.DATA / "datasets" / DATASET_ID_1 / f"{DATASET_ID_1}.db"
dataset_1 = load_dataset(dataset_path_1, DIRS_1)

DATASET_ID_J = "GRETI_2021_TEST_JOINED"
DIRS_DERIVED = ProjDirs(PROJECT_ROOT, RAW_DATA, DATASET_ID_J, mkdir=True)

joined_data = pd.concat([dataset_0.data, dataset_1.data])
joined_files = pd.concat([dataset_0.files, dataset_1.files])

joined_dataset = deepcopy(dataset_0)
joined_dataset.data, joined_dataset.files = joined_data, joined_files
# TODO: point to raw and segmented data from both the source datraframes
joined_dataset.DIRS = DIRS_DERIVED

joined_dataset.save_to_disk()
joined_dataset.to_csv(joined_dataset.DIRS.DATASET.parent)


DATASET_ID = "GRETI_2021_TEST_JOINED"
DIRS = ProjDirs(PROJECT_ROOT, RAW_DATA, DATASET_ID, mkdir=True)
dataset_path = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
joined_dataset = load_dataset(dataset_path, DIRS, relink_data=False)


joined_dataset.plot(joined_dataset.data.index[300])

# ──── BUILD DATASET FROM A CSV FILE ────────────────────────────────────────────

# # F defs
# def from_np_array(array_string):
#     array_string = ",".join(array_string.replace("[ ", "[").split())
#     return np.array(ast.literal_eval(array_string))


# def str_to_path(path):
#     return Path(path)

# # def dataset_from_csv()
# recover_csv = pd.read_csv(
#     dataset.DIRS.DATASET.parent / f"{DATASET_ID}_VOCS.csv",
#     index_col=0,
#     dtype={"auto_class": object, "class_label": object},
#     converters={
#         "unit_durations": from_np_array,
#         "onsets": from_np_array,
#         "offsets": from_np_array,
#         "silence_durations": eval,  # TODO why is this a list!
#         spectrogram": str_to_path,  # and all other paths!
#     },
# )

# if len(dataset.data) != len(recover_csv):
#     raise IndexError(
#         "The datasets are of unequal lengths "
#         f"{len(dataset.data)=},{len(recover_csv)=}"
#     )

# [(n, type(c)) for c, n in zip(dataset.data.iloc[0], dataset.data.columns)]

# # Overwrite dataset - careful!
# dataset.data = recover_csv
# dataset.save_to_disk()
# dataset = load_dataset(dataset_path, DIRS)  # Fixes paths on load

# ──── SUBSAMPLE DATASET FOR MODEL TRAINING ─────────────────────────────────────
"""
This create a unique song class label for each vocalisation in the dataset (a
combination of the ID and the label )

"""
# Settings:
min_sample = 10

# Remove rows from song types with fewer than 10 songs
ss_data = (
    dataset.data.query("noise == False")
    .groupby(["ID", "class_label"])
    .filter(lambda x: len(x) >= min_sample)
    .copy()
)

# Sample 10 songs per type and bird
sbs_data = pd.concat(
    [
        data.sample(n=min_sample)
        for _, data in ss_data.groupby(["ID", "class_label"])
    ]
)

# Remove songs labelled as noise (-1)
sbs_data = sbs_data.loc[sbs_data["class_label"] != "-1"]

# Add new unique song type ID
sbs_data["song_class"] = sbs_data["ID"] + "_" + sbs_data["class_label"]

# Print info
n_rem = len(set(dataset.data["ID"])) - len(set(sbs_data["ID"]))
print(f"Removed {n_rem} birds (no songs types with > {min_sample} examples)")

# Add spectorgram files
sbs_data["spectrogram"] = dataset.files["spectrogram"]

# ──── TRAIN / TEST SPLIT AND EXPORT ────────────────────────────────────────────

# Split into train and test subsets and save
train, test = train_test_split(
    sbs_data,
    test_size=0.3,
    shuffle=True,
    stratify=sbs_data["song_class"],
    random_state=42,
)

out_folder = dataset.DIRS.DATASET.parent / "ML"
train_folder, test_folder = out_folder / "train", out_folder / "test"

for dset, dname in zip([train, test], ["train", "test"]):
    # Save spectrograms as images
    to_export = dset.groupby("song_class")["spectrogram"].apply(list).to_dict()

    for i, (song_class, specs) in with_pbar(
        enumerate(to_export.items()), total=len(to_export)
    ):
        # if i > 40:  # REVIEW
        #     break  # !!!!
        folder = (
            train_folder if dname == "train" else test_folder
        ) / song_class
        folder.mkdir(parents=True, exist_ok=True)
        for i, spec in enumerate(specs):
            img = np.load(spec)
            img *= 255.0 / (img + img.min()).max()
            img = np.invert(np.flipud(np.floor(img).astype(int))) + 256
            img = Image.fromarray(np.uint8(cm.magma(img) * 255)).convert("RGB")
            img.save(folder / f"{spec.stem}.jpg")
