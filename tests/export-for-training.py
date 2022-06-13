#%%
import ast
from pathlib import Path
from matplotlib import cm

import numpy as np
import pandas as pd
import pkg_resources
from PIL import Image
from pykanto.dataset import KantoData
from pykanto.parameters import Parameters
from pykanto.utils.compute import with_pbar
from pykanto.utils.paths import ProjDirs, link_project_data, pykanto_data
from pykanto.utils.read import load_dataset
from sklearn.neural_network import MLPClassifier
from tqdm.auto import tqdm

#%%


# ──── # SET PROJECT UP ─────────────────────────────────────────────────────────

PROJECT_ROOT = Path("/home/nilomr/projects/greti-main")
DATA_LOCATION = Path("/media/nilomr/My Passport/SONGDATA/wytham-great-tit")
link_project_data(DATA_LOCATION, PROJECT_ROOT / "data")
DATASET_ID = "GRETI_2021"

RAW_DATA = PROJECT_ROOT / "data" / "raw" / DATASET_ID
DIRS = ProjDirs(PROJECT_ROOT, RAW_DATA, mkdir=True)

# Load existing dataset
dataset_path = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
dataset = load_dataset(dataset_path, DIRS)
dataset.plot("B14_20210501_040000_7")

# ──── # RECOVER DATASET CLASS LABELS FROM A CSV FILE: ──────────────────────────

# F defs
def from_np_array(array_string):
    array_string = ",".join(array_string.replace("[ ", "[").split())
    return np.array(ast.literal_eval(array_string))


def str_to_path(path):
    return Path(path)


# def dataset_from_csv()
recover_csv = pd.read_csv(
    dataset.DIRS.DATASET.parent / f"{DATASET_ID}_VOCS.csv",
    index_col=0,
    dtype={"auto_type_label": object, "type_label": object},
    converters={
        "unit_durations": from_np_array,
        "onsets": from_np_array,
        "offsets": from_np_array,
        "silence_durations": eval,  # TODO why is this a list!
        "spectrogram_loc": str_to_path,  # and all other paths!
    },
)

if len(dataset.vocs) != len(recover_csv):
    raise IndexError(
        "The datasets are of unequal lengths "
        f"{len(dataset.vocs)=},{len(recover_csv)=}"
    )

[(n, type(c)) for c, n in zip(dataset.vocs.iloc[0], dataset.vocs.columns)]

# Overwrite dataset - careful!
dataset.vocs = recover_csv
dataset.save_to_disk()
dataset = load_dataset(dataset_path, DIRS)  # Fixes paths on load

# ──── SUBSAMPLE DATASET FOR MODEL TRAINING ─────────────────────────────────────

# Settings:
min_sample = 10

# Remove rows from song types with fewer than 10 songs
ss_data = (
    dataset.vocs.groupby(["ID", "type_label"])
    .filter(lambda x: len(x) >= min_sample)
    .copy()
)

# Sample 10 songs per type and bird
sbs_data = pd.concat(
    [data.sample(n=10) for _, data in ss_data.groupby(["ID", "type_label"])]
)

# Remove songs labelled as noise (-1)
sbs_data = sbs_data.loc[sbs_data["type_label"] != "-1"]

# Add new unique song type ID
sbs_data["song_class"] = sbs_data["ID"] + "_" + sbs_data["type_label"]

# Print info
n_rem = len(set(dataset.vocs["ID"])) - len(set(sbs_data["ID"]))
print(f"Removed {n_rem} birds with no songs types with > {min_sample} examples")


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
    to_export = (
        dset.groupby("song_class")["spectrogram_loc"].apply(list).to_dict()
    )

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
            img = Image.fromarray(np.uint8(cm.bone(img) * 255)).convert("RGB")
            img.save(folder / f"{spec.stem}.jpg")
