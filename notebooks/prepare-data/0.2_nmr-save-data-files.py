# %%

# Save intermediate files necessary to make plots and run analyses

import glob
from datetime import datetime
from os import fspath

import numpy as np
import pandas as pd
from IPython import get_ipython
from src.avgn.dataset import DataSet
from src.avgn.utils.paths import ensure_dir
from src.greti.read.paths import *
from tqdm import tqdm

# %%

# ### get data

DATASET_ID = "GRETI_HQ_2020_segmented"
YEAR = "2020"

# save_loc = DATA_DIR / "syllable_dfs" / DATASET_ID / "{}.pickle".format(DATASET_ID)

embeds_dir = DATA_DIR / "embeddings" / DATASET_ID / "full_dataset.pickle"
syllable_df = pd.read_pickle(embeds_dir)


# %%
# Create a dataset object

dataset = DataSet(DATASET_ID)
len(dataset.data_files)


# %%
# Make dataframe with all metadata
metadata = []
for key in tqdm(dataset.data_files.keys(), leave=False):
    metadata.append(pd.DataFrame(dataset.data_files[key].data))
metadata = pd.concat(metadata)

save_loc = (
    DATA_DIR
    / "processed"
    / DATASET_ID
    / "metadata"
    / "{}_metadata.pickle".format(DATASET_ID)
)
ensure_dir(save_loc)
metadata.to_pickle(save_loc)


# %%
# Count the number of syllables per nest

syllable_n = pd.Series(
    [len(syllable_df[syllable_df.indv == ind]) for ind in syllable_df.indv.unique()]
)

# %%
# Get data for each individual (with cluster info)

DATASET_ID = "GRETI_HQ_2020_segmented"
YEAR = "2020"

all_indv_dfs = pd.concat(
    [
        pd.read_pickle(indv)
        for indv in tqdm(list((DATA_DIR / "indv_dfs" / DATASET_ID).glob("*.pickle")))
    ]
)

# Save this combined dataframe

save_loc = (
    DATA_DIR / "syllable_dfs" / DATASET_ID / "{}_with_labels.pickle".format(DATASET_ID)
)
ensure_dir(save_loc)
all_indv_dfs.to_pickle(save_loc)

# %%

# Prepare and save full dataset (one nestbox per row)

# Get number of songs per nest
date_counts = []
for nestbox in tqdm(metadata["nestbox"].unique()):
    n = metadata.nestbox.str.contains(nestbox).sum()
    date = min(metadata[metadata.nestbox == nestbox]["date"])
    date_counts.append([nestbox, n, date])

date_counts = pd.DataFrame(date_counts, columns=["nestbox", "song_count", "date"])
date_counts["date"] = pd.to_datetime(date_counts["date"])
date_counts = date_counts[date_counts.date != "2020-03-29"]  # remove early test

# import the latest brood data downloaded from https://ebmp.zoo.ox.ac.uk/broods
brood_data_path = RESOURCES_DIR / "brood_data" / "2020"
list_of_files = glob.glob(fspath(brood_data_path) + "/*.csv")
latest_file = max(list_of_files, key=os.path.getctime)
greti_nestboxes = pd.DataFrame(
    (pd.read_csv(latest_file, dayfirst=True).query('Species == "g"'))
)
greti_nestboxes["nestbox"] = greti_nestboxes["Pnum"].str[5:]
greti_nestboxes["Lay date"] = pd.to_datetime(greti_nestboxes["Lay date"], dayfirst=True)

# Merge dfs
date_counts = pd.merge(date_counts, greti_nestboxes, on="nestbox", how="outer")

# Add column = how long after egg laying onset was nest recorded?
date_counts["difference"] = (date_counts["Lay date"] - date_counts["date"]).dt.days

# Count the number of song types per bird

grouped = all_indv_dfs.groupby("indv")

type_counts = grouped.apply(
    lambda x: len(x["hdbscan_labels"].unique()[x["hdbscan_labels"].unique() >= 0])
)
# type_counts=type_counts[type_counts <= 0]


cols = ["nestbox", "n", "syll_type_n"]
data = []

for ind in tqdm(syllable_df.indv.unique()):
    try:
        syll_type_n = int(float(type_counts[ind]))
    except:
        syll_type_n = 0
    n = int(len(syllable_df[syllable_df.indv == ind]))
    zipped = zip(cols, [ind, n, syll_type_n])
    a_dictionary = dict(zipped)
    data.append(a_dictionary)

syllable_info = pd.DataFrame(columns=cols)
syllable_info = syllable_info.append(data, True)
# syllable_info[syllable_info.syll_type_n > 0].shape[0]

# Save to csv file
GRETI_dataset_2020 = pd.merge(date_counts, syllable_info, on="nestbox", how="outer")
out_dir = (
    DATA_DIR / "resources" / DATASET_ID / ("{}_nest_data".format(DATASET_ID) + ".csv")
)
ensure_dir(out_dir)
GRETI_dataset_2020.to_csv(out_dir, index=False)


#%%
# Convert coordinates for mapbox map

# coords_file = RESOURCES_DIR / "nestboxes" / "nestbox_coords.csv"
# coordinates = pd.read_csv(coords_file)


# bng = pyproj.Proj(init="epsg:27700")
# webmercator = pyproj.Proj(init="epsg:3857")
# wgs84 = pyproj.Proj(init="epsg:4326")


# def convertCoords(row):
#     x2, y2 = pyproj.transform(bng, wgs84, row["x"], row["y"])
#     return pd.Series([x2, y2])


# coordinates[["longitude", "latitude"]] = coordinates[["x", "y"]].apply(
#     convertCoords, axis=1
# )

# coords_out = RESOURCES_DIR / "nestboxes" / "nestbox_coords_transformed.csv"
# coordinates.to_csv(coords_out, index=False)
