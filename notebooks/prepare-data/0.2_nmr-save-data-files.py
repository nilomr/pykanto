# %%

# Code to prepare and save intermediate files necessary to make plots and run analyses
# Run only after data-analysis 0.3 is done

import glob
from datetime import datetime
from os import fspath
import numpy as np
import pandas as pd
import pyproj
from src.avgn.dataset import DataSet
from src.avgn.utils.paths import ensure_dir
from src.greti.read.paths import DATA_DIR, RESOURCES_DIR, Path, os
from tqdm import tqdm


def get_first_day(syllable_df, indv):
    return str(np.min(np.unique([datetime.strptime(datet.split('-')[2], '%Y%m%d_%H%M%S')
                                 for datet in syllable_df[syllable_df.indv == indv].key.values])).date())


def get_first_songtimes(metadata, indv):
    times = []
    try:
        for dat in np.unique(metadata[metadata.nestbox == indv].date):
            times.append(
                min(metadata[(metadata.nestbox == indv) & (metadata.date == dat)].time.values))
        return times
    except:
        print('LOL no')


def get_mean_first_songtime(metadata, indv):
    times = get_first_songtimes(metadata, indv)
    org = datetime(1900, 1, 1, 1, 0, 0, 0)

    def unix_time_millis(dt):
        return (dt - org).total_seconds() * 1000.0
    kk = [unix_time_millis(datetime.strptime(t, '%H:%M:%S.%f')) for t in times]
    temp = datetime.fromtimestamp(np.mean(kk) / 1000).strftime('%H:%M:%S')
    return temp

# %% Get data


DATASET_ID = "GRETI_HQ_2020_segmented"
YEAR = "2020"
cluster_labels = "hdbscan_labels_fixed"

# %% Create a dataset object

dataset = DataSet(DATASET_ID)
len(dataset.data_files)

# %%
# Make dataframe with all metadata
metadata = []
for key in tqdm(dataset.data_files.keys(), leave=False):
    metadata.append(pd.DataFrame(dataset.data_files[key].data))
metadata = pd.concat(metadata)
metadata['key'] = metadata['wav_loc'].apply(lambda x: Path(x).stem)

# Pickle metadata:
save_loc = (
    DATA_DIR
    / "processed"
    / DATASET_ID
    / "metadata"
    / "{}_metadata.pickle".format(DATASET_ID)
)
ensure_dir(save_loc)
metadata.to_pickle(save_loc)

# Save time metadata to csv:
save_loc = (
    DATA_DIR
    / "processed"
    / DATASET_ID
    / "metadata"
    / "{}_metadata.csv".format(DATASET_ID)
)
metadata.filter(['nestbox', 'date', 'time', 'key']).to_csv(save_loc)


# %% Prepare full dataset for further analysis

dfs_dir = DATA_DIR / "indv_dfs" / DATASET_ID
indv_dfs = pd.read_pickle(dfs_dir / (f"{DATASET_ID}_labelled_checked.pickle"))
syllable_df = pd.concat(
    [df for df in tqdm(indv_dfs.values())], ignore_index=True)

# Read note information
NOTE_DATASET_ID = DATASET_ID.replace('_segmented', '_notes')
out_dir = (DATA_DIR / "note_dfs" / NOTE_DATASET_ID /
           str(NOTE_DATASET_ID + ".csv"))
note_metadata = pd.read_csv(out_dir)

# Build dictionary for each bird
dict_list = []
for indv in tqdm(syllable_df.indv.unique()):
    dict_list.append(
        {
            'indv': indv,
            'n_notes': len(syllable_df[syllable_df.indv == indv]),
            'n_songs': len(np.unique(syllable_df[syllable_df.indv == indv].key.values)),
            'n_note_types': len(np.unique([lab for lab in syllable_df[syllable_df.indv == indv]
                                           [cluster_labels].values if lab != -1])),
            'n_song_types': len(np.unique(note_metadata[note_metadata.bird == indv].sequence)),
            'first_recording_date': get_first_day(syllable_df, indv),
            'first_song_times': get_first_songtimes(metadata, indv),
            'mean_first_song_time': get_mean_first_songtime(metadata, indv),
            'recording_dates': np.unique(metadata[metadata.nestbox == indv].date).tolist()
        }
    )
keep_df = pd.DataFrame(dict_list)


# %%

# Prepare and save full dataset (one nestbox per row)

# import the latest brood data downloaded from https://ebmp.zoo.ox.ac.uk/broods
brood_data_path = RESOURCES_DIR / "brood_data" / YEAR
list_of_files = glob.glob(fspath(brood_data_path) + "/*.csv")
latest_file = max(list_of_files, key=os.path.getctime)
greti_nestboxes = pd.DataFrame(
    (pd.read_csv(latest_file, dayfirst=True).query('Species == "g"'))
)
greti_nestboxes["nestbox"] = greti_nestboxes["Pnum"].str[5:]
greti_nestboxes["Lay date"] = pd.to_datetime(
    greti_nestboxes["Lay date"], dayfirst=True)

# Merge dfs
keep_df.rename(columns={'indv': 'nestbox'}, inplace=True)
general_df = pd.merge(keep_df, greti_nestboxes, on="nestbox", how="outer")
# Remove legacy columns
general_df = general_df[general_df.columns.drop(
    list(general_df.filter(regex='Legacy')))]

# Add column = how long after egg laying onset was nest recorded?
general_df["first_recording_date"] = pd.to_datetime(
    general_df["first_recording_date"])
general_df["difference"] = (
    general_df["Lay date"] - general_df["first_recording_date"]).dt.days


# %%
# Convert nestbox coordinates and add to df, if file does not already exist

coords_out = RESOURCES_DIR / "nestboxes" / "nestbox_coords_transformed.csv"

if coords_out.is_file():
    coordinates = pd.read_csv(coords_out)
else:
    coords_file = RESOURCES_DIR / "nestboxes" / "nestbox_coords.csv"
    coordinates = pd.read_csv(coords_file)
    bng = pyproj.Proj(init="epsg:27700")
    webmercator = pyproj.Proj(init="epsg:3857")
    wgs84 = pyproj.Proj(init="epsg:4326")

    def convertCoords(row):
        x2, y2 = pyproj.transform(bng, wgs84, row["x"], row["y"])
        return pd.Series([x2, y2])
    coordinates[["longitude", "latitude"]] = coordinates[["x", "y"]].apply(
        convertCoords, axis=1
    )
    coords_out = RESOURCES_DIR / "nestboxes" / "nestbox_coords_transformed.csv"
    coordinates.to_csv(coords_out, index=False)


# NOTE: you will lose ['HE1', 'HE3', 'HE5'] in the process - this is ok
final_df = pd.merge(general_df, coordinates, on='nestbox')

# %%
# Save the final dataframe to csv file
out_dir = (
    DATA_DIR / "resources" / DATASET_ID /
    (f"full_dataset" + ".csv")
)
ensure_dir(out_dir)
final_df.to_csv(out_dir, index=False)

# %%
