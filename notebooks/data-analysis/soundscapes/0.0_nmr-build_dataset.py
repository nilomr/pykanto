# %%

import datetime as dt
import json
import os
import re

import pandas as pd
from src.greti.read.paths import DATA_DIR
from src.greti.read.soundscape_data import batch_save_mean_chunks
from tqdm.auto import tqdm

# %%
# import recorded nestboxes
YEAR = "2020"
origin = DATA_DIR / "raw" / YEAR  # Folder to segment
files_path = DATA_DIR / "raw" / YEAR
DATASET_ID = "SOUNDSCAPES_2020"  # Name of output dataset

# %%
# TODO: resample so they are more evenly spaced
# Save json files with data and metadata

batch_save_mean_chunks(DATA_DIR, DATASET_ID, origin, time_range=(
    4, 10), average_over_min=5, normalise=False)

# %%

# Read json files
out_dir_syllables_json = DATA_DIR / "processed" / DATASET_ID / 'JSON'

# Prepare metadata
json_files = [pos_json for pos_json in os.listdir(
    out_dir_syllables_json) if pos_json.endswith('.JSON')]
dict_list = []
for index, js in tqdm(enumerate(json_files)):
    with open(os.path.join(out_dir_syllables_json, js)) as json_file:
        dict_list.append(json.load(json_file))

# Prepare and pickle DF for later use
m5_chunks_df = pd.DataFrame(dict_list)
m5_chunks_df['section'] = m5_chunks_df['nestbox'].apply(
    lambda x: re.findall('\d*\D+', x)[0])
m5_chunks_df['time'] = m5_chunks_df['time'].apply(
    lambda x: dt.datetime.strptime(x, '%H:%M:%S').time())
m5_chunks_df['hour'] = pd.to_datetime(
    m5_chunks_df['time'], format='%H:%M:%S').dt.hour
