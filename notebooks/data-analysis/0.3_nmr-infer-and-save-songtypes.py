# %%
import ast
import collections
import json
import os
import random
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib2 import Path
from src.avgn.utils.paths import ensure_dir
from src.greti.read.paths import DATA_DIR, safe_makedir
from src.greti.sequencing.seqfinder import find_syllable_sequences
from src.greti.write.save_syllables import (find_sublist, get_song_dir,
                                            is_a_in_x, join_and_save_notes,
                                            save_note_audio,
                                            save_syllable_audio,
                                            trim_or_extend_songtype)
from tqdm.autonotebook import tqdm

%load_ext autoreload
%autoreload 2

# %%

DATASET_ID = "GRETI_HQ_2020_segmented"
YEAR = "2020"
cluster_labels = "hdbscan_labels_fixed"

dfs_dir = DATA_DIR / "indv_dfs" / DATASET_ID
indv_dfs = pd.read_pickle(dfs_dir / (f"{DATASET_ID}_labelled_checked.pickle"))


all_birds = list(indv_dfs.keys())

exclude = [
    "CP34",
    "EX38A",
    "SW119",
    "B11",
    "MP20",
    "B161",
    "EX34",
    "CP28",
    "B49",
    "SW116",
]  # Data for these are too noisy, judged from UMAP scatterplot in previous step

indvs = [indv for indv in all_birds if indv not in exclude]

butter_lowcut = 1200
butter_highcut = 10000

out_dir = DATA_DIR / "processed" / DATASET_ID.replace('_segmented', "")
safe_makedir(out_dir)


# %%

# First make a dictionary with each bird's inferred repertoire
# NOTE: There is some stochasticity here - not much, but might want to make deterministic.
# Slight variations probably come from cases where two alternative song types are found the same number of times.
syllable_type_dict = {}
for indv in tqdm(indvs, desc="Inferring song type repertoires", leave=True):
    final_dict, n_seqs, _, _ = find_syllable_sequences(
        indv_dfs, indv, cluster_labels, min_freq=1, min_songs=1, double_note_threshold=0.5)
    syllable_type_dict[indv] = [ast.literal_eval(
        key) for key in list(final_dict.keys())]

# %%
# Plot distribution of repertoire size
sns.set_style("dark")
repsize = [len(d) for indv, d in syllable_type_dict.items()]
sns.countplot(repsize, color='grey')
# %%

# Check maximum sequence length:
seqlen = [len(seq) for indv, d in syllable_type_dict.items() for seq in d]
ax, fig = plt.subplots(figsize=(3, 5))
sns.countplot(seqlen, color='grey')
for n, freq in dict(sorted(collections.Counter(seqlen).items(),
                           key=lambda item: item[1],
                           reverse=True)).items():
    print(f'{n} notes = {freq} cases')

# %%
# Save syllables and their metadata
for indv, repertoire in tqdm(syllable_type_dict.items(),
                             desc="Saving syllable audio and metadata",
                             leave=True):
    save_syllable_audio(DATASET_ID,
                        indv_dfs,
                        indv,
                        repertoire,
                        cluster_labels,
                        out_dir,
                        shuffle_songs=True,
                        max_seqlength=3,
                        max_n_sylls=10)
# %%
# Save individual notes
for indv, repertoire in tqdm(syllable_type_dict.items(),
                             desc="Saving note audio and metadata",
                             leave=True):
    save_note_audio(DATASET_ID,
                    indv_dfs,
                    indv,  # dict keys
                    # dict values (lists of lists, where sublists are song types)
                    repertoire,
                    cluster_labels,
                    out_dir,
                    shuffle_songs=True,  # Useful to avoud using very similar examples
                    max_seqlength=3,  # There is no warrantee that this will work if > 3, haven't tested
                    max_n_sylls=10  # Max per song / bird
                    )


# %%
# Save joined notes.
# NOTE: the two functions above take a random subset of data,
# whereas this just takes the output of 'save_note_audio()' and joins notes into syllables.
# This means that you need to run 'save_note_audio()' AND this if you want to get
# a different random subset per song type and bird.

join_and_save_notes(out_dir, test_run=False, n_jobs=-1)


# %% Save metadata as csv to read with R (for individual notes)

DATASET_ID = "GRETI_HQ_2020_notes"
out_dir_notes_json = DATA_DIR / "processed" / DATASET_ID / 'JSON'

# Prepare metadata
json_files = [pos_json for pos_json in os.listdir(
    out_dir_notes_json) if pos_json.endswith('.json')]
dict_list = []
for index, js in tqdm(enumerate(json_files)):
    with open(os.path.join(out_dir_notes_json, js)) as json_file:
        dict_list.append(json.load(json_file))
syllables_df = pd.DataFrame(dict_list)
syllables_df['key'] = syllables_df['syll_wav_loc'].apply(
    lambda x: Path(x).stem)
syllables_df[['silence_1', 'silence_2']] = (syllables_df['silences']
                                            .transform([lambda x:x[0], lambda x:x[1]])
                                            .set_axis(['silence_1', 'silence_2'],
                                                      axis=1,
                                                      inplace=False)
                                            )

# Save
out_dir = (DATA_DIR / "note_dfs" / DATASET_ID / str(DATASET_ID + ".csv"))
ensure_dir(out_dir)
syllables_df.to_csv(out_dir, index=False)

# %%
