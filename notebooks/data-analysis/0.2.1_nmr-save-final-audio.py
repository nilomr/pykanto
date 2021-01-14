# %%
import collections
from collections.abc import Iterable
import re
import string
import seaborn as sns
import ast
from random import randrange

import librosa
import numpy as np
import pandas as pd
from src.avgn.utils.paths import ensure_dir, most_recent_subdirectory
from src.greti.read.paths import DATA_DIR
from src.greti.sequencing.seqfinder import dict_keys_to_symbol, find_syllable_sequences
from src.vocalseg.utils import butter_bandpass_filter
from tqdm.autonotebook import tqdm
from src.avgn.utils.general import flatten

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

out_dir = DATA_DIR / "processed" / f"{DATASET_ID}_notes"
ensure_dir(out_dir)

# %%
for indv in tqdm(indvs, desc="Saving wav files for each note", leave=True):
    for label in indv_dfs[indv][cluster_labels].unique():
        if label != -1:
            nrows = len(indv_dfs[indv][indv_dfs[indv]
                                       [cluster_labels] == label])
            index = randrange(nrows)  # choose one for now
            data = indv_dfs[indv][indv_dfs[indv]
                                  [cluster_labels] == label].iloc[index]
            wav_loc = (
                most_recent_subdirectory(
                    DATA_DIR / "processed" /
                    DATASET_ID.replace("_segmented", ""),
                    only_dirs=True,
                )
                / "WAV"
                / data.key
            )
            # Load and trim audio
            y, sr = librosa.load(wav_loc, sr=32000)
            y = y[int(data.start_time * sr): int(data.end_time * sr)]

            # Save note audio
            out_filedir = out_dir / f"{data.key[:-4]}-L{label}-I{index}.wav"
            librosa.output.write_wav(out_filedir, y, sr, norm=True)

# %%

# TODO: with sequences inferred, export syllable dataset in order, then read in R etc

# First make a dictionary with each bird's inferred repertoire
syllable_type_dict = {}
for indv in tqdm(indvs, desc="birds processed", leave=True):
    final_dict, n_seqs, _, _ = find_syllable_sequences(
        indv_dfs, indv, cluster_labels, min_freq=1, min_songs=1)
    syllable_type_dict[indv] = [ast.literal_eval(
        key) for key in list(final_dict.keys())]

# %%
# TESTING========================


# %%

indv = 'MP58'

final_dict, n_seqs, seqs, tally = find_syllable_sequences(
    indv_dfs, indv, cluster_labels, min_freq=1, min_songs=1)

testdic = tally
seq_str = seqs
#indv_dfs['B165'][indv_dfs['B165']['syllables_sequence_id'] == 10]
# %%
# remove sliding snakes


def collapse_slidable_seqs(tally):
    """
    Removes sequences that are contained in other sequences if repeated, 
    keeping those that serve as the start of a song most frerquently.
    """
    conflict = []
    for k in tally.keys():
        reference = k*4
        for k2 in tally.keys():
            if k2 != k and k2 in reference:
                print(k2, reference)
                conflict.append(k2)
    how_many_first = {pattrn: sum(
        [seq.startswith(pattrn) for seq in seq_str]) for pattrn in set(conflict)}
    maxkey = max(how_many_first, key=lambda key: how_many_first[key])
    remove = [key for key in conflict if key != maxkey]
    no_sliders = dict((k, tally[k])
                      for k in tally.keys() if k not in remove)
    return no_sliders

# %%


testdict = final_dict


# %%
for _ in range(3):
    testdict, state = remove_repeat_songs(testdict)
    print(testdict.keys())
    if state:
        break

# %%
# Plot distribution of repertoire size

repsize = [len(d) for indv, d in syllable_type_dict.items()]
sns.countplot(repsize)

# %%
# Now save examples of each syllable

for indv, l in tqdm(syllable_type_dict.items(), desc="Saving wav files for each note", leave=True):
    for label in set(flatten(l)):
        nrows = len(indv_dfs[indv][indv_dfs[indv]
                                   [cluster_labels] == label])
        # choose a random note. #TODO select more based on SNR
        index = randrange(nrows)
        data = indv_dfs[indv][indv_dfs[indv]
                              [cluster_labels] == label].iloc[index]
        wav_loc = (
            most_recent_subdirectory(
                DATA_DIR / "processed" /
                DATASET_ID.replace("_segmented", ""),
                only_dirs=True,
            )
            / "WAV"
            / data.key
        )

        # Load and trim audio
        y, sr = librosa.load(wav_loc, sr=32000)
        y = y[int(data.start_time * sr): int(data.end_time * sr)]
        # Save note audio
        out_filedir = out_dir / f"{indv}-{label}.wav"
        librosa.output.write_wav(out_filedir, y, sr, norm=True)

# %%

syllable_type_dict


# Provisional - read 2 csvs, get mean of each cell, then create song vectors according to dict

df1 = pd.read_csv(out_dir / 'acoustic_parameters.csv')
df2 = pd.read_csv(out_dir / 'acoustic_parameters_2.csv')
df = pd.concat([df1, df2])
by_row_index = df.groupby(df.index)
df_means = by_row_index.mean()
df_means.insert(0, 'key', df2.iloc[:, 0])

df_means['key'] = df_means['key'].apply(lambda x: x[:-6])

df_means.insert(1, 'indv', df_means['key'].apply(lambda x: x.split('-')[0]))
df_means.insert(2, 'note', df_means['key'].apply(lambda x: x.split('-')[1]))


full_dict = {}

for indv, l in tqdm(syllable_type_dict.items()):
    indv_df = df_means[df_means['indv'] == indv]
    indv_dict = {}
    for subl in l:
        if len(subl) > 2:  # if len is 3 do nothing, if 2 repeat the first note
            newrow = []
            code = f'{subl[0]}{subl[1]}{subl[2]}'
            for note in subl:
                newrow.append(indv_df[indv_df['note'] ==
                                      str(note)].values.tolist()[0][3:])
            newrow = [indv] + [code] + newrow
            indv_dict[code] = list(flatten(newrow))
        else:
            newrow = []
            code = f'{subl[0]}{subl[1]}{subl[0]}'
            for note in subl:
                newrow.append(indv_df[indv_df['note'] ==
                                      str(note)].values.tolist()[0][3:])
            newrow.append(indv_df[indv_df['note'] == str(
                subl[0])].values.tolist()[0][3:])
            newrow = [indv] + [code] + newrow
            indv_dict[code] = list(flatten(newrow))
    full_dict[indv] = indv_dict

concat_df = pd.concat(
    {k: pd.DataFrame(v).T for k, v in full_dict.items()}, axis=0)
colnames = list(df_means.columns.values)[3:]
newcolnames = ['indv', 'code'] + colnames + [colname +
                                             '_2' for colname in colnames] + [colname + '_3' for colname in colnames]
concat_df.columns = newcolnames

concat_df.to_csv(out_dir / 'acoustic_parameters_mean.csv',
                 index=False, index_label=False)


# %%
# no average


# Provisional - read 2 csvs, get mean of each cell, then create song vectors according to dict

df1 = pd.read_csv(out_dir / 'acoustic_parameters.csv')

df1.insert(0, 'key', df1.iloc[:, 0])
df1 = df1.drop(columns=df1.columns[1])

df1['key'] = df1['key'].apply(lambda x: x[:-6])

df1.insert(1, 'indv', df1['key'].apply(lambda x: x.split('-')[0]))
df1.insert(2, 'note', df1['key'].apply(lambda x: x.split('-')[1]))


full_dict = {}

for indv, l in tqdm(syllable_type_dict.items()):
    indv_df = df1[df1['indv'] == indv]
    indv_dict = {}
    for subl in l:
        if len(subl) > 2:  # if len is 3 do nothing, if 2 repeat the first note
            newrow = []
            code = f'{subl[0]}{subl[1]}{subl[2]}'
            for note in subl:
                newrow.append(indv_df[indv_df['note'] ==
                                      str(note)].values.tolist()[0][3:])
            newrow = [indv] + [code] + newrow
            indv_dict[code] = list(flatten(newrow))
        else:
            newrow = []
            code = f'{subl[0]}{subl[1]}{subl[0]}'
            for note in subl:
                newrow.append(indv_df[indv_df['note'] ==
                                      str(note)].values.tolist()[0][3:])
            newrow.append(indv_df[indv_df['note'] == str(
                subl[0])].values.tolist()[0][3:])
            newrow = [indv] + [code] + newrow
            indv_dict[code] = list(flatten(newrow))
    full_dict[indv] = indv_dict


concat_df = pd.concat(
    {k: pd.DataFrame(v).T for k, v in full_dict.items()}, axis=0)
colnames = list(df_means.columns.values)[3:]
newcolnames = ['indv', 'code'] + colnames + [colname +
                                             '_2' for colname in colnames] + [colname + '_3' for colname in colnames]
concat_df.columns = newcolnames

concat_df.to_csv(out_dir / 'acoustic_parameters_mean.csv',
                 index=False, index_label=False)
