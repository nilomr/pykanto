# %%
import ast
import collections
import re
import string
from collections.abc import Iterable
from random import randrange
import json
from joblib import Parallel, delayed
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prometheus_client import Counter
import seaborn as sns
from src.avgn.utils.general import flatten
from src.avgn.utils.paths import ensure_dir, most_recent_subdirectory
from src.greti.read.paths import DATA_DIR
from src.greti.sequencing.seqfinder import (collapse_palindromic_keys,
                                            collapse_slidable_seqs,
                                            dict_keys_to_int,
                                            dict_keys_to_symbol,
                                            find_syllable_sequences,
                                            get_mostfreq_pattern,
                                            get_seq_frequencies,
                                            list_sequences,
                                            remove_bad_syllables,
                                            remove_long_ngrams,
                                            remove_repeat_songs)
from src.vocalseg.utils import butter_bandpass_filter
from tqdm.autonotebook import tqdm
import random

from src.greti.write.save_syllables import save_syllable_audio

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

out_dir_notes_wav = DATA_DIR / "processed" / f"{DATASET_ID}_notes" / 'WAV'
out_dir_notes_json = DATA_DIR / "processed" / f"{DATASET_ID}_notes" / 'JSON'
out_dir_syllables_wav = DATA_DIR / "processed" / \
    f"{DATASET_ID}_syllables" / 'WAV'
out_dir_syllables_json = DATA_DIR / "processed" / \
    f"{DATASET_ID}_syllables" / 'JSON'


[ensure_dir(dir) for dir in [out_dir_notes_wav, out_dir_notes_json,
                             out_dir_syllables_wav, out_dir_syllables_json]]

# %%

# TODO: with sequences inferred, export syllable dataset in order, then read in R etc

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
# Save syllables and their metadata #TODO: add seq id etc to this function's output JSONs, also remove caps from 'Bird' key!

for indv, repertoire in tqdm(syllable_type_dict.items(), desc="Saving syllable audio and metadata", leave=True):
    save_syllable_audio(DATASET_ID,
                        indv_dfs,
                        indv,
                        repertoire,
                        cluster_labels,
                        out_dir_syllables_wav,
                        out_dir_syllables_json,
                        shuffle_songs=True,
                        max_seqlength=3,
                        max_n_sylls=10)
# %%

# Parallel:
n_jobs = 3
with Parallel(n_jobs=n_jobs, verbose=2) as parallel:
    all_specs_padded = parallel(
        delayed(save_syllable_audio)(DATASET_ID,
                                     indv_dfs,
                                     indv,
                                     repertoire,
                                     cluster_labels,
                                     out_dir_syllables_wav,
                                     out_dir_syllables_json,
                                     shuffle_songs=True,
                                     max_seqlength=3,
                                     max_n_sylls=10)
        for indv, repertoire in tqdm(
            syllable_type_dict.items(), desc="Saving syllable audio and metadata", leave=True
        )
    )

# %%
# TESTING========================

indv = 'MP69'

# indv_dfs,
# indv,
cluster_labels,
remove_noise = True,
remove_redundant = True,
collapse_palindromes = True,
collapse_subsequences = True,
remove_double_count_notes = True,
remove_double_count_songs = True,
use_n_songs = True,
min_freq = 1
min_songs = 1
double_note_threshold = 0.5  # half of the note duration

sequences_newlabels, labs = list_sequences(
    indv_dfs, indv, cluster_labels)
sym_dict = dict_keys_to_symbol(labs)
sequences = [[sym_dict[b] for b in i] for i in sequences_newlabels]

# Find n-grams
# Convert lists to strings
seq_str = ["".join(map(str, seq)) for seq in sequences]
seq_str_dict = {key: seq for key, seq in enumerate(seq_str)}

# Find repeating motifs; get most frequent in each song
result = get_mostfreq_pattern(seq_str_dict)

#!!
print(result)

# Frequencies for each combination
if use_n_songs:  # Whether to count total occurrences or total number of songs where it appears
    tally = get_seq_frequencies(result, seq_str)
else:
    tally = get_seq_frequencies(result, seq_str, by_song=False)

if remove_noise:
    # Remove sequences with noise (-1 labels)
    tally = {key: n for key, n in tally.items() if 'Z' not in key}

#!!
print(tally)

if remove_redundant:
    # Remove sequences already contained in other, shorter sequences
    # (where the longest of the pair does not have new notes)
    tally = {k: v for k, v in tally.items() if k in remove_long_ngrams([
        key for key in tally.keys()])}

#!!
print(tally, 'HERE')

# Remove absolute infrequent combinations (can help get rid of noise)
tally = {k: v for k, v in tally.items() if v >= min_freq}

#!!
print(tally, 'HERE2')

# if collapse_palindromes:  # Take the one that appears first in sequence most often
#     tally = collapse_palindromic_keys(tally, seq_str)

# #!!
# print(tally)


if collapse_subsequences:
    tally = collapse_slidable_seqs(tally, seq_str)

#!!
print(tally, 'DUP')

# Build dictionary of songs containing each sequence (allows duplicates)
song_dict = {}
for sequence, n in tally.items():
    seqlist = []
    for key, songseq in seq_str_dict.items():
        if sequence in songseq:
            seqlist.append(key)
    song_dict[sequence] = seqlist

# Remove sequences if present in fewer than min_songs
song_dict = {k: v for k, v in song_dict.items() if len(v) >= min_songs}

#!!
print(song_dict)

# Symbols to original labels
final_dict = dict_keys_to_int(song_dict, sym_dict)

#!!
print(final_dict)

# Remove double-counted notes?
if remove_double_count_notes:
    final_dict = remove_bad_syllables(
        indv_dfs, indv, cluster_labels, final_dict, threshold=double_note_threshold)

#!!
print(final_dict)

# Remove double-counted songs? #!EXPERIMENTAL
if remove_double_count_songs:
    for _ in range(10):
        final_dict, state = remove_repeat_songs(final_dict)
        if state:
            break

#!!
print(final_dict)
# %%

#indv_dfs['B165'][indv_dfs['B165']['syllables_sequence_id'] == 10]
# %%


# %%


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
