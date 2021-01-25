# %%
import math
import os
import hdbscan
import ast
from scipy.spatial import distance
import collections
import re

from sklearn.tree import plot_tree
from src.avgn.visualization.projections import scatter_projections
from src.avgn.signalprocessing.create_spectrogram_dataset import flatten_spectrograms
import string
from collections.abc import Iterable
from random import randrange
import json
from joblib import Parallel, delayed
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.avgn.utils.general import flatten
from src.avgn.utils.paths import ensure_dir, most_recent_subdirectory
from src.greti.read.paths import DATA_DIR, RESOURCES_DIR
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
import umap
from src.greti.write.save_syllables import save_syllable_audio
from sklearn.decomposition import IncrementalPCA
import phate

%load_ext autoreload
%autoreload 2

# %%

DATASET_ID = "GRETI_HQ_2020_segmented"
YEAR = "2020"
cluster_labels = "hdbscan_labels_fixed"
dfs_dir = DATA_DIR / "indv_dfs" / DATASET_ID
indv_dfs = pd.read_pickle(dfs_dir / (f"{DATASET_ID}_labelled_checked.pickle"))


# all_birds = list(indv_dfs.keys())

# exclude = [
#     "CP34",
#     "EX38A",
#     "SW119",
#     "B11",
#     "MP20",
#     "B161",
#     "EX34",
#     "CP28",
#     "B49",
#     "SW116",
# ]  # Data for these are too noisy, judged from UMAP scatterplot in previous step
# indvs = [indv for indv in all_birds if indv not in exclude]

# butter_lowcut = 1200
# butter_highcut = 10000
# out_dir_notes_wav = DATA_DIR / "processed" / f"{DATASET_ID}_notes" / 'WAV'
# out_dir_notes_json = DATA_DIR / "processed" / f"{DATASET_ID}_notes" / 'JSON'
# out_dir_syllables_wav = DATA_DIR / "processed" / \
#     f"{DATASET_ID}_syllables" / 'WAV'

out_dir_syllables_json = DATA_DIR / "processed" / \
    f"{DATASET_ID}_syllables" / 'JSON'


[ensure_dir(dir) for dir in [out_dir_notes_wav, out_dir_notes_json,
                             out_dir_syllables_wav, out_dir_syllables_json]]

# %%


# Prepare metadata
json_files = [pos_json for pos_json in os.listdir(
    out_dir_syllables_json) if pos_json.endswith('.json')]
dict_list = []
for index, js in tqdm(enumerate(json_files)):
    with open(os.path.join(out_dir_syllables_json, js)) as json_file:
        dict_list.append(json.load(json_file))
syllables_df = pd.DataFrame(dict_list)


# %%


# %%
# Get spectrograms (no time information)
for bdict in tqdm(dict_list):
    indv = bdict['bird']
    key = bdict['song_wav_loc'].split(os.sep)[-1]
    df = indv_dfs[indv][indv_dfs[indv]['key'] == key]
    spec1 = df[df['syllables_sequence_pos'] ==
               bdict['position'][0]].spectrogram.values[0]
    spec2 = df[df['syllables_sequence_pos'] ==
               bdict['position'][0]+1].spectrogram.values[0]
    spec3 = df[df['syllables_sequence_pos'] ==
               bdict['position'][1]].spectrogram.values[0]
    spectrogram = np.concatenate((spec1, spec2, spec3), axis=1)
    bdict['spectrogram'] = spectrogram

syllable_df = pd.DataFrame(dict_list)

# %%

specs = list(syllable_df.spectrogram.values)
specs = flatten_spectrograms(specs)

# %%

umap_parameters = {
    "n_neighbors": 30,
    "min_dist": 0.1,
    "n_components": 5,
    "verbose": True,
    "init": "spectral",
    "low_memory": True,
}
fit = umap.UMAP(**umap_parameters)
syllable_df["umap"] = list(fit.fit_transform(specs))

# # %%
# # PCA
# pca_parameters = {
#     "n_components": 2,
#     "batch_size": 10,
# }
# ipca = IncrementalPCA(**pca_parameters)
# syllable_df["pca"] = list(ipca.fit_transform(specs))

# #%%
# # PHATE
# phate_parameters = {"n_jobs": -1, "knn": 10, "n_pca": 100, "gamma": 1}
# phate_operator = phate.PHATE() #**phate_parameters
# syllable_df["phate"] = list(phate_operator.fit_transform(specs))

# %%

z = list(syllable_df["umap"].values)
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=10,
    min_samples=1,  # larger values = more conservative clustering
    cluster_selection_method="eom",
)
clusterer.fit(z)
syllable_df["hdbscan_labels"] = clusterer.labels_


labelitas = clusterer.labels_.tolist()
sns.distplot(labelitas, kde=False)
print(f'Noise: {labelitas.count(-1)}',
      f'Total labels: {len(np.unique(labelitas))}')
# %%

labs = list(syllable_df.hdbscan_labels)
projection = syllable_df["umap"].tolist()
plt.figure(figsize=(12, 12))
sns.scatterplot([item[0] for item in projection], [item[1]
                                                   for item in projection], hue=labs)

# %%
labs = list(syllable_df.Bird)
projection = syllable_df["pca"].tolist()
plt.figure(figsize=(12, 12))
sns.scatterplot([item[0] for item in projection], [item[1]
                                                   for item in projection], hue=labs)

# %%

labs = list(syllable_df.Bird)
projection = syllable_df["phate"].tolist()
plt.figure(figsize=(12, 12))
sns.scatterplot([item[0] for item in projection], [item[1]
                                                   for item in projection], hue=labs)

# %%

# Add nestbox positions to syllable_df
coords_file = RESOURCES_DIR / "nestboxes" / "nestbox_coords.csv"
tmpl = pd.read_csv(coords_file)
nestboxes = tmpl[tmpl["nestbox"].isin(syllable_df.Bird.unique())]
nestboxes["east_north"] = nestboxes[["x", "y"]].apply(tuple, axis=1)

# plt.figure(figsize=(8, 6), dpi=200)
# plt.scatter(nestboxes['x'], nestboxes['y'], s=6, c="k")
X = [(448500, 207000)]
for i in nestboxes.index:
    nestboxes.at[i, "dist_m"] = distance.cdist(
        X, [nestboxes.at[i, "east_north"]], "euclidean"
    )[0, 0]
nestboxes.filter(["nestbox", "east_north", "section", "dist_m"])

#  Add to syllable_df
syllable_df = pd.merge(
    syllable_df, nestboxes, how="inner", left_on="Bird", right_on="nestbox"
)

# %%
# Add syllable identifier to df
syllable_df['syll_id'] = syllable_df['Bird'] + '-' + \
    syllable_df['sequence'].apply(lambda x: "".join(str(i) for i in x))


df_array = [*syllable_df["east_north"]]
spatial_dist = distance.cdist(df_array, df_array)

data_list = []
clusters = syllable_df[["hdbscan_labels", "Bird", "syll_id"]].values.tolist()
for i, bird in tqdm(enumerate(clusters)):
    for j, bird2 in enumerate(clusters):
        if bird[1] == bird2[1]:
            continue
        else:
            # Mark syllables not assigned to any cluster as 'not shared'
            if bird[0] == bird2[0] and bird[0] != -1 and bird2[0] != -1:
                data_list.append(
                    [bird[1], bird2[1], 'YES', spatial_dist[i, j], bird[2], bird2[2]])
            else:
                data_list.append(
                    [bird[1], bird2[1], 'NO', spatial_dist[i, j], bird[2], bird2[2]])


data_frame = pd.DataFrame.from_records(data_list,  columns=[
                                       'bird1', 'bird2', 'share', 'dist', 'bird1_song', 'bird2_song'])

# %%

# Plot sharing vs distance
sns.set_style("darkgrid")

data_1km = data_frame[data_frame['dist'] < 2000]
x = data_1km[data_1km['share'] == 'NO'].dist.values
y = data_1km[data_1km['share'] == 'YES'].dist.values

fig, ax = plt.subplots(figsize=(6, 6))
for a in [x, y]:
    sns.distplot(a, ax=ax, kde=False, norm_hist=True, bins=15)
ax.grid(False)

# %%


# %%
# TODO: now do it by bird
grouped = data_frame.groupby(['bird1', 'bird2', 'dist', 'bird1_song', 'bird2_song']).agg(
    {'share': 'max'}).reset_index()

songtypes = grouped.groupby(['bird1'])['bird1_song'].unique().reset_index()
songtypes['n'] = songtypes['bird1_song'].apply(len)

sharing_df = pd.get_dummies(grouped, columns=['share']).groupby(
    ['bird1', 'bird2', 'dist']).sum().reset_index()

sharing_df = sharing_df.merge(songtypes[['bird1', 'n']], on='bird1').merge(
    songtypes[['bird1', 'n']], left_on=['bird2'], right_on=['bird1']).rename(columns={'bird1_x': 'bird1'})

sharing_df['rep_index'] = (2*sharing_df['share_YES']) / \
    (sharing_df['n_x'] + sharing_df['n_y'])
sharing_df = sharing_df.drop_duplicates()

# %%
# Plot
fig, ax = plt.subplots(figsize=(6, 6))
sns.lmplot(x='dist', y='rep_index', scatter_kws={
           's': 30, 'alpha': .2}, y_jitter=0.02, fit_reg=False, data=sharing_df)
ax.grid(False)


sns.lmplot(x='dist', y='rep_index', scatter_kws={
           's': 30, 'alpha': .2}, y_jitter=0.02, fit_reg=True, data=sharing_df[sharing_df['rep_index'] > 0])
ax.grid(False)


sns.regplot(x='dist', y='rep_index', scatter_kws={
            's': 30, 'alpha': .2}, data=sharing_df)


[sharing_df['dist'] < 1000]
[sharing_df['rep_index'] > 0]

# %%

data_1km.share.replace(['NO', 'YES'], [0, 1], inplace=True)

sns.regplot(x='dist', y='share', data=data_1km, n_boot=500, logistic=True)

# %%


def save_syllable_audio(DATASET_ID,
                        indv_dfs,
                        indv,  # dict keys
                        # dict values (lists of lists, where sublists are song types)
                        repertoire,
                        cluster_labels,
                        out_dir_syllables_wav,
                        out_dir_syllables_json,
                        shuffle_songs=True,  # Useful to avoud using very similar examples
                        max_seqlength=3,  # There is no warrantee that this will work with greater lenghts as is
                        max_n_sylls=10
                        ):

    # Get list of unique files (one per song)
    songs = np.unique(indv_dfs[indv].key.values.tolist())
    if shuffle_songs:
        random.shuffle(songs)
    songtype_counter = {}
    for song in songs:
        # Get song sequence
        sequence = indv_dfs[indv][indv_dfs[indv]
                                  ['key'] == song][cluster_labels].tolist()
        for songtype in repertoire:
            # Trim or extend syllable
            songtype = trim_or_extend_songtype(max_seqlength, songtype)
            typestring = ''.join(str(e) for e in songtype)

            # Check which song type is present in the sequence
            if is_a_in_x(songtype, sequence):
                # Extract a sequence (of max_seqlength length)
                indexes = find_sublist(songtype, sequence)
                for index in indexes:
                    # Get syllable times
                    starts = indv_dfs[indv][indv_dfs[indv]
                                            ['key'] == song].start_time.values.tolist()
                    ends = indv_dfs[indv][indv_dfs[indv]
                                          ['key'] == song].end_time.values.tolist()

                    # Get IOIs, silences, etc, build dictionary
                    substarts = [starts[i]
                                 for i in [index[0], index[0]+1, index[1]]]
                    subends = [ends[i]
                               for i in [index[0], index[0]+1, index[1]]]
                    subdurs = [y - x for x, y in zip(substarts, subends)]
                    subIOIs = [y - x for x, y in zip(substarts, substarts[1:])]
                    subsilences = [y - x for x,
                                   y in zip(subends, substarts[1:])]

                    # Add to songtype counter dictionary
                    if typestring in songtype_counter:
                        songtype_counter[typestring] += 1
                        # Stop if max number reached
                        if songtype_counter[typestring] > max_n_sylls:
                            break
                    else:
                        songtype_counter[typestring] = 0

                    wav_loc = get_song_dir(DATASET_ID, song)
                    out_filedir = out_dir_syllables_wav / \
                        f"{indv}-{typestring}-{songtype_counter[typestring]}.wav"

                    # Fetch, trim, and save audio
                    y, sr = librosa.load(wav_loc, sr=32000)
                    y = y[int(substarts[0] * sr): int(subends[-1] * sr)]
                    librosa.output.write_wav(out_filedir, y, sr, norm=True)

                    # Save dictionary
                    syllable_dict = {'Bird': indv,
                                     'start_time': substarts[0], 'end_time': subends[-1],
                                     'starts': substarts, 'ends': subends,
                                     'total_duration': subends[-1] - substarts[0],
                                     'durations': subdurs, 'IOIs': subIOIs, 'silences': subsilences,
                                     'rate': sr,
                                     'position': index, 'sequence': songtype,
                                     'song_wav_loc': str(wav_loc),
                                     'syll_wav_loc': str(out_filedir)}

                    out_filejson = out_dir_syllables_json / \
                        f"{indv}-{typestring}-{songtype_counter[typestring]}.json"

                    json.dump(syllable_dict, open(out_filejson,
                                                  'w', encoding="utf8"), sort_keys=True)


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
# Save syllables and their metadata

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
