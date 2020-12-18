# %%
import ast
import pickle
import random
import re
import string
from datetime import datetime

import hdbscan
import librosa
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import phate
import seaborn as sns
import umap
from joblib import Parallel, delayed
from matplotlib.collections import PatchCollection
from networkx.classes import ordered
from scipy.spatial import distance
from src.avgn.dataset import DataSet
from src.avgn.signalprocessing.create_spectrogram_dataset import (
    flatten_spectrograms,
    log_resize_spec,
)
from src.avgn.utils.general import save_fig
from src.avgn.utils.hparams import HParams
from src.avgn.utils.paths import ensure_dir, most_recent_subdirectory
from src.avgn.visualization.barcodes import indv_barcode, plot_sorted_barcodes
from src.avgn.visualization.network_graph import (
    build_transition_matrix,
    compute_graph,
    draw_networkx_edges,
    plot_network_graph,
)
from src.avgn.visualization.projections import scatter_projections, scatter_spec
from src.avgn.visualization.quickplots import draw_projection_plots, quad_plot_syllables
from src.avgn.visualization.spectrogram import draw_spec_set
from src.greti.read.paths import DATA_DIR, FIGURE_DIR, RESOURCES_DIR
from src.vocalseg.utils import (
    butter_bandpass_filter,
    int16tofloat32,
    plot_spec,
    spectrogram,
)
from tqdm.autonotebook import tqdm

# from sklearn.cluster import MiniBatchKMeans
# from cuml.manifold.umap import UMAP as cumlUMAP

# import importlib
# importlib.reload(src)

get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

# %%


def get_true_labels(transition_matrix, drop_list):
    # Get the labels used to classify notes in the dataset
    true_label = []
    used_drop_list = []
    for i in range(len(transition_matrix)):
        for drop in drop_list:
            if i + len(used_drop_list) >= drop:
                if drop not in used_drop_list:
                    used_drop_list.append(drop)
        true_label.append(i + len(used_drop_list))
    return true_label


def duplicate_1grams(result):
    # Duplicate 1-grams
    result = [seq * 2 if len(seq) == 1 else seq for seq in set(result)]
    return result


def remove_palindromes(result):
    # Keep one of each pair of palindromes
    minus_palindromes = []
    for seq in result:
        if (
            seq[::-1] in result
            and seq not in minus_palindromes
            and seq[::-1] not in minus_palindromes
            or seq[::-1] not in result
        ):
            minus_palindromes.append(seq)
    return minus_palindromes


def remove_long_ngrams(minus_palindromes):
    # Remove collapsible sequences
    remove = [
        seq2
        for seq2 in minus_palindromes
        for seq in minus_palindromes
        if seq != seq2
        and seq in seq2
        and not [
            substr2 for substr2 in seq2 if substr2 not in seq
        ]  # contains no new notes
    ]
    return [seq for seq in minus_palindromes if seq not in remove]


def dict_keys_to_symbol(labs):
    # Change keys from int to symbol: this is necessary to be able to search for patterns where int labels >9
    unique_labs = np.unique(labs)
    nlabs = len(unique_labs)
    symbolic_dict = {
        lab: symbol
        for lab, symbol in zip(unique_labs, list(string.ascii_uppercase[0:nlabs]))
    }
    return symbolic_dict


# %%

DATASET_ID = "GRETI_HQ_2020_segmented"
YEAR = "2020"


dfs_dir = DATA_DIR / "indv_dfs" / DATASET_ID
# indv_dfs = pd.read_pickle(dfs_dir / (f"{DATASET_ID}_labelled_checked.pickle")) #! Change to this once manual check done

indv_dfs = pd.read_pickle(
    "/media/nilomr/SONGDATA/syllable_dfs/GRETI_HQ_2020_segmented/GRETI_HQ_2020_segmented_with_labels.pickle"
)
indv_dfs = {indv: indv_dfs[indv_dfs.indv == indv] for indv in indv_dfs.indv.unique()}

indvs = list(indv_dfs.keys())


# syll_dir = (
#     DATA_DIR
#     / "syllable_dfs"
#     / DATASET_ID
#     / "{}_with_labels.pickle".format(DATASET_ID)  #! Change to proper ('_fixed') labels
# )
# syllable_df = pd.read_pickle(syll_dir)


#%%

# Prepare colour palettes

label = "hdbscan_labels"
facecolour = "#f2f1f0"
colourcodes = [
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
    "#e78ac3",
    "#a6d854",
    "#ffd92f",
    "#e5c494",
    "#b3b3b3",
    "#fc6c62",
    "#7c7cc4",
    "#57b6bd",
    "#e0b255",
]

pal = sns.set_palette(sns.color_palette(colourcodes))
unique_indvs = list(indv_dfs.keys())[:2]


# %%
# Select one bird to test

bird = "SW5"


indv_dict = {}

color_lists, trans_lists, label_pal_dict, label_pal, label_dict = indv_barcode(
    indv_dfs[bird], time_resolution=0.02, label=label, pal=pal,
)

indv_dict[bird] = {"label_pal_dict": label_pal_dict, "label_dict": label_dict}


#%%

# Build transition matrix
# TODO: Optimise this

labs = indv_dfs[bird][label].values
sequence_ids = np.array(indv_dfs[bird]["syllables_sequence_id"])
sequences = [labs[sequence_ids == i] for i in np.unique(sequence_ids)]

transition_matrix, element_dict, drop_list = build_transition_matrix(
    sequences, min_cluster_samples=10
)

element_dict_r = {v: k for k, v in element_dict.items()}

true_label = get_true_labels(transition_matrix, drop_list)

# Transition matrix in DataFrame format, with true labels
matrix = pd.DataFrame(transition_matrix, columns=true_label, index=true_label)
sns.heatmap(matrix)

# Substitute labels
sequences_newlabels = []
for song in sequences:
    sequences_newlabels.append([element_dict.get(n, n) for n in song])


# %%

# * Define song types

# Convert int labels to symbolic labels
sym_dict = dict_keys_to_symbol(labs)
sequences = [[sym_dict[b] for b in i] for i in sequences_newlabels]

# Find n-grams

# Convert lists to strings
seq_str = ["".join(map(str, seq)) for seq in sequences]

# Find repeating motifs
pattern, result = re.compile(r"(.+?)\1+"), []
[result.extend(pattern.findall(item)) for item in seq_str]

# Duplicate 1-grams and make set
result_d1 = duplicate_1grams(result)

# Keep one of each pair of palindromes
minus_palindromes = remove_palindromes(result_d1)

# Remove collapsible sequences
clean_list = remove_long_ngrams(minus_palindromes)

# Count how many occurrences of each ngram
counts = {seq: sum(seq in s for s in seq_str) for seq in clean_list}

# Sort dictionary
counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


def dict_keys_to_int(counts, sym_dict):
    # Change keys back to int matching use in syllable_df
    sym_dict_r = {v: k for k, v in sym_dict.items()}
    return {
        str([sym_dict_r[element] for element in key]): value
        for key, value in counts.items()
    }


final_dict = dict_keys_to_int(counts, sym_dict)

# TODO: remove combinations that ocurr very infrequently?

#%%

# * Get wav files

songs = [ast.literal_eval(key) for key in final_dict.keys()]

counts_r = {v: k for k, v in counts.items()}

keys = indv_dfs[bird].key.values
ids = np.array(indv_dfs[bird]["syllables_sequence_id"])
k_keys = [keys[ids == i] for i in np.unique(ids)]

# get song keys for each song type
song_type_keys = {}
for lab in list(counts.keys()):
    length = 0
    keylist = []
    for seq, key in zip(seq_str, k_keys):
        if lab in seq:
            length += len(key[0][0])
            keylist.append(key[0])
    song_type_keys[label] = keylist


# TODO: add song type column to main dataframe, then extract and join spectrograms for further analysis


#%%

# Plot average of each note type
bird = "W99A"
# note_labels = np.unique(indv_dfs[bird].hdbscan_labels.values)
note_labels = [
    note for note in np.unique(indv_dfs["MP69"].hdbscan_labels.values) if note != -1
]


average_specs = {}
fig, ax = plt.subplots(nrows=1, ncols=len(note_labels), figsize=(20, 10))
for col, note in zip(ax, note_labels):
    specs = [
        np.array(spec)
        for spec in indv_dfs[bird][indv_dfs[bird][label] == note].spectrogram.values
    ]
    avg = np.array(np.mean(specs, axis=(0)))
    col.imshow(avg, cmap="bone")
    col.axis("off")


# %%
# TODO: select one of each and plot spec

len_all = len([key for sublist in song_type_keys.values() for key in sublist])
len_unique = len({key for sublist in song_type_keys.values() for key in sublist})


sequ_label = "AA"
key_n = song_type_keys[sequ_label][0]

wav_dir = (
    most_recent_subdirectory(
        DATA_DIR / "processed" / DATASET_ID.replace("_segmented", ""), only_dirs=True,
    )
    / "WAV"
    / key_n
)

data, rate = librosa.core.load(wav_dir, sr=22050)
from src.greti.audio.filter import dereverberate

spec = spectrogram(
    data,
    rate,
    n_fft=512,
    hop_length_ms=3,
    win_length_ms=15,
    ref_level_db=20,
    min_level_db=-40,
)

spec = dereverberate(spec, echo_range=130, echo_reduction=8, hop_length_ms=3)
spec[spec < 0] = 0

plot_spec(
    spec,
    fig=None,
    ax=None,
    rate=rate,
    hop_len_ms=3,
    cmap=plt.cm.viridis,
    show_cbar=True,
    spectral_range=(5000, 10000),
    time_range=None,
    figsize=(20, 6),
)


# %%
# get spec for each song type

indv_dfs[bird]
seq_types = [ast.literal_eval(key) for key in final_dict.keys()]


#%%

np.unique(indv_dfs[bird].key.values)
seq_str

keys = indv_dfs[bird].key.values
label_list = indv_dfs[bird][label].values
index_label = 0


for key, label in zip(keys, label_list):
    if label in songs[index_label]:
        key

length = len(indv_dfs[bird][indv_dfs[bird][label].isin(songs[index_label])])


for index, row in enumerate(
    indv_dfs[bird][indv_dfs[bird][label].isin(songs[index_label])].iloc[:]
):
    if index < 3:
        print(row)

#%%


len_label = len(indv_dfs[bird].loc[indv_dfs[bird][label] == lab].key)
index = random.sample(range(len_label), 1)[0]

key = indv_dfs[bird].loc[indv_dfs[bird][label] == lab].key.iloc[index]

wav_dir = (
    most_recent_subdirectory(
        DATA_DIR / "processed" / DATASET_ID.replace("_segmented", ""), only_dirs=True,
    )
    / "WAV"
    / key
)
wav, rate = librosa.core.load(wav_dir, sr=None)


#%%


labs = indv_dfs[bird][label].values
sequence_ids = np.array(indv_dfs[bird]["syllables_sequence_id"])


#%%

# Compute graph
G = compute_graph(transition_matrix, min_connections=0, column_names=true_label)

# Plot graph

# Set node position
pos = nx.circular_layout(G)
pos.keys()

# Get the colour palette and prepare colour lists
label_palette = sns.color_palette(pal, len(np.unique(labs)))

colours = {}
for note, colour in indv_dict[bird]["label_pal_dict"].items():
    for number, label in indv_dict[bird]["label_dict"].items():
        if label is note:
            colours[number] = colour

pos_colours = []
for i in pos.keys():
    for note, colour in colours.items():
        for original, newlab in element_dict.items():
            if i == newlab:
                if note == original:
                    pos_colours.append(colour)

# Get positions, weights, widths, colours
pos_locs = np.vstack(pos.values())
graph_weights = [G[edge[0]][edge[1]]["weight"] for edge in G.edges()]
rgba_cols = [[0, 0, 0] + [i] for i in graph_weights]
edge_widthds = graph_weights

# Plot
fig, ax = plt.subplots(figsize=(10, 10))

nx.draw(
    G,
    pos,
    connectionstyle="arc3, rad = 0.3",
    with_labels=True,
    ax=ax,
    node_color=pos_colours,
    edge_color=(np.array(rgba_cols) * 0.8) + 0.01,
    width=np.array(edge_widthds),
    arrowstyle="fancy",
    arrowsize=30,
    node_size=800,
)

for item in ax.get_children():
    if not isinstance(item, mpl.patches.FancyArrowPatch):
        continue
    item.set_edgecolor([0, 0, 0, 0])

ax.set_facecolor(facecolour)
ax.set_xticks([])  # remove ticks
ax.set_yticks([])  # remove ticks
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

plt.show()

plt.plot(sorted(edge_widthds))


# %%
# Get longest repeated string for each song
def guess_seq_len(seq):
    guess = 1
    max_len = len(seq) / 2
    for x in range(2, int(max_len)):
        if seq[0:x] == seq[x : 2 * x]:
            return x
    return guess


for song in sequences_newlabels:
    print(guess_seq_len(song), song)


#%%


# %%

n_jobs = -2
verbosity = 10

# create a dataset object
dataset = DataSet(DATASET_ID)
print(len(dataset.data_files))

# List of individual birds
birds = np.array([list(i)[0] for i in dataset.json_indv])
facecolour = "#f2f1f0"


#%%

# for each individual in the dataset

indv_dict = {}
for indv in tqdm(unique_indvs):
    color_lists, trans_lists, label_pal_dict, label_pal, label_dict = indv_barcode(
        indv_dfs[bird], time_resolution=0.03, label=label, pal=pal,
    )
    indv_dict[indv] = {"label_pal_dict": label_pal_dict, "label_dict": label_dict}
    fig, ax = plt.subplots(figsize=(8, 3))
    plot_sorted_barcodes(
        color_lists,
        trans_lists,
        max_list_len=150,
        seq_len=20,
        nex=300,
        figsize=(10, 4),
        ax=ax,
    )

# %%


# %%


for indv in np.unique(birds):
    if indv in unique_indvs:
        for i in range(0, 5):

            # get key for individual
            key = np.array(list(dataset.data_files.keys()))[birds == indv][i]

            # load the wav
            wav, rate = librosa.core.load(
                dataset.data_files[key].data["wav_loc"], sr=None
            )
            # wav = librosa.util.normalize(wav)
            data = butter_bandpass_filter(wav, 1200, 10000, rate)

            # create the spectrogram
            spec = spectrogram(
                data,
                rate,
                n_fft=1024,
                hop_length_ms=3,
                win_length_ms=15,
                ref_level_db=30,
                min_level_db=-50,
            )

            # plot the spectrogram with labels
            fig, ax = plt.subplots(figsize=(100, 15))
            plot_spec(
                spec, fig, ax, hop_len_ms=3, rate=rate, show_cbar=False, cmap="binary"
            )
            ax.set_facecolor(facecolour)
            plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ymin, ymax = ax.get_ylim()
            for ix, row in tqdm(
                syllable_df[syllable_df.key == key].iterrows()
            ):  #! this requires merging dict of birds into a single dataframe, leftover from before: change if needed
                # ax.axvline(row.start_time)
                # ax.axvline(row.end_time)
                color = indv_dict[row.indv]["label_pal_dict"][
                    indv_dict[row.indv]["label_dict"][row[label]]
                ]
                ax.add_patch(
                    mpatches.Rectangle(
                        [row.start_time, ymax - (ymax - ymin) / 10],
                        row.end_time - row.start_time,
                        (ymax - ymin) / 10,
                        ec="none",
                        color=color,
                    )
                )
            # ax.set_xlim([0.7, 9.3])
            ax.xaxis.tick_bottom()
            ensure_dir(FIGURE_DIR / YEAR / "barcode" / DATASET_ID)
            save_fig(
                FIGURE_DIR
                / YEAR
                / "barcode"
                / DATASET_ID
                / (indv + "_" + str(i) + "_spectrogram"),
                save_png=True,
            )


# %%
# Plot barcode plots for a single individual

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_facecolor(facecolour)
plot_sorted_barcodes(
    color_lists, trans_lists, max_list_len=100, seq_len=20, nex=500, ax=ax,
)

fig_out = (
    FIGURE_DIR
    / YEAR
    / "examples"
    / (bird + "_barcode_" + str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")) + ".png")
)

# ensure_dir(fig_out)
# plt.savefig(
#     fig_out, dpi=300, bbox_inches="tight", pad_inches=0.1, transparent=False,
# )
# plt.close()

plt.show()

# %%

# Plot an spectrogram along with the inferred notes - for a subset of the songs of a single bird
# len(syllable_df[syllable_df.indv == bird].key.unique())

for i in range(8, 10):

    # get key for individual
    key = np.array(list(dataset.data_files.keys()))[birds == bird][i]

    # load the wav
    wav, rate = librosa.core.load(dataset.data_files[key].data["wav_loc"], sr=None)
    # wav = librosa.util.normalize(wav)
    data = butter_bandpass_filter(wav, 1200, 10000, rate)

    # create the spectrogram
    spec = spectrogram(
        data,
        rate,
        n_fft=1024,
        hop_length_ms=3,
        win_length_ms=15,
        ref_level_db=30,
        min_level_db=-60,
    )

    from src.greti.audio.filter import dereverberate

    spec = dereverberate(spec, echo_range=100, echo_reduction=3, hop_length_ms=3)
    spec[spec < 0] = 0

    # plot the spectrogram with labels
    fig, ax = plt.subplots(figsize=(len(data) * 0.00009, 3))
    plot_spec(spec, fig, ax, hop_len_ms=3, rate=rate, show_cbar=False, cmap="binary")
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ymin, ymax = ax.get_ylim()
    ax.set_facecolor(facecolour)
    for ix, row in tqdm(syllable_df[syllable_df.key == key].iterrows()):
        if row.hdbscan_labels >= 0:  # don't plot noise

            color = indv_dict[row.indv]["label_pal_dict"][
                indv_dict[row.indv]["label_dict"][row[label]]
            ]
            ax.add_patch(
                mpatches.Rectangle(
                    [row.start_time, ymax - (ymax - ymin) / 10],
                    row.end_time - row.start_time,
                    (ymax - ymin) / 10,
                    ec="none",
                    color=color,
                )
            )
    # ax.set_xlim([0.7, 9.3])
    ax.xaxis.tick_bottom()

    fig_out = (
        FIGURE_DIR
        / YEAR
        / "barcode"
        / DATASET_ID
        / (bird + "_" + str(i) + "_spectrogram" + ".png")
    )

    # ensure_dir(fig_out)
    # plt.savefig(
    #     fig_out, dpi=300, bbox_inches="tight", pad_inches=0.1, transparent=False,
    # )
    # plt.close()
    plt.show()
