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
from itertools import chain, repeat
import pickle

from src.avgn.signalprocessing.create_spectrogram_dataset import prepare_wav, make_spec
from librosa.display import specshow


# from sklearn.cluster import MiniBatchKMeans
# from cuml.manifold.umap import UMAP as cumlUMAP

# import importlib
# importlib.reload(src)

get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

# %%


def build_indv_dict(bird, label, pal):
    """Returns a dictionary of note labels and colours
    """
    indv_dict = {}
    color_lists, trans_lists, label_pal_dict, label_pal, label_dict = indv_barcode(
        indv_dfs[bird], time_resolution=0.02, label=label, pal=pal,
    )
    indv_dict[bird] = {"label_pal_dict": label_pal_dict, "label_dict": label_dict}
    return indv_dict


def list_sequences(bird, label, min_cluster_samples=10):
    """Returns a lists of lists where each list contains the sequence of notes in one song
    """

    # TODO: Optimise this
    labs = indv_dfs[bird][label].values
    sequence_ids = np.array(indv_dfs[bird]["syllables_sequence_id"])
    sequences = [labs[sequence_ids == i] for i in np.unique(sequence_ids)]

    transition_matrix, element_dict, drop_list = build_transition_matrix(
        sequences, min_cluster_samples=10
    )

    element_dict_r = {v: k for k, v in element_dict.items()}

    # true_label = get_true_labels(transition_matrix, drop_list)

    ## Transition matrix in DataFrame format, with true labels
    # matrix = pd.DataFrame(transition_matrix, columns=true_label, index=true_label)
    # sns.heatmap(matrix)

    # Substitute labels
    sequences_newlabels = [list(seq) for seq in sequences]

    return sequences_newlabels, labs


def find_song_types(sequences_newlabels, labs, bird, label, plot=False):
    """Finds repeating notes and returns a dictionary of average song type spectrograms
    """
    # Convert int labels to symbolic labels
    sym_dict = dict_keys_to_symbol(labs)
    sequences = [[sym_dict[b] for b in i] for i in sequences_newlabels]

    # Find n-grams
    # Convert lists to strings
    seq_str = ["".join(map(str, seq)) for seq in sequences]

    # Find repeating motifs;
    pattern = re.compile(r"(.+?)(?=\1)")
    result = {
        seq: duplicate_1grams(pattern.findall(seq))
        if len(pattern.findall(seq)) > 0
        else [seq]
        for seq in seq_str
    }

    # Frequencies for each combination
    tally = get_seq_frequencies(result, seq_str)

    # Symbols to original labels
    final_dict = dict_keys_to_int(tally, sym_dict)

    # Build dic tionary with spectrograms for all combinations
    spec_dict = get_all_spec_combinations(final_dict, bird, label)

    # Plot test
    if plot is True:
        plt.figure(figsize=(10, 4))
        plt.imshow(spec_dict[list(spec_dict.keys())[-1]], cmap="bone", origin="lower")
        col.axis("off")

    return spec_dict


def get_seq_frequencies(result, seq_str):
    """Returns a sorted dictionary of sequence frequencies in the entire output of an individual bird
    Args:
        result (dict): A dictionary with repeated sequencies for each song
    Returns:
        dict: Sorted dictionary
    """
    unique_seqs = set([seq for subset in list(result.values()) for seq in subset])
    tally = {}
    for seq in unique_seqs:
        n = 0
        for element in seq_str:
            n += element.count(seq)
        tally[seq] = n
    tally = dict(sorted(tally.items(), key=lambda x: x[1], reverse=True))
    return tally


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
    sym_list = list(string.ascii_uppercase[0:nlabs])
    symbolic_dict = {
        lab : (symbol if lab != -1 else 'Z')
        for lab, symbol in zip(unique_labs, sym_list)
    }
    return symbolic_dict



def dict_keys_to_int(counts, sym_dict):
    # Change keys back to int matching use in syllable_df
    sym_dict_r = {v: k for k, v in sym_dict.items()}
    return {
        str([sym_dict_r[element] for element in key]): value
        for key, value in counts.items()
    }


def get_average_note(bird, label, note):
    """Returns the average of all spectrograms of a same note type
    """
    specs = [
        np.array(spec)
        for spec in indv_dfs[bird][indv_dfs[bird][label] == note].spectrogram.values
    ]
    avg = np.array(np.mean(specs, axis=(0)))
    return avg


def get_all_spec_combinations(final_dict, bird, label):
    """Returns a dictionary with average spectrograms for each note combination found in a bird's repertoire
    Returns:
        dict: keys are sequences, values are average combined spectrograms
    """
    seq_combinations = [ast.literal_eval(key) for key in final_dict.keys()]
    spec_dict = {}
    for notes in seq_combinations:
        spec = []
        for note in notes:
            average = get_average_note(bird, label, note)
            spec.append(average)
        spec = np.concatenate(spec, axis=1)
        spec_dict[str(notes)] = spec.astype(np.uint8)
    return spec_dict


# %%

DATASET_ID = "GRETI_HQ_2020_segmented"
YEAR = "2020"

dfs_dir = DATA_DIR / "indv_dfs" / DATASET_ID
indv_dfs = pd.read_pickle(dfs_dir / (f"{DATASET_ID}_labelled_checked.pickle")) #! Change to this once manual check done
indvs = list(indv_dfs.keys())

exclude = ['CP34', 'EX38A', 'SW119', 'B11', 'MP20', 'B161', 'EX34', 'CP28', 'B49', 'SW116'] # Data for these are too noisy, judged from UMAP scatterplot in previous step
indvs = [indv for indv in indvs if indv not in exclude]


#%%
# If testing before manual check of labels use the below instead
# indv_dfs = pd.read_pickle(
#     "/media/nilomr/SONGDATA/syllable_dfs/GRETI_HQ_2020_segmented/GRETI_HQ_2020_segmented_with_labels.pickle"
# )
# indv_dfs = {indv: indv_dfs[indv_dfs.indv == indv] for indv in indv_dfs.indv.unique()}
# indvs = list(indv_dfs.keys())

#%%

# Prepare colour palettes
# label = "hdbscan_labels" # unchecked
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
unique_indvs = list(indv_dfs.keys())


# %%
# Select one bird to test
#bird = "EX21"
cluster_labels = "hdbscan_labels_fixed"

#TODO: remove seqs that arise due to undetected (not noise) note in the middle
# Otherwise this function ready

def find_syllable_sequences(bird, cluster_labels, remove_noise=True, remove_redundant=True):
    # Define sequences
    sequences_newlabels, labs = list_sequences(bird, cluster_labels, min_cluster_samples=5)
    sym_dict = dict_keys_to_symbol(labs)
    sequences = [[sym_dict[b] for b in i] for i in sequences_newlabels]

    # Find n-grams
    # Convert lists to strings
    seq_str = ["".join(map(str, seq)) for seq in sequences]
    seq_str_dict = {key : seq for key, seq in enumerate(seq_str)}

    # Find repeating motifs;
    pattern = re.compile(r"(.+?)(?=\1)")
    result = {
        key : (duplicate_1grams(pattern.findall(seq))
        if len(pattern.findall(seq)) > 0
        else [seq])
        for key, seq in seq_str_dict.items()
    }

    # Frequencies for each combination
    tally = get_seq_frequencies(result, seq_str)

    if remove_noise:
        # Remove sequences with noise (-1 labels)
        tally = {key : n for key, n in tally.items() if 'Z' not in key}

    if remove_redundant:
        # Remove sequences already contained in other, shorter sequences (where the longest of the pair does not have new notes)
        tally = {k:v for k,v in tally.items() if k in remove_long_ngrams([key for key in tally.keys()])}

    # Build dictionary of songs containing each sequence (allows duplicates)
    song_dict = {}
    for sequence, n in tally.items():
        seqlist = []
        for key, seqs in result.items():
            if len(seqs) > 1:
                for seq in seqs:
                    if seq == sequence:
                        seqlist.append(key)
            else:
                if seqs == sequence:
                    seqlist.append(key)
                
        song_dict[sequence] = seqlist

    # Symbols to original labels
    final_dict = dict_keys_to_int(song_dict, sym_dict)

    return final_dict, len(sequences_newlabels), sequences_newlabels, dict_keys_to_int(tally, sym_dict)

#%%



#%%


def create_melspec(hparams, data, sr, logscale=True):
    Sb = librosa.feature.melspectrogram(data, 
                                       sr=sr, 
                                       n_mels=hparams.num_mel_bins,
                                       hop_length=hparams.hop_length_ms,
                                       n_fft=hparams.n_fft,
                                       fmin=hparams.butter_lowcut,
                                       fmax=(sr // 2))

    if logscale:
        Sb = librosa.power_to_db(Sb, ref=np.max)

    Sb = Sb.astype(np.float32)
    return Sb

def display_melspec(hparams, mels, sr): 
    specshow(mels, x_axis='time', y_axis='mel',
                             sr=sr, hop_length=hparams.hop_length_ms,
                             fmin=hparams.butter_lowcut, fmax=(sr // 2))
    plt.colorbar()
    plt.show()

def subfinder(mylist, pattern):
    matches = []
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
            matches.append([i, i+len(pattern)-1])
    return matches

def right_pad_spec(spec, maxlen, values='minimum', constant_values=(-0.4)):
    pad = maxlen - spec.shape[1]
    spec = np.pad(spec, ((0, 0), (0, pad)), values, constant_values=constant_values)
    return spec


import operator

def get_max_list_in_list(list_of_lists):
    all_dict = {i:np.max(l) for i, l in enumerate(list_of_lists)}
    index = max(all_dict.items(), key=operator.itemgetter(1))[0]
    return list_of_lists[index][0] if isinstance(list_of_lists[index][0], list) else list_of_lists[index] # this works but I don't really know why, check at some point



hparams = HParams(
    num_mel_bins=200,
    n_fft=1024,
    win_length_ms=1000,
    hop_length_ms=100,
    butter_lowcut=1200,
    butter_highcut=10000,
    ref_level_db=30,
    min_level_db=-20,
    mask_spec=True,
)

#%%
#!

syllable_type_dict = {}

for indv in tqdm(indvs[2:4], desc="birds processed", leave=True):

    final_dict, n_seqs, _, _ = find_syllable_sequences(indv, cluster_labels, remove_noise=True, remove_redundant=True)
    # remove keys that do not appear in at least 2 songs
    # final_dict = {key:val for key, val in final_dict.items() if len(val) > 1}

    syll_audio = {}

    index_list = [item for sublist in list(final_dict.values()) for item in sublist]

    for song in tqdm(index_list, desc="Getting syllable audio data and metadata; building mel spectrograms", leave=True):

        # get audio key, starts and ends for this song
        file_key = indv_dfs[indv][indv_dfs[indv].syllables_sequence_id == song].key[0]
        indv_dfs[indv][indv_dfs[indv].key == file_key]
        starts = indv_dfs[indv][indv_dfs[indv].key == file_key].start_time.values
        ends = indv_dfs[indv][indv_dfs[indv].key == file_key].end_time.values

        # load song audio
        wav_loc = most_recent_subdirectory(DATA_DIR /'processed'/ DATASET_ID.replace('_segmented', ''), only_dirs=True) / 'WAV' / file_key
        data, sr = librosa.load(wav_loc, sr = 32000)
        data = butter_bandpass_filter(
            data, hparams.butter_lowcut, hparams.butter_highcut, sr, order=5
        )

        # get sequences present in song
        seqlist = []
        for seq, keys in final_dict.items():
            for key in keys:
                if key == song:
                    seqlist.append(ast.literal_eval(seq))
        seqlist = seqlist if any(isinstance(el, list) for el in seqlist) else [seqlist[0]]

        # list labels in this song
        labels = list(indv_dfs[indv][indv_dfs[indv].key == file_key].hdbscan_labels_fixed)

        # Add instances of each sequence to dictionary

        for seq in seqlist:
            positions = subfinder(labels, seq) # get starts and ends of syllables
            sts = starts[[position[0] for position in positions]]
            ets = ends[[position[1] for position in positions]]

            for st, et in zip(sts, ets):
                audio = data[int(st * sr) : int(et * sr)]

                if f'{seq}' in syll_audio:
                    syll_audio[f'{seq}'].append([list(audio)])
                else:
                    syll_audio[f'{seq}'] = [list(audio)]

    indv_songs = {}

    mfcc_ = False

    # keep clearest syllable and make mel spectrogram
    for seq, specs in syll_audio.items():
        if len(specs) > 2: # discard if only one - reduce cases of noisy segmentation
            try:
                exemplar = np.array(get_max_list_in_list(specs))
                if mfcc_:
                    indv_songs[seq] = librosa.feature.mfcc(exemplar, sr, n_mfcc=13)
                else:
                    indv_songs[seq] = librosa.util.normalize(create_melspec(hparams, exemplar, sr, logscale=True))
            except:
                print('Problema')
        else:
            continue

    syllable_type_dict[indv] = indv_songs

# pickle.dump(syllable_type_dict, open(dfs_dir / (f"{DATASET_ID}_song_types.pickle"), "wb"))

#%%
# Load data
syllable_type_dict = pd.read_pickle(dfs_dir / (f"{DATASET_ID}_song_types.pickle"))

#%%
# Prepare data for DTW distance calculation
bird_list = []
syllable_specs = []

for indv, repertoire in syllable_type_dict.items():
    for key, spec in repertoire.items():
        syllable_specs.append(spec)
        bird_list.append(indv)

all_specs = [spec.T for spec in tqdm(syllable_specs)]

#%%
# DTW distance matrix
from dtaidistance import dtw
ds = dtw.distance_matrix_fast(all_specs)
ds = ds/(ds.max()/1)

#%%
# Slower test
from scipy.spatial.distance import cosine
from fastdtw import fastdtw
dist = lambda p1, p2: fastdtw(p1.T, p2.T, dist=cosine)[0]
dm = np.asarray([[dist(p1, p2) for p2 in syllable_specs] for p1 in tqdm(syllable_specs)])

#%% 
# all sylls

dist_df = pd.DataFrame(np.tril(prox_mat, k=-1), columns=bird_list, index=bird_list)
y = list(dist_df.stack())

# Build matrix of distances in metres
coords = [
    list(nestboxes[nestboxes["nestbox"] == bird].east_north.values[0]) for bird in bird_list
]
spatial_dist = distance.cdist(coords, coords)
spatial_dist_df = pd.DataFrame(np.tril(spatial_dist, k=-1), columns=bird_list, index=bird_list)
x = list(spatial_dist_df.stack())


#%%
# indvs = indvs[20:30]

length = len(np.unique(bird_list))
birds = np.unique(bird_list)

dist_df = pd.DataFrame(np.tril(prox_mat, k=-1), columns=bird_list, index=bird_list)

matrix = np.zeros((length, length))

for index, indv_1 in tqdm(enumerate(birds), desc="Building distance matrix", leave=True):
    for index2, indv_2 in enumerate(birds):
        if indv_1 == indv_2:
            matrix[index, index] = 0
        elif matrix[index, index2] == 0 and matrix[index, index2] == 0:
            mean_dist = np.mean(np.mean(dist_df[indv_1].loc[indv_2]))
            matrix[index, index2] = mean_dist
            matrix[index2, index] = mean_dist

# sim matrix
# matrix = 1 - (matrix / np.max(matrix))
distances_df = pd.DataFrame(np.tril(matrix, k=-1), columns=birds, index=birds)
y = list(distances_df.stack())

#%%
from scipy.spatial import distance

# Import nestbox coordinates
from src.greti.read.paths import RESOURCES_DIR

coords_file = RESOURCES_DIR / "nestboxes" / "nestbox_coords.csv"
tmpl = pd.read_csv(coords_file)
nestboxes = tmpl[tmpl["nestbox"].isin(birds)]
nestboxes["east_north"] = nestboxes[["x", "y"]].apply(tuple, axis=1)

# Build matrix of distances in metres
coords = [
    list(nestboxes[nestboxes["nestbox"] == bird].east_north.values[0]) for bird in birds
]
spatial_dist = distance.cdist(coords, coords)
spatial_dist_df = pd.DataFrame(np.tril(spatial_dist, k=-1), columns=birds, index=birds)
x = list(spatial_dist_df.stack())

# %%
# Plot acoustic distance vs spatial distance
# y_norm = list((np.array(y) - min(y))/(max(y)-min(y)))

df = pd.DataFrame({"s_dist": x, "a_dist": y})
#df = df[df["a_dist"] != 0]
#df = df[df["s_dist"] < 1250]
df = df[df["s_dist"] != 0]

fig_dims = (5, 5)
fig, ax = plt.subplots(figsize=fig_dims)
sns.regplot(
    x="s_dist", y="a_dist", data=df, marker="o", scatter_kws={"s": 7, "alpha": 0.01}
)
#plt.yscale("log")

fig_dims = (5, 5)
fig, ax = plt.subplots(figsize=fig_dims)
sns.regplot(x="s_dist", y="a_dist", data=df, x_bins=10)

fig_dims = (5, 5)
fig, ax = plt.subplots(figsize=fig_dims)
sns.regplot(x="s_dist", y="a_dist", data=df, lowess=True, scatter=True, scatter_kws={"s": 7, "alpha": 0.01})
#plt.yscale("log")

#%%
# dtw

from dtaidistance import dtw
#all_specs_padded = [right_pad_spec(spec, maxlen).flatten() for spec in tqdm(syllable_specs, desc="padding specs", leave=True)]
ds = dtw.distance_matrix_fast(all_specs)

#%%
# Test DTW

from dtw import dtw
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

for i in range(len(syllable_specs)):
    dist, cost, acc_cost, path = dtw(syllable_specs[0].T, syllable_specs[i].T, dist=cosine)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle(dist)
    ax1.imshow(cost.T, origin='lower', cmap=plt.cm.gray_r, interpolation='nearest')
    ax1.plot(path[0], path[1], 'w')
    ax2.imshow(syllable_specs[0], origin='lower', cmap=plt.cm.gray, interpolation='nearest')
    ax3.imshow(syllable_specs[i], origin='lower', cmap=plt.cm.gray, interpolation='nearest')
    ax1.set_xlim([-0.5, cost.shape[0]-0.5])
    ax1.set_ylim([-0.5, cost.shape[1]-0.5])

    for ax in [ax1, ax2, ax3]:
        ax.axis('off')

#%%
# Build DTW distance matrix
from dtw import dtw
dist = lambda p1, p2: dtw(p1.T, p2.T, dist=cosine)[0]
dm = np.asarray([[dist(p1, p2) for p2 in syllable_specs] for p1 in tqdm(syllable_specs)])

#%%
from fastdtw import fastdtw
dist = lambda p1, p2: fastdtw(p1.T, p2.T, dist=cosine)[0]
dm = np.asarray([[dist(p1, p2) for p2 in syllable_specs] for p1 in tqdm(syllable_specs)])


#%%


#%%
from fastdtw import fastdtw
distance, path = fastdtw(all_specs_padded[0], all_specs_padded[1], dist=cosine)
from scipy.spatial.distance import cosine

all_specs_padded = [right_pad_spec(spec, maxlen) for spec in tqdm(syllable_specs, desc="padding specs", leave=True)]
distance, path = fastdtw(all_specs_padded[0], all_specs_padded[1], dist=cosine)

for index, indv_1 in tqdm(enumerate(indvs), desc="Building distance matrix", leave=True):
    for index2, indv_2 in enumerate(indvs):
        if indv_1 == indv_2:
            matrix[index, index] = 0
        elif matrix[index, index2] == 0 and matrix[index, index2] == 0:
            mean_dist = np.mean(np.mean(dist_df[indv_1].loc[indv_2]))
            matrix[index, index2] = mean_dist
            matrix[index2, index] = mean_dist

#%%
# Plot all sequences
n = 0
for bird, dic in syllable_type_dict.items():
    print(bird, len(dic))
    n +=1
    if n == 10:
        break
    ncols = len(dic.keys())
    fig, axes = plt.subplots(1, ncols, figsize=(20, 4))
    plt.suptitle(bird)
    for (key, spec), (i, ax) in zip(dic.items(), enumerate(axes)):
        ax.imshow(spec, origin='lower',  aspect='equal')
        ax.axis('off')
        ax.title.set_text(str(i) + " " + key)

#%%

#TODO =====================
#* To pad
# check distribution of sequence lengths and pad/trim to a reasonable point

#!#################### URF TESTS HERE ################################

from URF.main import random_forest_cluster, plot_cluster_result


def pad_spectrogram(spectrogram, pad_length):
    """ Pads a spectrogram to being a certain length
    """
    excess_needed = pad_length - np.shape(spectrogram)[1]
    pad_left = np.floor(float(excess_needed) / 2).astype("int")
    pad_right = np.ceil(float(excess_needed) / 2).astype("int")
    return np.pad(
        spectrogram, [(0, 0), (pad_left, pad_right)], "constant", constant_values=0
    )


timelengths = [spec.shape[1] for spec in syllable_specs]
sns.distplot(timelengths) # check dist of lengths
pad_length = 40
padded_specs = [pad_spectrogram(spec, pad_length) if spec.shape[1] < pad_length else spec[:,:40] for spec in syllable_specs]

specs = [i.flatten() for i in padded_specs]
#%%

clf, prox_mat, cluster_ids = random_forest_cluster(np.array(specs), k=100, max_depth=None, n_estimators = 300, random_state=0)


#%%
# Dist matrix for all song types

from scipy.spatial import distance
from joblib import Parallel, delayed
from src.avgn.signalprocessing.create_spectrogram_dataset import pad_spectrogram

syll_lens = [np.shape(i)[1] for i in syllable_specs]
pad_length = np.max(syll_lens)


def pad_spectrogram(spectrogram, pad_length):
    """ Pads a spectrogram to being a certain length
    """
    excess_needed = pad_length - np.shape(spectrogram)[1]
    pad_left = np.floor(float(excess_needed) / 2).astype("int")
    pad_right = np.ceil(float(excess_needed) / 2).astype("int")
    return np.pad(
        spectrogram, [(0, 0), (pad_left, pad_right)], "constant", constant_values=-0.4
    )


with Parallel(n_jobs=-2, verbose=2) as parallel:
    all_specs_padded = parallel(
        delayed(pad_spectrogram)(spec, pad_length)
        for spec in tqdm(
            syllable_specs, desc="padding spectrograms", leave=False
        )
    )


specs = flatten_spectrograms([i for i in all_specs_padded])
#%%
m = distance.cdist(specs, specs, "cosine")
dist_df = pd.DataFrame(np.tril(m, k=-1), columns=bird_list, index=bird_list)

#%%
# Dist matrix (pairwise average distances)
length = len(np.unique(bird_list))
birds = np.unique(bird_list)

matrix = np.zeros((len(birds), len(birds)))
for index, indv_1 in tqdm(enumerate(birds), desc="Building distance matrix", leave=True):
    for index2, indv_2 in enumerate(birds):
        if indv_1 == indv_2:
            matrix[index, index] = 0
        elif matrix[index, index2] == 0 and matrix[index, index2] == 0:
            mean_dist = np.mean(np.mean(dist_df[indv_1].loc[indv_2]))
            matrix[index, index2] = mean_dist
            matrix[index2, index] = mean_dist

# sim matrix
# matrix = 1 - (matrix / np.max(matrix))
distances_df = pd.DataFrame(np.tril(matrix, k=-1), columns=birds, index=birds)
y = list(distances_df.stack())



#%%

scatter_labs = list(
    chain.from_iterable(
        repeat(indv, len(values)) for indv, values in syllable_type_dict.items()
    )
)
scatter_labs = [item[0] for item in scatter_labs]


# colourcodes
#%%


#%%
# project
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from src.avgn.signalprocessing.create_spectrogram_dataset import pad_spectrogram

syll_lens = [np.shape(i)[1] for i in syllable_specs]
pad_length = np.max(syll_lens)


def pad_spectrogram(spectrogram, pad_length):
    """ Pads a spectrogram to being a certain length
    """
    excess_needed = pad_length - np.shape(spectrogram)[1]
    pad_left = np.floor(float(excess_needed) / 2).astype("int")
    pad_right = np.ceil(float(excess_needed) / 2).astype("int")
    return np.pad(
        spectrogram, [(0, 0), (pad_left, pad_right)], "constant", constant_values=-0.4
    )


with Parallel(n_jobs=-2, verbose=2) as parallel:
    all_specs_padded = parallel(
        delayed(pad_spectrogram)(spec, pad_length)
        for spec in tqdm(
            syllable_specs, desc="padding spectrograms", leave=False
        )
    )


specs = flatten_spectrograms([i for i in all_specs_padded])

#%%

# First build a dtw with `dtw_metric = build_dtw_mse(x[0].shape),
# then umap.UMAP(metric=dtw_metric).fit_transform(x.reshape(len(x), -1))



#%%
import umap

# PCA
# pca2 = PCA(n_components=2)
# pca_embed = pca2.fit_transform(specs)

# UMAP
umap_parameters = {
    "n_neighbors": 60,
    "min_dist": 0.8,
    "n_components": 2,
    "verbose": True,
    "init": "spectral",
    "low_memory": True,
}

#all_specs_padded = np.array(all_specs_padded)


# dtw_metric = build_dtw_mse(all_specs_padded[0].shape)
# umap_embed = umap.UMAP(metric=dtw_metric, **umap_parameters).fit_transform(all_specs_padded.reshape(len(all_specs_padded), -1))

fit = umap.UMAP(**umap_parameters)
umap_embed =fit.fit_transform(specs)


#%%
plt.figure(figsize=(12, 12))
sns.scatterplot(umap_embed[:, 0], umap_embed[:, 1], hue=bird_list)

# %%

scatter_spec(
    umap_embed,
    specs=all_specs_padded,
    column_size=9,
    # x_range = [-5.5,7],
    # y_range = [-10,10],
    pal_color="hls",
    color_points=False,
    enlarge_points=30,
    range_pad=0.1,
    figsize=(15, 15),
    scatter_kwargs={
        "labels": scatter_labs,
        "alpha": 0.60,
        "s": 5,
        "color_palette": pal,
        "show_legend": False,
    },
    matshow_kwargs={"cmap": plt.cm.Greys},
    line_kwargs={"lw": 1, "ls": "solid", "alpha": 0.11},
    draw_lines=True,
    border_line_width=0,
    facecolour="#f2f1f0",
)

#%%

fig, ax = plt.subplots(1, figsize=(10, 10))
sns.kdeplot(
    list(umap_embed), levels=12, shade=True, cmap="inferno", ax=ax,
)

ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()


#%%
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

final_dict = dict_keys_to_int(counts, sym_dict)

# TODO: remove combinations that ocurr very infrequently?

# %%



#%%


#%%


def get_average_note(bird, label, note):
    specs = [
        np.array(spec)
        for spec in indv_dfs[bird][indv_dfs[bird][label] == note].spectrogram.values
    ]
    avg = np.array(np.mean(specs, axis=(0)))
    return avg


# Plot average of each note type
note_labels = [
    note for note in np.unique(indv_dfs[bird].hdbscan_labels.values) if note != -1
]
fig, ax = plt.subplots(nrows=1, ncols=len(note_labels), figsize=(10, 4))
for col, note in zip(ax, note_labels):
    avg = get_average_note(bird, label, note)
    col.imshow(avg, cmap="bone", origin="lower")
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
            return x, seq[0:x]
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
for indv in tqdm(unique_indvs, desc="clustering individuals", leave=True)):
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
