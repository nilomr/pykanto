# %%
import pickle
from datetime import datetime

import hdbscan
import librosa
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phate
import seaborn as sns
import umap
from joblib import Parallel, delayed
from matplotlib.collections import PatchCollection
from scipy.spatial import distance
from tqdm.autonotebook import tqdm

from src.avgn.dataset import DataSet
from src.avgn.signalprocessing.create_spectrogram_dataset import (
    flatten_spectrograms,
    log_resize_spec,
)
from src.avgn.utils.general import save_fig
from src.avgn.utils.hparams import HParams
from src.avgn.utils.paths import ensure_dir, most_recent_subdirectory
from src.avgn.visualization.barcodes import indv_barcode, plot_sorted_barcodes
from src.avgn.visualization.network_graph import plot_network_graph
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

# from sklearn.cluster import MiniBatchKMeans
# from cuml.manifold.umap import UMAP as cumlUMAP

# import importlib
# importlib.reload(src)


# %%

DATASET_ID = "GRETI_HQ_2020_segmented"
YEAR = "2020"

syllable_df = pd.concat(
    [
        pd.read_pickle(i)
        for i in list((DATA_DIR / "indv_dfs" / DATASET_ID).glob("*.pickle"))
    ]
)

#%%
label = "hdbscan_labels"
facecolour = "#f2f1f0"
colours = [
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

palette = sns.set_palette(sns.color_palette(colours))
unique_indvs = syllable_df.indv.unique()[:2]


#%%

# for each individual in the dataset
indv_dict = {}
for indv in tqdm(unique_indvs):
    color_lists, trans_lists, label_pal_dict, label_pal, label_dict = indv_barcode(
        syllable_df[syllable_df.indv == indv],
        time_resolution=0.03,
        label=label,
        pal=palette,
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
# 1 ind only

indv_dict = {}
bird = "SW5"

color_lists, trans_lists, label_pal_dict, label_pal, label_dict = indv_barcode(
    syllable_df[syllable_df.indv == bird],
    time_resolution=0.01,
    label=label,
    pal=palette,
)
indv_dict["SW5"] = {"label_pal_dict": label_pal_dict, "label_dict": label_dict}
fig, ax = plt.subplots(figsize=(35, 10))
ax.set_facecolor(facecolour)
plot_sorted_barcodes(
    color_lists,
    trans_lists,
    max_list_len=300,
    seq_len=5,
    nex=100,
    figsize=(10, 4),
    ax=ax,
)

fig_out = (
    FIGURE_DIR
    / YEAR
    / "examples"
    / (bird + "_barcode_" + str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")) + ".png")
)

ensure_dir(fig_out)
plt.savefig(
    fig_out, dpi=300, bbox_inches="tight", pad_inches=0.1, transparent=False,
)
# plt.show()
plt.close()


# %%

DATASET_ID = "GRETI_HQ_2020_segmented"

n_jobs = -2
verbosity = 10

hparams = HParams(
    num_mel_bins=64,
    n_fft=1024,
    win_length_ms=15,
    hop_length_ms=3,
    mel_lower_edge_hertz=1200,
    mel_upper_edge_hertz=10000,
    butter_lowcut=1200,
    butter_highcut=10000,
    ref_level_db=30,
    min_level_db=-19,
    mask_spec=True,
    n_jobs=-2,
    verbosity=1,
    nex=-1,
)


# %%

# create a dataset object
dataset = DataSet(DATASET_ID, hparams=hparams)
print(dataset.sample_json)
print(len(dataset.data_files))

# %%

birds = np.array([list(i)[0] for i in dataset.json_indv])
facecolour = "#f2f1f0"
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
            for ix, row in tqdm(syllable_df[syllable_df.key == key].iterrows()):
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

for i in range(0, 30):

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
        min_level_db=-50,
    )

    # plot the spectrogram with labels
    fig, ax = plt.subplots(figsize=(15, 3))
    plot_spec(spec, fig, ax, hop_len_ms=3, rate=rate, show_cbar=False, cmap="binary")
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ymin, ymax = ax.get_ylim()
    ax.set_facecolor(facecolour)
    for ix, row in tqdm(syllable_df[syllable_df.key == key].iterrows()):
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

    fig_out = (
        FIGURE_DIR
        / YEAR
        / "barcode"
        / DATASET_ID
        / (bird + "_" + str(i) + "_spectrogram" + ".png")
    )

    ensure_dir(fig_out)
    plt.savefig(
        fig_out, dpi=300, bbox_inches="tight", pad_inches=0.1, transparent=False,
    )
    # plt.show()
    plt.close()

# %%
