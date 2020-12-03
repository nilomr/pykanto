# %%
# from IPython import get_ipython

# get_ipython().run_line_magic("env", "CUDA_DEVICE_ORDER=PCI_BUS_ID")
# get_ipython().run_line_magic("env", "CUDA_VISIBLE_DEVICES=2")
%load_ext autoreload
%autoreload 2
# get_ipython().run_line_magic("matplotlib", "inline")

from os import error, wait
import pickle
from datetime import datetime

import hdbscan
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.shape_base import column_stack
import pandas as pd
import phate
import seaborn as sns
import umap
from joblib import Parallel, delayed
from scipy.spatial import distance
from sklearn.decomposition import PCA
from src.avgn.signalprocessing.create_spectrogram_dataset import (
    create_syllable_df, flatten_spectrograms, log_resize_spec)
from src.avgn.utils.general import save_fig
from src.avgn.utils.paths import ensure_dir, most_recent_subdirectory
from src.avgn.visualization.network_graph import plot_network_graph
from src.avgn.visualization.projections import (scatter_projections,
                                                scatter_spec)
from src.avgn.visualization.quickplots import (draw_projection_plots,
                                               quad_plot_syllables)
from src.avgn.visualization.spectrogram import draw_spec_set
from src.greti.read.paths import DATA_DIR, FIGURE_DIR, RESOURCES_DIR
from tqdm.autonotebook import tqdm

# from sklearn.cluster import MiniBatchKMeans


# import importlib
# importlib.reload(src)

# %%

# ### get data

DATASET_ID = "GRETI_HQ_2020_segmented"
YEAR = "2020"

note_df_dir = (
    most_recent_subdirectory(DATA_DIR / "syllable_dfs" / DATASET_ID, only_dirs=True) / "{}.pickle".format(DATASET_ID)
)  # This gets the last dataframe generated by the previous script (in /prepare-data)
syllable_df = pd.read_pickle(note_df_dir)

# %%

indvs = [
    ind
    for ind in syllable_df.indv.unique()[5:9]  #!!!! Remove subsetting !!!!
    if len(syllable_df[syllable_df.indv == ind])
    > 80  # This threshold is based on the need to have clusters >1 member
]

len(indvs)

# %%
# Number of syllables per nest - will need this later
syllable_n = pd.Series(
    [len(syllable_df[syllable_df.indv == ind]) for ind in syllable_df.indv.unique()]
)

# %%
# Colours

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

pal = sns.set_palette(sns.color_palette(colours))

# %%
# # Projections (for clustering and visualisation)
# + Note sequence information

indv_dfs = {}

for indvi, indv in enumerate(tqdm(indvs)):

    indv_dfs[indv] = syllable_df[syllable_df.indv == indv]
    indv_dfs[indv] = indv_dfs[indv].sort_values(by=["key", "start_time"])
    print(indv, len(indv_dfs[indv]))
    specs = [i for i in indv_dfs[indv].spectrogram.values]

    with Parallel(n_jobs=-2, verbose=2) as parallel:
        specs = parallel(
            delayed(log_resize_spec)(spec, scaling_factor=8)
            for spec in tqdm(specs, desc="scaling spectrograms", leave=False)
        )

    # Add note sequences to dataframe for later use
    indv_dfs[indv]["syllables_sequence_id"] = None
    indv_dfs[indv]["syllables_sequence_pos"] = None
    for ki, key in enumerate(indv_dfs[indv].key.unique()):
        indv_dfs[indv].loc[indv_dfs[indv].key == key, "syllables_sequence_id"] = ki
        indv_dfs[indv].loc[
            indv_dfs[indv].key == key, "syllables_sequence_pos"
        ] = np.arange(np.sum(indv_dfs[indv].key == key))

    specs_flattened = flatten_spectrograms(specs)

    # # PHATE
    # phate_operator = phate.PHATE(n_jobs=-1, knn=5, decay=None, t=110, gamma=0)
    # z = list(phate_operator.fit_transform(specs_flattened))
    # indv_dfs[indv]["phate"] = z

    # # PHATE_cluster
    # phate_operator = phate.PHATE(n_jobs=-1, knn=5, decay=30, n_components=5)
    # z = list(phate_operator.fit_transform(specs_flattened))
    # indv_dfs[indv]["phate_cluster"] = z

    # pca = PCA(n_components=2)
    # indv_dfs[indv]["pca_viz"] = list(pca.fit_transform(specs_flattened))

    pca2 = PCA(n_components=10)
    indv_dfs[indv]["pca_cluster"] = list(pca2.fit_transform(specs_flattened))

    # # umap_cluster 
    # fit = umap.UMAP(n_neighbors=20, min_dist=0.05, n_components=10, verbose=True)
    # z = list(fit.fit_transform(specs_flattened))
    # indv_dfs[indv]["umap_cluster"] = z

    # Set min distance (for visualisation only) depending on # syllables
    min_dist = (
        ((len(specs_flattened) - min(syllable_n)) * (0.3 - 0.07))
        / (max(syllable_n) - min(syllable_n))
    ) + 0.07

    # umap_viz
    #n_neighbors=60, min_dist=min_dist, n_components=2, verbose=True
    fit = umap.UMAP(n_components=2, min_dist=min_dist)
    z = list(fit.fit_transform(specs_flattened))
    indv_dfs[indv]["umap_viz"] = z


# %%
# Cluster using HDBSCAN

for indv in tqdm(indv_dfs.keys()):
    z = list(indv_dfs[indv]["pca_cluster"].values)
    min_cluster_size = int(len(z) * 0.02) # smallest cluster size allowed
    if min_cluster_size < 2:
        min_cluster_size = 2
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,  
        min_samples=10,  # larger values = more conservative clustering
        cluster_selection_method="eom",
    )
    clusterer.fit(z)
    indv_dfs[indv]["hdbscan_labels"] = clusterer.labels_

    # # Plot
    # n_colours = len(indv_dfs[indv]["hdbscan_labels"].unique())
    # color_palette = sns.color_palette("deep", n_colours)
    # cluster_colors = [
    #     color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusterer.labels_
    # ]
    # cluster_member_colors = [
    #     sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)
    # ]

    # x = np.array(list(indv_dfs[indv]["umap_viz"].values))[:, 0]
    # y = np.array(list(indv_dfs[indv]["umap_viz"].values))[:, 1]
    # plt.scatter(x, y, s=10, linewidth=0, c=cluster_member_colors, alpha=0.3)
    # plt.show()

    # clusterer.condensed_tree_.plot(
    #     select_clusters=True, selection_palette=sns.color_palette("deep", 14)
    # )

    # plt.show()
    
    # # Plot outliers
    # sns.distplot(clusterer.outlier_scores_[np.isfinite(clusterer.outlier_scores_)], rug=True)
    # plt.show()
    # threshold = pd.Series(clusterer.outlier_scores_).quantile(0.99)
    # outliers = np.where(clusterer.outlier_scores_ > threshold)[0]
    # plt.scatter(x,y, s=10, linewidth=0, c='gray', alpha=0.25)
    # plt.scatter(x[outliers], y[outliers], s=10, linewidth=0, c='red', alpha=0.5)
    # plt.show()

    # Count labels
    print(indv + ":" + str(len(indv_dfs[indv]["hdbscan_labels"].unique())))

# %%

# Save dataframe for each individual
out_dir = DATA_DIR / "indv_dfs" / DATASET_ID
ensure_dir(out_dir)


for indv in tqdm(indv_dfs.keys()):
    indv_dfs[indv].to_pickle(out_dir / (indv + ".pickle"))



# %%
# ### plot each individual's repertoire


for indv in tqdm(indv_dfs.keys()):
    labs = indv_dfs[indv]["hdbscan_labels"].values
    proj = np.array(list(indv_dfs[indv]["umap_viz"].values))
    specs = indv_dfs[indv].spectrogram.values

    scatter_spec(
        proj,
        specs=indv_dfs[indv].spectrogram.values,
        column_size=8,
        # x_range = [-5.5,7],
        # y_range = [-10,10],
        pal_color="hls",
        color_points=False,
        enlarge_points=20,
        range_pad=0.1,
        figsize=(10, 10),
        scatter_kwargs={
            "labels": labs,
            "alpha": 0.60,
            "s": 5,
            "color_palette": pal,
            "show_legend": False,
        },
        matshow_kwargs={"cmap": plt.cm.Greys},
        line_kwargs={"lw": 1, "ls": "solid", "alpha": 0.11},
        draw_lines=True,
        border_line_width=0,
        facecolour=facecolour,
    )

    fig_out = (
        FIGURE_DIR
        / YEAR
        / "ind_repertoires"
        / (
            "{}_repertoire_".format(indv)
            + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            + ".png"
        )
    )
    ensure_dir(fig_out)
    plt.savefig(
        fig_out, dpi=300, bbox_inches="tight", pad_inches=0.3, transparent=False,
    )
    plt.show()
    plt.close()


# %%

# Plot: scatter, transitions, examples, per nestbox
# Saves figures to FIGURE_DIR / year / "ind_repertoires" / (indv + ".png")


quad_plot_syllables(indv_dfs, YEAR, "umap_viz", palette=pal, facecolour=facecolour)

# %%
import string
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.avgn.utils.paths import ensure_dir, most_recent_subdirectory
from src.avgn.visualization.network_graph import plot_network_graph
from src.avgn.visualization.projections import (
    draw_projection_transitions,
    plot_label_cluster_transitions,
    scatter_projections,
    scatter_spec,
)
from src.avgn.visualization.spectrogram import draw_spec_set, plot_example_specs
from src.greti.read.paths import DATA_DIR, FIGURE_DIR
from tqdm.autonotebook import tqdm
from matplotlib.lines import Line2D

#%%
viz_proj="umap_viz"
pal="tab20"
#indv = 'MP42'

for indv in tqdm(indv_dfs.keys()):

    f = plt.figure(figsize=(10, 10))
    gs = f.add_gridspec(ncols=3, nrows=2, width_ratios=[1, 1, 1], height_ratios = [0.5, 1], hspace=0.1, wspace=0.2)

    axes = [f.add_subplot(gs[i]) for i in range(3)]
    axes = axes + [f.add_subplot(gs[1, :])]

    f.suptitle("Syllable clusters and transitions for {}".format(indv), fontsize=16,)

    hdbscan_labs = indv_dfs[indv]["hdbscan_labels"]
    labs = hdbscan_labs.values
    unique_labs = hdbscan_labs.unique()
    nlabs = len(unique_labs)


    proj = np.array(list(indv_dfs[indv][viz_proj].values))[:, 0:2]
    sequence_ids = np.array(indv_dfs[indv]["syllables_sequence_id"])
    specs = np.invert(indv_dfs[indv].spectrogram.values)
    specs = np.where(specs == 255, 242, specs)  # grey

    palette = sns.color_palette(pal, n_colors=len(np.unique(labs)))

    # Projection scatterplot, labeled by cluster
    scatter_projections(
        projection=proj,
        labels=labs,
        color_palette=palette,
        alpha=0.60,
        s=2,
        facecolour=facecolour,
        show_legend=False,
        range_pad=0.1,
        ax=axes[0],
    )


    # Draw lines between consecutive syllables
    draw_projection_transitions(
        projections=proj,
        sequence_ids=indv_dfs[indv]["syllables_sequence_id"],
        sequence_pos=indv_dfs[indv]["syllables_sequence_pos"],
        cmap=plt.get_cmap("ocean"),
        facecolour=facecolour,
        linewidth=0.8,
        range_pad=0.05,
        alpha=0.05,
        ax=axes[1],
    )

    # Plot inferred directed network
    plot_network_graph(
        labs,
        proj,
        sequence_ids,
        color_palette=palette,
        min_cluster_samples=0,
        min_connections=0,
        facecolour=facecolour,
        ax=axes[2],
        edge_width=1,
        point_size=60
    )

    # Plot examples of each cluster
    plot_example_specs(
        specs=specs,
        labels=labs,
        clusters_to_viz=unique_labs[unique_labs >= 0],  # do not show 'noisy' points
        custom_pal=palette,
        cmap=plt.cm.bone,
        nex=10,
        line_width=5,
        ax=axes[3],
    )

    # color labels

    lab_dict = {lab: palette[i] for i, lab in enumerate(np.unique(labs))}

    lab_dict[-1] = (
        0.83137254902,
        0.83137254902,
        0.83137254902
        )  # colour noisy data grey

    legend_elements = [
        Line2D([0], [0], marker="o", color=value, label=key)
        for key, value in lab_dict.items()
    ]

    axes[3].legend(handles=legend_elements, bbox_to_anchor=(1.04, 0.65))

    # labels = string.ascii_uppercase[0 : len(axes)]

    # for ax, labels in zip(axes, labels):
    #     bbox = ax.get_tightbbox(f.canvas.get_renderer())
    #     f.text(
    #         0.03,
    #         0.97,
    #         labels,
    #         fontsize=25,
    #         fontweight="bold",
    #         va="top",
    #         ha="left",
    #         transform=ax.transAxes,
    #     )

    plt.show()





# def make_interactive_plot(new_df, all_coords_nas, colour):

#     # Start plotting interactive fig
#     fig = make_subplots(rows=1, cols=2)

#     # Add transition lines
#     fig.add_trace(go.Scatter(x=all_coords_nas.x, y=all_coords_nas.y,
#                                 mode='lines', name = 'T', line=dict(color="rgba(255,255,255,0.7)", width=0.05)), row=1, col=1)
#     fig.update_traces(connectgaps=False, marker=dict(size=5))

#     # Add each label to scatterplot in a loop
#     label_list = new_df.labs.unique().tolist()
#     label_list.sort(key=int)
#     for label in label_list:
#         fig.add_trace(go.Scatter(
#             x = new_df.loc[new_df.labs == label].x,
#             y = new_df.loc[new_df.labs == label].y,
#             mode = 'markers',
#             name = label,
#             marker=dict(size=5, color=colour[label])
#         ),row=1, col=1)

#     # Aesthetics
#     fig.update_xaxes(showgrid=False, zeroline=False, visible=False, showticklabels=False)
#     fig.update_yaxes(showgrid=False, zeroline=False, visible=False, showticklabels=False)
#     fig.update_layout(
#         autosize=False,
#         width=1300,
#         height=700,
#         legend=dict(
#         orientation="v"),
#         legend_title_text='Label',
#         font_color="#cfcfcf",
#         title_font_color="#cfcfcf",
#         legend_title_font_color="#cfcfcf",
#         title={
#         'text': f"{indv}",
#         'y':0.95,
#         'x':0.5,
#         'xanchor': 'center',
#         'yanchor': 'top'},
#         xaxis_range=(new_df.x.min() - 1, new_df.x.max() + 1),
#         yaxis_range=(new_df.y.min() - 1, new_df.y.max() + 1),
#         plot_bgcolor='black',
#         paper_bgcolor = 'black'

#     )

#     # convert to figurewidget (listen for selections)
#     fig  = go.FigureWidget(fig)

#     return fig

#%%
from src.greti.viz.interactive import assign_new_label, interactive_plot

#%%

#*KEEP THIS ON TOP - IMPORTANT
# Add new label column every bird's dataframe
for indv in indv_dfs.keys():
    if 'hdbscan_labels_fixed' not in indv_dfs[indv]:
            indv_dfs[indv]['hdbscan_labels_fixed'] = indv_dfs[indv]['hdbscan_labels']
    else:
        raise Exception('Column already exists')


#%%

# indv = 'O36'
pal_name = "tab20"
viz_proj="umap_viz"



# %%

#! CAREFUL
i = -1
already_checked = []
#! CAREFUL


# %%
i += 1

if i >= len(indvs):
    raise Exception("End of list")
else:
    indv = indvs[i]
    already_checked.append(indv)

if 'fig' in locals() or 'fig' in globals():
    del(fig)

fig, colour, new_df = interactive_plot(indv_dfs, indv, pal_name, viz_proj, original_labels="hdbscan_labels");
# change to original_labels="hdbscan_labels_fixed" if you don't want to reset
fig


# %%

progress_out = DATA_DIR / 'resources' / DATASET_ID / 'label_fix_progress' / f'progress_{str(datetime.now().strftime("%Y-%m-%d_%H-%M"))}.txt'
ensure_dir(progress_out)

with open(progress_out, "w") as output:
    output.write(str(already_checked))

# %%
import time

for indv in indvs:

    # prepare data (scatterplot)
    new_df, colour, palette = prepare_interactive_data(indv, pal_name)
    # get data ready (transition lines)
    all_coords_nas = get_transition_df(indv, viz_proj)
    fig = make_interactive_plot(new_df, all_coords_nas, colour)
    fig
    done = input("are you done, you fuckwit? y/n")

    while done == 'n':

        new_label = input("1: Select notes. 2: Enter new label and press enter")

        while [f.selectedpoints for f in fig.data] is None:
            time.sleep(2)
        else:
            assign_new_label(new_label)
    else:
        break


# %%


import base64
from io import BytesIO

prefix = f'data:image/png;base64,'

prefix + base64.b64encode(example_image).decode('utf-8')

example_image




# %%
import re
from src.avgn.visualization.network_graph import plot_network_graph

def plot_directed_graph():

    # Prepare necessary data
    projections = np.array(list(indv_dfs[indv][viz_proj].values))[:, 0:2]

    hdbscan_labs = indv_dfs[indv]["hdbscan_labels_fixed"]
    labs = hdbscan_labs.values
    unique_labs = hdbscan_labs.unique()
    nlabs = len(unique_labs)

    sequence_ids = np.array(indv_dfs[indv]["syllables_sequence_id"])

    # Convert dictionary to palette, scaling colour values from 0 to 1
    net_palette = [tuple([int(s) / 255 for s in re.findall(r'\b\d+\b', col)]) for col in colour.values()]

    # Make plot
    fig, ax = fig, ax = plt.subplots(figsize=(10, 10))

    ax = plot_network_graph(
        labs,
        projections,
        sequence_ids,
        color_palette=net_palette,
        min_cluster_samples=0,
        min_connections=0,
        facecolour='black',
        edge_width=0.1,
        edge_colour='white',
        point_size=300,
        arrowsize = 40,
        ax=ax,
    )

    plt.subplots_adjust(wspace=0, hspace=0.1)

    fig.set_facecolor("black")
    ax.set_facecolor("black")

    figure = fig2img(fig)
    plt.close()

    figure

#%%
from src.avgn.visualization.barcodes import indv_barcode
from src.avgn.visualization.network_graph import build_transition_matrix, compute_graph
import networkx as nx
import matplotlib as mpl
from PIL import Image
import random
import librosa
import matplotlib.patches as mpatches



from src.vocalseg.utils import (
    butter_bandpass_filter,
    int16tofloat32,
    plot_spec,
    spectrogram,
)

#TODO: get a few sample songs with labels to aid classification

# first get index of wavs that contain desired labels 

label = 3
original_labels="hdbscan_labels_fixed"

len_label = len(indv_dfs[indv].loc[indv_dfs[indv]['hdbscan_labels_fixed'] == label].key)

#for i in random.sample(range(len_label), 3)

# load the wav
key = indv_dfs[indv].loc[indv_dfs[indv]['hdbscan_labels_fixed'] == label].key.iloc[i]
wav_dir = most_recent_subdirectory(DATA_DIR / "processed" / DATASET_ID.replace('_segmented', ''), only_dirs=True) / 'WAV' / key
wav, rate = librosa.core.load(wav_dir, sr=None)

# Bandpass
data = butter_bandpass_filter(wav, 1200, 10000, rate)

# Create the spectrogram
spec = spectrogram(
    data,
    rate,
    n_fft=1024,
    hop_length_ms=3,
    win_length_ms=15,
    ref_level_db=30,
    min_level_db=-60,
)

# Label colours

lab_colours = {int(lab) : tuple([int(s) / 255 for s in re.findall(r"\b\d+\b", col)]) for lab, col in colour.items()} 

# from src.greti.audio.filter import dereverberate

# spec = dereverberate(spec, echo_range=100, echo_reduction=3, hop_length_ms=3)
# spec[spec < 0] = 0

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

#! now invert spectrogram etc etc

for ix, row in indv_dfs[indv][indv_dfs[indv].key == key].iterrows():
    if row[original_labels] > -1:  # don't plot noise

        color = lab_colours[row[original_labels]]
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


plt.show()
