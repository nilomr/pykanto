# %%
# from IPython import get_ipython

# get_ipython().run_line_magic("env", "CUDA_DEVICE_ORDER=PCI_BUS_ID")
# get_ipython().run_line_magic("env", "CUDA_VISIBLE_DEVICES=2")
# get_ipython().run_line_magic("load_ext", "autoreload")
# get_ipython().run_line_magic("autoreload", "2")
# get_ipython().run_line_magic("matplotlib", "inline")

import pickle
from datetime import datetime

import hdbscan
import matplotlib.pyplot as plt
import numpy as np

# %%
import pandas as pd
import phate
import seaborn as sns
import umap
from joblib import Parallel, delayed
from scipy.spatial import distance
from tqdm.autonotebook import tqdm

from src.avgn.signalprocessing.create_spectrogram_dataset import (
    flatten_spectrograms,
    log_resize_spec,
)
from src.avgn.utils.general import save_fig
from src.avgn.utils.paths import ensure_dir, most_recent_subdirectory
from src.avgn.visualization.network_graph import plot_network_graph
from src.avgn.visualization.projections import scatter_projections, scatter_spec
from src.avgn.visualization.quickplots import draw_projection_plots, quad_plot_syllables
from src.avgn.visualization.spectrogram import draw_spec_set
from src.greti.read.paths import DATA_DIR, FIGURE_DIR, RESOURCES_DIR

# from sklearn.cluster import MiniBatchKMeans
# from cuml.manifold.umap import UMAP as cumlUMAP

# import importlib
# importlib.reload(src)

# %%

# ### get data

DATASET_ID = "GRETI_HQ_2020_segmented"
YEAR = "2020"

save_loc = DATA_DIR / "syllable_dfs" / DATASET_ID / "{}.pickle".format(DATASET_ID)
syllable_df = pd.read_pickle(save_loc)


# %%
# Rescale spectrograms

log_scaling_factor = 10

with Parallel(n_jobs=-1, verbose=2) as parallel:
    syllables_spec = parallel(
        delayed(log_resize_spec)(spec, scaling_factor=log_scaling_factor)
        for spec in tqdm(
            syllable_df["spectrogram"], desc="scaling spectrograms", leave=False
        )
    )
# %%
syllable_df["spectrogram"] = syllables_spec


# %%
# indvs = syllable_df.indv.unique()#[:3] # remove subsetting, this is a precaution

indvs = [
    ind
    for ind in syllable_df.indv.unique()[15:20]  #!!!! Remove subsetting !!!!
    if len(syllable_df[syllable_df.indv == ind])
    > 60  # This threshold is based on the need to have minimum clustr sizes of size >1
]

len(indvs)

# %%
# Number of syllables per nest - will need this later
syllable_n = pd.Series(
    [len(syllable_df[syllable_df.indv == ind]) for ind in syllable_df.indv.unique()]
)
# %%

# Get individuals
indvs = [
    ind
    for ind in syllable_df.indv.unique()  #! Remove subsetting
    if len(syllable_df[syllable_df.indv == ind])
    > 60  # This threshold is based on the need to have minimum clustr sizes of size >1
]

# %% [markdown]
# ###

indv_dfs = {}

for indvi, indv in enumerate(tqdm(indvs)):

    indv_dfs[indv] = syllable_df[syllable_df.indv == indv]
    indv_dfs[indv] = indv_dfs[indv].sort_values(by=["key", "start_time"])
    print(indv, len(indv_dfs[indv]))
    specs = [i for i in indv_dfs[indv].spectrogram.values]

    # sequencing
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

    # umap_cluster
    fit = umap.UMAP(
        n_neighbors=20, min_dist=0.1, n_components=20, verbose=True, init="random"
    )
    z = list(fit.fit_transform(specs_flattened))
    indv_dfs[indv]["umap_cluster"] = z

    # Set min distance (for visualisation only) depending on # syllables
    min_dist = (
        ((len(z) - min(syllable_n)) * (0.4 - 0.1)) / (max(syllable_n) - min(syllable_n))
    ) + 0.1

    # umap_viz
    fit = umap.UMAP(n_neighbors=30, min_dist=min_dist, n_components=2, verbose=True)
    z = list(fit.fit_transform(specs_flattened))
    indv_dfs[indv]["umap_viz"] = z


# %%
# Cluster using HDBSCAN

for indv in tqdm(indv_dfs.keys()):
    z = list(indv_dfs[indv]["umap_cluster"].values)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(
            len(z) * 0.035
        ),  # the smallest size we would expect a cluster to be
        min_samples=4,  # larger values = more conservative clustering
        cluster_selection_method="eom",
    )
    clusterer.fit(z)
    indv_dfs[indv]["hdbscan_labels"] = clusterer.labels_


# for indv in tqdm(indv_dfs.keys()):
#     print(indv + ":" + str(len(indv_dfs[indv]["hdbscan_labels"].unique())))

# %%

# Save dataframe for each individual
out_dir = DATA_DIR / "indv_dfs" / DATASET_ID
ensure_dir(out_dir)


for indv in tqdm(indv_dfs.keys()):
    indv_dfs[indv].to_pickle(out_dir / (indv + ".pickle"))


# %%
# Plot settings

facecolour = "#f2f1f0"
pal = "Set2"

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
            + str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
            + ".png"
        )
    )
    ensure_dir(fig_out)
    plt.savefig(
        fig_out, dpi=300, bbox_inches="tight", pad_inches=0.3, transparent=False,
    )
    # plt.show()
    plt.close()


# %%

# Plot: scatter, transitions, examples, per nestbox
# Saves figures to FIGURE_DIR / year / "ind_repertoires" / (indv + ".png")

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

spectral_mod = sns.set_palette(sns.color_palette(colours))
quad_plot_syllables(
    indv_dfs, YEAR, "umap_viz", palette=spectral_mod, facecolour=facecolour
)

# %%
