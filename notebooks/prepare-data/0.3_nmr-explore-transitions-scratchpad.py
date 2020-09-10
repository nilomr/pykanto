# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

get_ipython().run_line_magic("env", "CUDA_DEVICE_ORDER=PCI_BUS_ID")
get_ipython().run_line_magic("env", "CUDA_VISIBLE_DEVICES=2")
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic("matplotlib", "inline")


from tqdm.autonotebook import tqdm
from joblib import Parallel, delayed
import umap
import hdbscan
import pandas as pd
from src.avgn.utils.paths import DATA_DIR, most_recent_subdirectory, ensure_dir
from src.avgn.signalprocessing.create_spectrogram_dataset import flatten_spectrograms
from src.avgn.visualization.spectrogram import draw_spec_set
from src.avgn.visualization.quickplots import draw_projection_plots

from src.greti.read.paths import DATA_DIR, FIGURE_DIR
from src.avgn.visualization.projections import (
    scatter_projections,
    draw_projection_transitions,
    scatter_spec,
    plot_label_cluster_transitions,
)
from src.avgn.visualization.network_graph import plot_network_graph

from src.avgn.utils.general import save_fig

# from sklearn.cluster import MiniBatchKMeans
# from cuml.manifold.umap import UMAP as cumlUMAP


# %%

# ### get data

DATASET_ID = "GRETI_HQ_2020_segmented"

save_loc = DATA_DIR / "syllable_dfs" / DATASET_ID / "{}.pickle".format(DATASET_ID)
syllable_df = pd.read_pickle(save_loc)


# %%
def norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


# %% [markdown]
# ### cluster


# %%
indvs = syllable_df.indv.unique()[:3]

# %% [markdown]
# ###

indv_dfs = {}

for indvi, indv in enumerate(tqdm(indvs)):
    # if indv != 'Bird5': continue
    indv_dfs[indv] = syllable_df[syllable_df.indv == indv]
    indv_dfs[indv] = indv_dfs[indv].sort_values(by=["key", "start_time"])
    print(indv, len(indv_dfs[indv]))
    specs = [norm(i) for i in indv_dfs[indv].spectrogram.values]

    # sequencing
    indv_dfs[indv]["syllables_sequence_id"] = None
    indv_dfs[indv]["syllables_sequence_pos"] = None
    for ki, key in enumerate(indv_dfs[indv].key.unique()):
        indv_dfs[indv].loc[indv_dfs[indv].key == key, "syllables_sequence_id"] = ki
        indv_dfs[indv].loc[
            indv_dfs[indv].key == key, "syllables_sequence_pos"
        ] = np.arange(np.sum(indv_dfs[indv].key == key))

    # umap
    specs_flattened = flatten_spectrograms(specs)
    fit = umap.UMAP(n_neighbors=40, min_dist=0.1, n_components=10)
    z = list(fit.fit_transform(specs_flattened))
    indv_dfs[indv]["umap_cluster"] = z

    # umap_viz
    specs_flattened = flatten_spectrograms(specs)
    fit = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=2)
    z = list(fit.fit_transform(specs_flattened))
    indv_dfs[indv]["umap_viz"] = z


# %%
# Cluster

for indv in tqdm(indv_dfs.keys()):
    # HDBSCAN UMAP
    z = list(indv_dfs[indv]["umap_cluster"].values)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(
            len(z) * 0.04
        ),  # the smallest size we would expect a cluster to be
        min_samples=2,  # larger values = more conservative clustering
    )
    clusterer.fit(z)
    indv_dfs[indv]["hdbscan_labels"] = clusterer.labels_

    # HDBSCAN
    specs = [norm(i) for i in indv_dfs[indv].spectrogram.values]
    specs_flattened = flatten_spectrograms(specs)

# %%

for indv in tqdm(indv_dfs.keys()):
    print(indv + ":" + str(len(indv_dfs[indv]["hdbscan_labels"].unique())))

# %% [markdown]
# ### plot

# %%

facecolour = "#f2f1f0"
pal = "Set2"

from matplotlib import gridspec


# %%

for indv in tqdm(indv_dfs.keys()):
    labs = indv_dfs[indv]["hdbscan_labels"].values
    proj = np.array(list(indv_dfs[indv]["umap_viz"].values))

    scatter_spec(
        proj,
        specs=indv_dfs[indv].spectrogram.values,
        column_size=8,
        # x_range = [-5.5,7],
        # y_range = [-10,10],
        pal_color="hls",
        color_points=False,
        enlarge_points=30,
        range_pad=0.1,
        figsize=(10, 10),
        scatter_kwargs={
            "labels": labs,
            "alpha": 0.60,
            "s": 7,
            "color_palette": pal,
            "show_legend": False,
        },
        matshow_kwargs={"cmap": plt.cm.Greys},
        line_kwargs={"lw": 1, "ls": "solid", "alpha": 0.25},
        draw_lines=True,
        border_line_width=0,
        facecolour=facecolour,
    )

# %%

for indv in tqdm(indv_dfs.keys()):

    f = plt.figure(figsize=(30, 10))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    ax = f.add_subplot(gs[0])
    ax1 = f.add_subplot(gs[1])
    ax2 = f.add_subplot(gs[2])

    labs = indv_dfs[indv]["hdbscan_labels"].values
    proj = np.array(list(indv_dfs[indv]["umap_viz"].values))
    sequence_ids = np.array(indv_dfs[indv]["syllables_sequence_id"])

    # fig.suptitle("UMAP projection and trajectories for {}".format(indv), fontsize=30)

    scatter_projections(
        projection=proj,
        labels=labs,
        color_palette=pal,
        alpha=0.60,
        s=7,
        facecolour=facecolour,
        show_legend=False,
        range_pad=0.1,
        ax=ax,
    )
    ax.set_facecolor(facecolour)

    draw_projection_transitions(
        projections=proj,
        sequence_ids=indv_dfs[indv]["syllables_sequence_id"],
        sequence_pos=indv_dfs[indv]["syllables_sequence_pos"],
        cmap=plt.get_cmap("ocean"),
        facecolour=facecolour,
        range_pad=0.15,
        alpha=0.02,
        ax=ax1,
    )

    plot_network_graph(
        labs, proj, sequence_ids, color_palette=pal, min_cluster_samples=40, ax=ax2
    )

    # ax3.plot(net)

    # plt.gca().set_axis_off()
    # plt.subplots_adjust(
    #     top=1, bottom=0, right=1, left=0, hspace=0, wspace=0,
    # )
    # plt.margins(3, 0)
    # fig.tight_layout(pad=1)
    # fig.subplots_adjust(top=0.90)

    plt.show()


# %%

plot_example_specs(
    specs=np.array(list(syllable_df.syllables_spec.values)),
    labels=np.array(list(syllable_df.hdbscan_labels.values)),
    clusters_to_viz=[17, 18, 9, 20],
    custom_pal=custom_pal,
    ax=ax,
    nex=4,
    line_width=2,
)


# %%


zoom = 6
ncols = 3
fig, axs = plt.subplots(ncols=ncols, figsize=(zoom * ncols, zoom))


proj = np.array(list(indv_dfs[indv]["umap_viz"].values))
labs = indv_dfs[indv]["hdbscan_labels"].values
pal = "Set2"
facecolour = "#f2f1f0"

sequence_ids = np.array(indv_dfs[indv]["syllables_sequence_id"])
plot_network_graph(
    labs, proj, sequence_ids, color_palette=pal, min_cluster_samples=50,
)

ax = axs[0]
# plot scatter
scatter_projections(
    projection=proj, labels=labs, color_palette=pal, ax=ax,
)
ax.axis("off")


# transition plot
ax = axs[1]

draw_projection_transitions(
    projections=proj,
    sequence_ids=indv_dfs[indv]["syllables_sequence_id"],
    sequence_pos=indv_dfs[indv]["syllables_sequence_pos"],
    cmap=plt.get_cmap("ocean"),
    facecolour=facecolour,
    alpha=0.03,
    ax=axs[0],
)

ax.axis("off")

# # transitions
# ax = axs[2]
# plot_label_cluster_transitions(
#     syllable_df,
#     '6',
#     superlabel="hdbscan_labels",
#     sublabel="hdbscan_labels",
#     projection_column="umap_viz",
#     color_palette=pal,
#     scatter_alpha=0.05,
#     ax=ax,
# )

# ax.axis("off")


# network graph HDBSCAN
ax = axs[2]
sequence_ids = np.array(indv_dfs[indv]["syllables_sequence_id"])
plot_network_graph(
    labs, proj, sequence_ids, color_palette=pal, min_cluster_samples=50,
)

ax.axis("off")


for ax, lab in zip(axs, ["A", "B", "C"]):
    ax.text(
        -0.1,
        1,
        lab,
        transform=ax.transAxes,
        size=24,
        **{"ha": "center", "va": "center", "family": "sans-serif", "fontweight": "bold"}
    )


# save_fig(FIGURE_DIR / "sober_bf_transitions", save_pdf=False, save_svg=False)


# %%
