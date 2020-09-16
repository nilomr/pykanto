# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# from IPython import get_ipython

# get_ipython().run_line_magic("env", "CUDA_DEVICE_ORDER=PCI_BUS_ID")
# get_ipython().run_line_magic("env", "CUDA_VISIBLE_DEVICES=2")
# get_ipython().run_line_magic("load_ext", "autoreload")
# get_ipython().run_line_magic("autoreload", "2")
# get_ipython().run_line_magic("matplotlib", "inline")

import pickle

import hdbscan
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phate
import scprep
import seaborn as sns
import umap
from joblib import Parallel, delayed
from matplotlib import gridspec
from scipy.spatial import distance
from tqdm.autonotebook import tqdm

from src.avgn.signalprocessing.create_spectrogram_dataset import flatten_spectrograms
from src.avgn.utils.general import save_fig
from src.avgn.utils.paths import ensure_dir, most_recent_subdirectory
from src.avgn.visualization.network_graph import plot_network_graph
from src.avgn.visualization.projections import (
    draw_projection_transitions,
    plot_label_cluster_transitions,
    scatter_projections,
    scatter_spec,
)
from src.avgn.visualization.quickplots import draw_projection_plots, quad_plot_syllables
from src.greti.read.paths import DATA_DIR, FIGURE_DIR, RESOURCES_DIR

# from sklearn.cluster import MiniBatchKMeans
# from cuml.manifold.umap import UMAP as cumlUMAP


# %%

# ### get data

DATASET_ID = "GRETI_HQ_2020_segmented"
year = "2020"

save_loc = DATA_DIR / "syllable_dfs" / DATASET_ID / "{}.pickle".format(DATASET_ID)
syllable_df = pd.read_pickle(save_loc)


# %%
# indvs = syllable_df.indv.unique()#[:3] # remove subsetting, this is a precaution

indvs = [
    ind
    for ind in syllable_df.indv.unique()
    if len(syllable_df[syllable_df.indv == ind])
    > 60  # This threshold is based on the need to have minimum clustr sizes of size >1
]

len(indvs)


# %%

# Add nestbox positions to syllable_df

coords_file = RESOURCES_DIR / "nestboxes" / "nestbox_coords.csv"
tmpl = pd.read_csv(coords_file)

nestboxes = tmpl[tmpl["nestbox"].isin(syllable_df.indv.unique())]
nestboxes["east_north"] = nestboxes[["x", "y"]].apply(tuple, axis=1)


X = [(447000, 208000)]

for i in nestboxes.index:
    nestboxes.at[i, "dist_m"] = distance.cdist(
        X, [nestboxes.at[i, "east_north"]], "euclidean"
    )[0, 0]

nestboxes.filter(["nestbox", "east_north", "section", "dist_m"])

#  Add to syllable_df
syllable_df = pd.merge(
    syllable_df, nestboxes, how="inner", left_on="indv", right_on="nestbox"
)

# To convert BNG to WGS84:

# import pyproj

# bng=pyproj.Proj(init='epsg:27700')
# wgs84 = pyproj.Proj(init='epsg:4326')

# def convertCoords(row):
#     x2,y2 = pyproj.transform(bng,wgs84,row['x'],row['y'])
#     return pd.Series([x2, y2])

# nestboxes[['long','lat']] = nestboxes[['x', 'y']].apply(convertCoords,axis=1)

# %%
# Plot settings

facecolour = "#f2f1f0"
pal = "Set2"


# %% [markdown]
# ###

indv_dfs = {}

for indvi, indv in enumerate(tqdm(indvs)):
    # if indv != 'Bird5': continue
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

    # PHATE
    phate_op = phate.PHATE()
    phate_operator = phate.PHATE(n_jobs=-1, knn=15, alpha_decay=0)
    z = list(phate_operator.fit_transform(specs_flattened))
    indv_dfs[indv]["phate"] = z

    # umap_cluster
    fit = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=7)
    z = list(fit.fit_transform(specs_flattened))
    indv_dfs[indv]["umap_cluster"] = z

    # umap_viz
    fit = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    z = list(fit.fit_transform(specs_flattened))
    indv_dfs[indv]["umap_viz"] = z


# %%
# Cluster using HDBSCAN

for indv in tqdm(indv_dfs.keys()):
    z = list(indv_dfs[indv]["umap_cluster"].values)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(
            len(z) * 0.03
        ),  # the smallest size we would expect a cluster to be
        min_samples=5,  # larger values = more conservative clustering
    )
    clusterer.fit(z)
    indv_dfs[indv]["hdbscan_labels"] = clusterer.labels_


for indv in tqdm(indv_dfs.keys()):
    print(indv + ":" + str(len(indv_dfs[indv]["hdbscan_labels"].unique())))

# %%

# Save dataframe with every individual

out_dir = DATA_DIR / "embeddings" / DATASET_ID
ensure_dir(out_dir)

pickle.dump(indv_dfs, open(out_dir / ("individual_embeddings" + ".pickle"), "wb"))


# %%

# UMAP and PHATE embeddings to visualise full dataset

specs = list(syllable_df.spectrogram.values)
specs = [i / np.max(i) for i in specs]
specs_flattened = flatten_spectrograms(specs)

# UMAP embedding for all birds in dataset
fit = umap.UMAP(n_neighbors=30, min_dist=0.2, n_components=2)
umap_proj = list(fit.fit_transform(specs_flattened))

# PHATE
phate_op = phate.PHATE()
phate_operator = phate.PHATE(n_jobs=-1, knn=15, alpha_decay=1)
phate_proj = list(phate_operator.fit_transform(specs_flattened))


# Save embeddings
out_dir = DATA_DIR / "embeddings" / DATASET_ID
ensure_dir(out_dir)

syllable_df["umap"] = list(umap_proj)
syllable_df["phate"] = list(phate_proj)

syllable_df.to_pickle(out_dir / ("full_dataset" + ".pickle"))

# %%


# %%
# Load datasets if they already exist

DATASET_ID = "GRETI_HQ_2020_segmented"
year = "2020"

syll_loc = DATA_DIR / "embeddings" / DATASET_ID / "full_dataset.pickle"
syllable_df = pd.read_pickle(syll_loc)

ind_loc = DATA_DIR / "embeddings" / DATASET_ID / "individual_embeddings.pickle"
indv_dfs = pd.read_pickle(ind_loc)


# %%
# Plot projections of all individuals, colour=distance

umap_proj = list(syllable_df["umap"])
phate_proj = list(syllable_df["phate"])

labs = syllable_df.dist_m.values

cmap = sns.cubehelix_palette(
    n_colors=len(np.unique(labs)),
    start=0,
    rot=0.8,  # if 0 no hue change
    gamma=0.7,
    hue=0.8,
    light=0.95,
    dark=0.15,
    reverse=False,
    as_cmap=True,
)

for proj in [phate_proj, umap_proj]:

    if proj is phate_proj:
        name = "PHATE"
    elif proj is umap_proj:
        name = "UMAP"

    scatter_projections(
        projection=proj,
        labels=labs,
        alpha=1,
        s=1,
        color_palette="cubehelix",
        cmap=cmap,
        show_legend=False,
        facecolour="k",
        colourbar=True,
        figsize=(10, 10),
    )

    from datetime import datetime

    fig_out = (
        FIGURE_DIR
        / year
        / "population"
        / (
            "{}_scatter".format(name)
            + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            + ".svg"
        )
    )
    ensure_dir(fig_out)
    plt.savefig(
        fig_out, dpi=300, bbox_inches="tight", pad_inches=0.3, transparent=False,
    )
    # plt.show()
    plt.close()


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
            "s": 7,
            "color_palette": pal,
            "show_legend": False,
        },
        matshow_kwargs={"cmap": plt.cm.Greys},
        line_kwargs={"lw": 1, "ls": "solid", "alpha": 0.18},
        draw_lines=True,
        border_line_width=0,
        facecolour=facecolour,
    )

    fig_out = (
        FIGURE_DIR
        / year
        / "ind_repertoires"
        / (
            "{}_repertoire_".format(indv)
            + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
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

quad_plot_syllables(indv_dfs, year)
