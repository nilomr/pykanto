# %%
from datetime import datetime

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phate
import seaborn as sns
import umap
from joblib import Parallel, delayed
from scipy.spatial import distance
from sklearn.decomposition import IncrementalPCA
from tqdm.autonotebook import tqdm

from src.avgn.signalprocessing.create_spectrogram_dataset import flatten_spectrograms
from src.avgn.utils.paths import ensure_dir, most_recent_subdirectory
from src.avgn.visualization.network_graph import plot_network_graph
from src.avgn.visualization.projections import scatter_projections
from src.greti.read.paths import DATA_DIR, FIGURE_DIR, RESOURCES_DIR

# from sklearn.cluster import MiniBatchKMeans
# from cuml.manifold.umap import UMAP as cumlUMAP


# %%

# get data

DATASET_ID = "GRETI_HQ_2020_segmented"
YEAR = "2020"

save_loc = DATA_DIR / "syllable_dfs" / DATASET_ID / "{}.pickle".format(DATASET_ID)
syllable_df = pd.read_pickle(save_loc)


# %%

# Add nestbox positions to syllable_df

coords_file = RESOURCES_DIR / "nestboxes" / "nestbox_coords.csv"
tmpl = pd.read_csv(coords_file)

nestboxes = tmpl[tmpl["nestbox"].isin(syllable_df.indv.unique())]
nestboxes["east_north"] = nestboxes[["x", "y"]].apply(tuple, axis=1)

# plt.scatter(nestboxes['x'], nestboxes['y'])

X = [(446000, 208000)]

for i in nestboxes.index:
    nestboxes.at[i, "dist_m"] = distance.cdist(
        X, [nestboxes.at[i, "east_north"]], "euclidean"
    )[0, 0]

nestboxes.filter(["nestbox", "east_north", "section", "dist_m"])

#  Add to syllable_df
syllable_df = pd.merge(
    syllable_df, nestboxes, how="inner", left_on="indv", right_on="nestbox"
)

#%%
# prepare spectrograms

specs = list(syllable_df.spectrogram.values)
specs = flatten_spectrograms([i / np.max(i) for i in specs])

#%%
# PCA, UMAP and PHATE embeddings to visualise full dataset

# %%
# PCA
pca_parameters = {
    "n_components": 2,
    "batch_size": 10,
}
ipca = IncrementalPCA(**pca_parameters)
pca_proj = list(ipca.fit_transform(specs))


# %%

# UMAP
umap_parameters = {
    "n_neighbors": 300,
    "min_dist": 0.3,
    "n_components": 2,
    "verbose": True,
    "init": "random",
    "low_memory": True,
}
fit = umap.UMAP(**umap_parameters)
umap_proj = list(fit.fit_transform(specs))


# %%
# PHATE
phate_parameters = {"n_jobs": -1, "knn": 30, "n_pca": 19, "gamma": 0}
phate_operator = phate.PHATE(**phate_parameters)
phate_proj = list(phate_operator.fit_transform(specs))

# %%

# Save embeddings
out_dir = DATA_DIR / "embeddings" / DATASET_ID
ensure_dir(out_dir)

syllable_df["umap"] = umap_proj
syllable_df["phate"] = phate_proj
syllable_df["pca"] = pca_proj

syllable_df.to_pickle(out_dir / ("full_dataset" + ".pickle"))


# %%
# Load datasets if they already exist

# DATASET_ID = "GRETI_HQ_2020_segmented"
# YEAR = "2020"

# syll_loc = DATA_DIR / "embeddings" / DATASET_ID / "full_dataset.pickle"
# syllable_df = pd.read_pickle(syll_loc)

# ind_loc = DATA_DIR / "embeddings" / DATASET_ID / "individual_embeddings.pickle"
# indv_dfs = pd.read_pickle(ind_loc)


#%%
# labels and palette for plots

labs = syllable_df.dist_m.values

cmap = sns.cubehelix_palette(
    n_colors=len(np.unique(labs)),
    start=0,
    rot=0.9,  # if 0 no hue change
    gamma=1,
    hue=0.9,
    light=0.95,
    dark=0.25,
    reverse=False,
    as_cmap=True,
)


# %%
# Plot projections of all individuals, colour=distance


projections = [phate_proj, pca_proj, umap_proj]


def replace_params(params_dict):
    params = str(params_dict).replace(" ", "").replace("'", "")
    return params


for proj in projections:

    if proj is phate_proj:
        name = "PHATE"
        params = replace_params(phate_parameters)
    elif proj is pca_proj:
        name = "PCA"
        params = replace_params(pca_parameters)
    else:
        name = "UMAP"
        params = replace_params(umap_parameters)

    scatter_projections(
        projection=list(proj),
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

    fig_out = (
        FIGURE_DIR
        / YEAR
        / "population"
        / (
            "{}_scatter_".format(name)
            + str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
            + "_"
            + params
        )
    )
    ensure_dir(FIGURE_DIR / YEAR / "population")
    for extension in [".svg", ".png"]:

        plt.savefig(
            (str(fig_out) + extension),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.3,
            transparent=False,
        )
        # plt.show()

    plt.close()


# %%
# Plot KDE

for proj in projections:

    if proj is phate_proj:
        name = "PHATE"
        params = replace_params(phate_parameters)
    elif proj is pca_proj:
        name = "PCA"
        params = replace_params(pca_parameters)
    else:
        name = "UMAP"
        params = replace_params(umap_parameters)

    fig, ax = plt.subplots(1, figsize=(11, 10))
    sns.kdeplot(
        pca_proj, n_levels=100, shade=True, cmap="inferno", zorder=0, ax=ax,
    )

    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()

    fig_out = (
        FIGURE_DIR
        / YEAR
        / "population"
        / (name + "_KDE_" + str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")) + ".png")
    )
    ensure_dir(fig_out)

    plt.savefig(
        fig_out, dpi=300, bbox_inches="tight", pad_inches=0.3, transparent=False,
    )


print("Done")

# %%
