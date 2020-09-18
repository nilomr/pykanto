# %%
from datetime import datetime

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phate
import seaborn as sns

# import umap
from joblib import Parallel, delayed
from scipy.spatial import distance
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

# %%

# UMAP and PHATE embeddings to visualise full dataset

specs = list(syllable_df.spectrogram.values)
specs = [i / np.max(i) for i in specs]
specs_flattened = flatten_spectrograms(specs)

# # UMAP embedding for all birds in dataset
# fit = umap.UMAP(
#     n_neighbors=300, min_dist=0.3, n_components=2, verbose=True, init="random"
# )
# umap_proj = list(fit.fit_transform(specs_flattened))

# PHATE
phate_op = phate.PHATE()
phate_operator = phate.PHATE(n_jobs=-1, knn=20)
phate_proj = list(phate_operator.fit_transform(specs_flattened))

# %%

# Save embeddings
out_dir = DATA_DIR / "embeddings" / DATASET_ID
ensure_dir(out_dir)

# syllable_df["umap"] = list(umap_proj)
syllable_df["phate"] = list(phate_proj)

syllable_df.to_pickle(out_dir / ("full_dataset_phate" + ".pickle"))


# %%
# Load datasets if they already exist

# DATASET_ID = "GRETI_HQ_2020_segmented"
# YEAR = "2020"

# syll_loc = DATA_DIR / "embeddings" / DATASET_ID / "full_dataset.pickle"
# syllable_df = pd.read_pickle(syll_loc)

# ind_loc = DATA_DIR / "embeddings" / DATASET_ID / "individual_embeddings.pickle"
# indv_dfs = pd.read_pickle(ind_loc)


# %%
# Plot projections of all individuals, colour=distance

# umap_proj = list(syllable_df["umap"])
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

# for proj in [phate_proj, umap_proj]:

#     if proj is phate_proj:
#         name = "PHATE"
#     elif proj is umap_proj:
#         name = "UMAP"

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

fig_out = (
    FIGURE_DIR
    / YEAR
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


print("Done")
