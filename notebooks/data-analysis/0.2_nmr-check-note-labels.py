#%%

from warnings import WarningMessage

from matplotlib.pyplot import figimage
from src.greti.viz.interactive import assign_new_label, interactive_plot, check_new_bird
from src.greti.read.paths import DATA_DIR, FIGURE_DIR, RESOURCES_DIR
import pandas as pd
from src.avgn.utils.paths import ensure_dir

%load_ext autoreload
%autoreload 2


# %%
# Import data

DATASET_ID = "GRETI_HQ_2020_segmented"
YEAR = "2020"

dfs_dir = DATA_DIR / "indv_dfs" / DATASET_ID
indv_dfs = pd.read_pickle(dfs_dir / (f"{DATASET_ID}_labelled.pickle"))

indvs = list(indv_dfs.keys())

#%%
# * Run this first

# Add new label column every bird's dataframe

for indv in indv_dfs.keys():
    if "hdbscan_labels_fixed" not in indv_dfs[indv]:
        indv_dfs[indv]["hdbscan_labels_fixed"] = indv_dfs[indv]["hdbscan_labels"]
    else:
        raise Exception("Column already exists")


#%%
# Colour palette to use
pal_name = "tab20"

# Name of projection to be used for visualisation
# (one of 'umap_viz', 'pca_cluster'.
# This has no consequences for labelling)
viz_proj = "umap_viz"

# Do not run this once you have started checking birds
#!
run = 0

# %%
# Start a counter and a list of birds that have already been checked. Can only be run once per session

if run == 0:
    i = -1
    already_checked = []
    run = 1
else:
    raise InterruptedError("You have already started a session")


# %%

# Execute this cell every time you want to change to the next bird or reset the current bird.

if "indv" not in locals() or "indv" not in globals():
    indv = None

fig, i, colour, new_df, indv = check_new_bird(
    DATASET_ID,
    dfs_dir,
    indv_dfs,
    indv,
    indvs,
    i,
    pal_name,
    viz_proj,
    original_labels="hdbscan_labels",
    reset_bird=False # True if you want to reset the current bird intead of advancing. Don't forget to set to False again.
)

# If you want to relabel a note cluster, use the following function:
# Note: you can hide a label by clicking on its legend. Hidden labels will not be selected.

"""
colour = assign_new_label(
    DATASET_ID,
    indv_dfs,
    indv,
    fig,
    pal_name,
    colour,
    new_df,
    -1,  # New label (can be a new label or an existing label, including noise)
    relabel_noise=False,  # whether noise will be ignored or relabelled
)
"""

fig

#%%
