
import phate
import umap
import hdbscan
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from tqdm.autonotebook import tqdm

from src.avgn.signalprocessing.create_spectrogram_dataset import (
    flatten_spectrograms, log_resize_spec)


def project_individual(syllable_df, indv_dfs, indv, syllable_n):
    """Returns a dataframe with note and sequence information, and PCA and UMAP projections 
    (for clustering and visualisation, respectively).

    Args:
        indv_dfs (dict): Dictionary where to add the bird dataframe
        indv (str): Bird to project
        syllable_n (pandas series): Series with number of syllables per bird, used to calculate parameters for UMAP visualization
    """    

    indv_dfs[indv] = syllable_df[syllable_df.indv == indv]
    indv_dfs[indv] = indv_dfs[indv].sort_values(by=["key", "start_time"])
    print(indv, len(indv_dfs[indv]))
    specs = [i for i in indv_dfs[indv].spectrogram.values]

    specs_scaled = []
    for spec in tqdm(specs, desc="scaling spectrograms", leave=False):
        specs_scaled.append(log_resize_spec(spec, scaling_factor=8))


    # with Parallel(n_jobs=n_jobs, verbose=2) as parallel:
    #     specs = parallel(
    #         delayed(log_resize_spec)(spec, scaling_factor=8)
    #         for spec in tqdm(specs, desc="scaling spectrograms", leave=False)
    #     )

    # Add note sequences to dataframe for later use
    indv_dfs[indv]["syllables_sequence_id"] = None
    indv_dfs[indv]["syllables_sequence_pos"] = None
    for ki, key in enumerate(indv_dfs[indv].key.unique()):
        indv_dfs[indv].loc[indv_dfs[indv].key == key, "syllables_sequence_id"] = ki
        indv_dfs[indv].loc[
            indv_dfs[indv].key == key, "syllables_sequence_pos"
        ] = np.arange(np.sum(indv_dfs[indv].key == key))

    specs_flattened = flatten_spectrograms(specs_scaled)

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
        ((len(specs_flattened) - min(syllable_n)) * (0.1 - 0.02))
        / (max(syllable_n) - min(syllable_n))
    ) + 0.02

    # umap_viz
    #n_neighbors=60, min_dist=min_dist, n_components=2, verbose=True
    fit = umap.UMAP(n_components=2, n_neighbors=80, min_dist=min_dist)
    z = list(fit.fit_transform(specs_flattened))
    indv_dfs[indv]["umap_viz"] = z

    return {indv : indv_dfs[indv]}


def cluster_individual(indv_dfs, indv):
    """Cluster notes into types for each bird using HDBSCAN and from a PCA embedding. (NOTE: -1 = noise label)

    Args:
        indv_dfs (dict): A dictionary of dataframes, one per individual
        indv (str): Individual to select

    Returns:
        dict: Same dictionary with cluster membership added to selected data
    """    
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
    # print(indv + ":" + str(len(indv_dfs[indv]["hdbscan_labels"].unique())))
    return {indv : indv_dfs[indv]}