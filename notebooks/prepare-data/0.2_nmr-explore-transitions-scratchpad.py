# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().run_line_magic('env', 'CUDA_DEVICE_ORDER=PCI_BUS_ID')
get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=2')


# %%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# %%
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm.autonotebook import tqdm
from joblib import Parallel, delayed
import umap
import pandas as pd


# %%
from avgn.utils.paths import DATA_DIR, most_recent_subdirectory, ensure_dir
from avgn.signalprocessing.create_spectrogram_dataset import flatten_spectrograms
from avgn.visualization.spectrogram import draw_spec_set
from avgn.visualization.quickplots import draw_projection_plots

# %% [markdown]
# ### Collect data

# %%
DATASET_ID = 'bengalese_finch_sober'


# %%
from avgn.visualization.projections import (
    scatter_projections,
    draw_projection_transitions,
)


# %%
df_loc =  DATA_DIR / 'syllable_dfs' / DATASET_ID / 'bf.pickle'
df_loc


# %%
syllable_df = pd.read_pickle(df_loc)


# %%
syllable_df[:3]


# %%
len(syllable_df)


# %%
def norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

# %% [markdown]
# ### cluster

# %%
from avgn.visualization.projections import scatter_spec
from avgn.utils.general import save_fig
from avgn.utils.paths import FIGURE_DIR, ensure_dir


# %%
from cuml.manifold.umap import UMAP as cumlUMAP


# %%
import hdbscan


# %%
from avgn.visualization.projections import draw_projection_transitions


# %%
fig, ax = plt.subplots(nrows=2, ncols=len(syllable_df.indv.unique()), figsize=(10*len(syllable_df.indv.unique()), 20))

indv_dfs = {}
for indvi, indv in enumerate(tqdm(syllable_df.indv.unique())):
    #if indv != 'Bird5': continue
    indv_dfs[indv] = syllable_df[syllable_df.indv == indv]
    indv_dfs[indv] = indv_dfs[indv].sort_values(by=["key", "start_time"])
    print(indv, len(indv_dfs[indv]))
    specs = [norm(i) for i in indv_dfs[indv].spectrogram.values]
    
    # sequencing
    indv_dfs[indv]["syllables_sequence_id"] = None
    indv_dfs[indv]["syllables_sequence_pos"] = None
    for ki, key in enumerate(indv_dfs[indv].key.unique()):
        indv_dfs[indv].loc[indv_dfs[indv].key == key, "syllables_sequence_id"] = ki
        indv_dfs[indv].loc[indv_dfs[indv].key == key, "syllables_sequence_pos"] = np.arange(
            np.sum(indv_dfs[indv].key == key)
        )
        
    # umap
    specs_flattened = flatten_spectrograms(specs)
    cuml_umap = cumlUMAP(min_dist=0.5)
    z = list(cuml_umap.fit_transform(specs_flattened))
    indv_dfs[indv]["umap"] = z

    # plot
    scatter_spec(
        np.vstack(z),
        specs,
        column_size=15,
        #x_range = [-5.5,7],
        #y_range = [-10,10],
        pal_color="hls",
        color_points=False,
        enlarge_points=20,
        figsize=(10, 10),
        scatter_kwargs = {
            'labels': list(indv_dfs[indv].labels.values),
            'alpha':0.25,
            's': 1,
            'show_legend': False
        },
        matshow_kwargs = {
            'cmap': plt.cm.Greys
        },
        line_kwargs = {
            'lw':1,
            'ls':"solid",
            'alpha':0.25,
        },
        draw_lines=True,
        ax= ax[0,indvi]
    );
    
    draw_projection_transitions(
        projections=np.array(list(indv_dfs[indv]['umap'].values)),
        sequence_ids=indv_dfs[indv]["syllables_sequence_id"],
        sequence_pos=indv_dfs[indv]["syllables_sequence_pos"],
        ax=ax[1,indvi],
    )

    

# %% [markdown]
# ### label

# %%
from sklearn.cluster import MiniBatchKMeans


# %%
for indv in tqdm(indv_dfs.keys()):
    ### cluster
    #break
    z = list(indv_dfs[indv]["umap"].values)
    # HDBSCAN UMAP
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(len(z) * 0.01), # the smallest size we would expect a cluster to be
        min_samples=1, # larger values = more conservative clustering
    )
    clusterer.fit(z);
    indv_dfs[indv]['hdbscan_labels'] = clusterer.labels_
    
    # HDBSCAN 
    specs = [norm(i) for i in indv_dfs[indv].spectrogram.values]
    specs_flattened = flatten_spectrograms(specs)
    
    # kmeans
    # get number of possible states
    n_states = len(indv_dfs[indv].labels.unique())

    kmeans = MiniBatchKMeans(n_clusters = n_states).fit(z)
    indv_dfs[indv]['kmeans_labels'] = kmeans.labels_
    
    # kmeans data
    kmeans = MiniBatchKMeans(n_clusters = n_states).fit(specs_flattened)
    indv_dfs[indv]['kmeans__pix_labels'] = kmeans.labels_

# %% [markdown]
# ### plot

# %%
for indv in tqdm(indv_dfs.keys()):
    fig, axs = plt.subplots(ncols=5, figsize=(50, 10))
    draw_projection_transitions(
        projections=np.array(list(indv_dfs[indv]["umap"].values)),
        sequence_ids=indv_dfs[indv]["syllables_sequence_id"],
        sequence_pos=indv_dfs[indv]["syllables_sequence_pos"],
        ax=axs[0],
    )

    for i, lab in enumerate(
        [
            "labels",
            "kmeans_labels",
            "kmeans__pix_labels",
            "hdbscan_labels",
        ]
    ):

        scatter_spec(
            np.array(list(indv_dfs[indv]["umap"].values)),
            specs = indv_dfs[indv].spectrogram.values,
            column_size=15,
            # x_range = [-5.5,7],
            # y_range = [-10,10],
            pal_color="hls",
            color_points=False,
            enlarge_points=20,
            figsize=(10, 10),
            scatter_kwargs={
                "labels": list(indv_dfs[indv][lab].values),
                "alpha": 0.25,
                "s": 1,
                "show_legend": False,
            },
            matshow_kwargs={"cmap": plt.cm.Greys},
            line_kwargs={"lw": 1, "ls": "solid", "alpha": 0.25},
            draw_lines=True,
            ax=axs[i + 1],
        )
        axs[i + 1].set_title(lab)
    plt.show()

# %% [markdown]
# ### human vs algorithmic labelling similarity

# %%
import sklearn.metrics


# %%
performance_df = pd.DataFrame(columns = ['indv', 'cluster', 'homogeneity', 'completeness', 'V-Measure', 'Adj. MI'])
for indv in tqdm(indv_dfs.keys()):
    for cluster in ['hdbscan_labels', 'kmeans__pix_labels', 'kmeans_labels']:
        homogenaity, completeness, v_measure = sklearn.metrics.homogeneity_completeness_v_measure(
            list(indv_dfs[indv].labels), list(indv_dfs[indv][cluster].values)
        )
        ami = sklearn.metrics.adjusted_mutual_info_score(
            list(indv_dfs[indv].labels), list(indv_dfs[indv][cluster].values)
        )
        performance_df.loc[len(performance_df)] = [indv, cluster, homogenaity, completeness, v_measure, ami]


# %%
performance_df[:4]


# %%
summary = performance_df.groupby(['cluster']).describe()


# %%
import seaborn as sns
sns.set_context("paper", font_scale=2)


# %%
fig, axs = plt.subplots(ncols = 4, figsize=(20,4))

for ci, column in enumerate(["homogeneity", "completeness", "V-Measure", "Adj. MI"]):
    sns.boxplot(x="cluster", y=column, data =performance_df, ax = axs[ci])
    sns.swarmplot(x="cluster", y=column, data =performance_df, ax = axs[ci], color=".25")
    axs[ci].set_ylim([0.5,1])
    axs[ci].set_xticklabels(['HDBSCAN/UMAP', 'KMeans', 'KMeans/UMAP'], rotation=45, ha='right')
    axs[ci].set_xlabel('')
    axs[ci].set_ylabel('')
    axs[ci].set_title(column)


# %%
cats = ["homogeneity", "completeness", "V-Measure", "Adj. MI"]
results_latex_df = pd.DataFrame(
    columns=["Homogeneity", "Completeness", "V-Measure", "Adjusted MI"]
)

labs = [['hdbscan_labels', 'HDBSCAN/UMAP'], ['kmeans__pix_labels', 'KMeans'], ['kmeans_labels', 'KMeans/UMAP']]

for lab, name in labs:
    results_latex_df.loc[name] = [
        str(
            round(np.mean(performance_df[performance_df.cluster == lab][i].values),3)
        ).zfill(3)
        + "\u00B1"
        + str(
            round(np.std(performance_df[performance_df.cluster == lab][i].values),3)
        ).zfill(3)
        for i in cats
    ]
results_latex_df = pd.concat([results_latex_df], keys=[''], names=['Nicholson et al., ()'])
results_latex_df


# %%
results_string = results_latex_df.to_latex(bold_rows=True, escape=False)      .replace('>', '$>$').replace('±', '$\pm$')      .replace('<', '$<$')      .replace('superlabel', '')     .replace('\n\\textbf', '\n\midrule\n\\textbf')
print(results_string)


# %%



