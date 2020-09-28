# %%
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from IPython import get_ipython
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm.autonotebook import tqdm

from src.avgn.dataset import DataSet
from src.avgn.signalprocessing.create_spectrogram_dataset import *
from src.avgn.utils.hparams import HParams
from src.avgn.utils.paths import ensure_dir, most_recent_subdirectory
from src.avgn.visualization.projections import scatter_spec
from src.avgn.visualization.quickplots import draw_projection_plots

# from cuml.manifold.umap import UMAP as cumlUMAP
from src.avgn.visualization.spectrogram import draw_spec_set
from src.greti.read.paths import DATA_DIR

# get_ipython().run_line_magic("load_ext", "autoreload")
# get_ipython().run_line_magic("autoreload", "2")
# get_ipython().run_line_magic("matplotlib", "inline")


# %%
DATASET_ID = "GRETI_HQ_2020_segmented"


# %% [markdown]
# ### create dataset

# %%

hparams = HParams(
    num_mel_bins=64,
    n_fft=1024,
    win_length_ms=15,
    hop_length_ms=3,
    mel_lower_edge_hertz=1200,
    mel_upper_edge_hertz=9000,
    butter_lowcut=1200,
    butter_highcut=9000,
    ref_level_db=30,
    min_level_db=-30,
    mask_spec=True,
    n_jobs=-2,
    verbosity=1,
    nex=-1,
)


# %%
# create a dataset object
dataset = DataSet(DATASET_ID, hparams=hparams)
dataset.sample_json
len(dataset.data_files)

# %% [markdown]
# #### Create dataset from JSON files

# %%

n_jobs = -2
verbosity = 10

# %%
with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
    syllable_dfs = parallel(
        delayed(create_label_df)(
            dataset.data_files[key].data,
            hparams=dataset.hparams,
            labels_to_retain=[],
            unit="syllables",
            dict_features_to_retain=[],
            key=key,
        )
        for key in tqdm(dataset.data_files.keys())
    )

syllable_df = pd.concat(syllable_dfs)
len(syllable_df)
syllable_df[:3]

# %% [markdown]
# ### get audio for dataset

# %%
with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
    syllable_dfs = parallel(
        delayed(get_row_audio)(
            syllable_df[syllable_df.key == key],
            dataset.data_files[key].data["wav_loc"],
            dataset.hparams,
        )
        for key in tqdm(syllable_df.key.unique())
    )
syllable_df = pd.concat(syllable_dfs)
len(syllable_df)

# %%

df_mask = np.array([len(i) > 0 for i in tqdm(syllable_df.audio.values)])
syllable_df = syllable_df[np.array(df_mask)]
syllable_df[:5]
sylls = syllable_df.audio.values

syllable_df["audio"] = [librosa.util.normalize(i) for i in syllable_df.audio.values]
sylls = syllable_df["audio"].values

# %%

# Plot amplitude envelope of a few syllables
nrows = 10
ncols = 5
zoom = 2
fig, axs = plt.subplots(
    ncols=ncols, nrows=nrows, figsize=(ncols * zoom, nrows + zoom / 1.5)
)
for i, syll in tqdm(enumerate(sylls), total=nrows * ncols):
    ax = axs.flatten()[i]
    ax.plot(syll)
    if i == nrows * ncols - 1:
        break

fig.tight_layout(pad=3.0)

# %% [markdown]
# ### Create spectrograms

# %%
syllables_wav = syllable_df.audio.values
syllables_rate = syllable_df.rate.values

# %%

with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
    # create spectrograms
    syllables_spec = parallel(
        delayed(make_spec)(
            syllable,
            rate,
            hparams=dataset.hparams,
            mel_matrix=dataset.mel_matrix,
            use_mel=True,
            use_tensorflow=False,
        )
        for syllable, rate in tqdm(
            zip(syllables_wav, syllables_rate),
            total=len(syllables_rate),
            desc="getting syllable spectrograms",
            leave=False,
        )
    )

# %%
plt.matshow(syllables_spec[20])

# %% [markdown]
# ### Rescale spectrogram
# - using log rescaling

# %%
log_scaling_factor = 10

with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
    syllables_spec = parallel(
        delayed(log_resize_spec)(spec, scaling_factor=log_scaling_factor)
        for spec in tqdm(syllables_spec, desc="scaling spectrograms", leave=False)
    )


# %%
draw_spec_set(syllables_spec, zoom=1, maxrows=20, colsize=15)


# %% [markdown]
# ### Pad spectrograms

# %%
syll_lens = [np.shape(i)[1] for i in syllables_spec]
pad_length = np.max(syll_lens)

# %%

# for indv in np.unique(syllable_df.indv):
#     sns.distplot(
#         np.log(
#             syllable_df[syllable_df.indv == indv]["end_time"]
#             - syllable_df[syllable_df.indv == indv]["start_time"]
#         ),
#         label=indv,
#     )

# plt.legend()


# %%
with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:

    syllables_spec = parallel(
        delayed(pad_spectrogram)(spec, pad_length)
        for spec in tqdm(syllables_spec, desc="padding spectrograms", leave=False)
    )


# %%
draw_spec_set(syllables_spec, zoom=1, maxrows=15, colsize=15)
np.shape(syllables_spec)


# %%

# Clip range to add contrast


def contrast(x):
    minval = np.percentile(x, 5)
    maxval = np.percentile(x, 100)
    x = np.clip(x, minval, maxval)
    x = ((x - minval) / (maxval - minval)) * 255
    return x


# convert to uint8 to save space
syllables_spec = [contrast(i).astype("uint8") for i in tqdm(syllables_spec)]
syllable_df["spectrogram"] = syllables_spec


# %% [markdown]
# ### view syllables per indv

# %%
syllable_df.indv.unique()


# %%
# for indv in np.sort(syllable_df.indv.unique()):
#     print(indv, np.sum(syllable_df.indv == indv))
#     specs = np.array(
#         [
#             i / np.max(i)
#             for i in syllable_df[syllable_df.indv == indv].spectrogram.values
#         ]
#     )
#     specs[specs < 0] = 0
#     draw_spec_set(specs, zoom=2, maxrows=20, colsize=20)


# %% [markdown]
# ### save dataset

# %%

save_loc = DATA_DIR / "syllable_dfs" / DATASET_ID / "{}.pickle".format(DATASET_ID)
ensure_dir(save_loc)
syllable_df.drop("audio", 1).to_pickle(save_loc)


# %%

# UMAP embedding for all birds in dataset

specs = list(syllable_df.spectrogram.values)
specs = [i / np.max(i) for i in specs]
specs_flattened = flatten_spectrograms(specs)
np.shape(specs_flattened)

specs_flattened = flatten_spectrograms(specs)
fit = umap.UMAP(min_dist=0.25)
z = list(fit.fit_transform(specs_flattened))


# %%

scatter_spec(
    z,
    specs,
    column_size=15,
    # x_range = [-5.5,7],
    # y_range = [-10,10],
    pal_color="hls",
    color_points=False,
    enlarge_points=20,
    figsize=(10, 10),
    scatter_kwargs={
        "labels": syllable_df.indv.values,
        "alpha": 1.0,
        "s": 3,
        "color_palette": "Set2",
        "show_legend": True,
    },
    matshow_kwargs={"cmap": plt.cm.Greys},
    line_kwargs={"lw": 1, "ls": "solid", "alpha": 0.25,},
    draw_lines=True,
    border_line_width=0.5,
)
# %%

# For each bird

for indv in np.sort(syllable_df.indv.unique()):
    # if indv != "Bird4":
    #     continue
    print(indv, np.sum(syllable_df.indv == indv))
    specs = np.array(
        [
            i / np.max(i)
            for i in syllable_df[syllable_df.indv == indv].spectrogram.values
        ]
    )

    specs_flattened = flatten_spectrograms(specs)
    fit = umap.UMAP(min_dist=0.40)
    z = list(fit.fit_transform(specs_flattened))

    scatter_spec(
        np.vstack(z),
        specs,
        column_size=15,
        # x_range = [-5.5,7],
        # y_range = [-10,10],
        pal_color="hls",
        color_points=False,
        enlarge_points=20,
        figsize=(10, 10),
        scatter_kwargs={
            "labels": list(syllable_df[syllable_df.indv == indv]["indv"].values),
            "alpha": 0.30,
            "s": 1,
            "show_legend": True,
        },
        matshow_kwargs={"cmap": plt.cm.Greys},
        line_kwargs={"lw": 1, "ls": "solid", "alpha": 0.25,},
        draw_lines=True,
        border_line_width=0.5,
    )
    plt.show()


# %%

# Save dataframe with embeddings for each bird
ensure_dir(DATA_DIR / "embeddings" / DATASET_ID)


for indv in tqdm(syllable_df.indv.unique()):
    subset_df = syllable_df[syllable_df.indv == indv]

    specs = list(subset_df.spectrogram.values)
    specs = [i / np.max(i) for i in tqdm(specs)]
    specs_flattened = flatten_spectrograms(specs)
    print(np.shape(specs_flattened))

    fit = umap.UMAP(min_dist=0.20)
    embedding = fit.fit_transform(specs_flattened)
    subset_df["umap"] = list(embedding)
    subset_df.to_pickle(DATA_DIR / "embeddings" / DATASET_ID / (indv + ".pickle"))
