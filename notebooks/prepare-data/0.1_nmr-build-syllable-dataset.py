# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from datetime import datetime

import librosa
import matplotlib.pyplot as plt
import numpy as np

# %%
import pandas as pd
from IPython import get_ipython
from joblib import Parallel, delayed
from src.avgn.dataset import DataSet
from src.avgn.signalprocessing.create_spectrogram_dataset import *
from src.avgn.utils.hparams import HParams
from src.avgn.utils.paths import ensure_dir, most_recent_subdirectory
from src.avgn.visualization.spectrogram import draw_spec_set
from src.greti.read.paths import DATA_DIR
from tqdm.autonotebook import tqdm

# %% [markdown]
# ## 0.1 Build dataset of song syllables with their spectrograms
#
# ### This notebook does the following:
#  - Creates spectrograms for each syllable
#  - Saves a dataset to be used in furhter analyses
#

# %%
# Reload modules automatically
# get_ipython().run_line_magic("load_ext", "autoreload")
# get_ipython().run_line_magic("autoreload", "2")


# get_ipython().run_line_magic("matplotlib", "inline")

# import seaborn as sns
# import src


# %% [markdown]
# ### Select dataset and parameters
# > Use output of the previous notebook (*_segmented)

# %%
DATASET_ID = "GRETI_HQ_2020_segmented"

n_jobs = -1
verbosity = 10

hparams = HParams(
    num_mel_bins=64,
    n_fft=1024,
    win_length_ms=15,
    hop_length_ms=3,
    mel_lower_edge_hertz=1200,
    mel_upper_edge_hertz=10000,
    butter_lowcut=1200,
    butter_highcut=10000,
    ref_level_db=30,
    min_level_db=-19,
    mask_spec=True,
    n_jobs=-1,
    verbosity=1,
    nex=-1,
)


# %%

# create a dataset object
dataset = DataSet(DATASET_ID, hparams=hparams)
print(dataset.sample_json)
print(len(dataset.data_files))


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
print(len(syllable_df))
# syllable_df[:3]

# %% [markdown]
# ### Get audio for dataset

# %%

with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
    syllable_dfs = parallel(
        delayed(get_row_audio)(
            syllable_df[syllable_df.key == key],
            dataset.data_files[key]
            .data["wav_loc"]
            .replace("/home/nilomr", "/data/zool-songbird/shil5293"),
            # dataset.data_files[key].data["wav_loc"], # if local machine
            # replace with your respective home directiories
            dataset.hparams,
        )
        for key in tqdm(syllable_df.key.unique(), position=0, leave=True)
    )
syllable_df = pd.concat(syllable_dfs)
print(len(syllable_df))

# %% [markdown]
# ### Normalise audio

# %%

df_mask = np.array([len(i) > 0 for i in tqdm(syllable_df.audio.values)])
syllable_df = syllable_df[np.array(df_mask)]
sylls = syllable_df.audio.values

syllable_df["audio"] = [librosa.util.normalize(i) for i in syllable_df.audio.values]
sylls = syllable_df["audio"].values

# %% [markdown]
# - Plot amplitude envelope of a few syllables

# %%
# nrows = 5
# ncols = 5
# zoom = 2
# fig, axs = plt.subplots(
#     ncols=ncols, nrows=nrows, figsize=(ncols * zoom, nrows + zoom / 1.5)
# )
# for i, syll in tqdm(enumerate(sylls), total=nrows * ncols):
#     ax = axs.flatten()[i]
#     ax.plot(syll)
#     if i == nrows * ncols - 1:
#         break

# fig.tight_layout(pad=3.0)

# %% [markdown]
# ### Create spectrograms

# %%

syllables_wav = syllable_df.audio.values
syllables_rate = syllable_df.rate.values


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
            position=0,
            leave=True,
        )
    )

# %% [markdown]
# - Plot one example

# %%
# plt.matshow(syllables_spec[20])

# %% [markdown]
# ### Rescale and pad spectrograms

# %%

log_scaling_factor = 10

with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
    syllables_spec = parallel(
        delayed(log_resize_spec)(spec, scaling_factor=log_scaling_factor)
        for spec in tqdm(syllables_spec, desc="scaling spectrograms", leave=False)
    )


# %%
syll_lens = [np.shape(i)[1] for i in syllables_spec]
pad_length = np.max(syll_lens)


with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
    syllables_spec = parallel(
        delayed(pad_spectrogram)(spec, pad_length)
        for spec in tqdm(syllables_spec, desc="padding spectrograms", leave=False)
    )

# %% [markdown]
# - Plot a few

# %%
print(np.shape(syllables_spec))
# draw_spec_set(syllables_spec, zoom=2, maxrows=15, colsize=15)

# %% [markdown]
# ### Clip range and convert to uint8

# %%
# Clip range to add contrast
def contrast(x):
    minval = np.percentile(x, 5)
    maxval = np.percentile(x, 100)
    x = np.clip(x, minval, maxval)
    x = ((x - minval) / (maxval - minval)) * 255
    return x


syllables_spec = [contrast(i).astype("uint8") for i in tqdm(syllables_spec)]
syllable_df["spectrogram"] = syllables_spec

# %% [markdown]
# ### Plot a few sylables per individual

# %%
# syllable_df.indv.unique()

# for indv in np.sort(syllable_df.indv.unique()):
#     print(indv, np.sum(syllable_df.indv == indv))
#     specs = np.array(
#         [
#             i / np.max(i)
#             for i in syllable_df[syllable_df.indv == indv].spectrogram.values
#         ]
#     )
#     specs[specs < 0] = 0
#     draw_spec_set(specs, zoom=2, maxrows=7, colsize=15)

# %% [markdown]
# ### Save entire dataset

# %%
DT_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

save_loc = (
    DATA_DIR / "syllable_dfs" / DATASET_ID / DT_ID / "{}.pickle".format(DATASET_ID)
)
ensure_dir(save_loc)
syllable_df.drop("audio", 1).to_pickle(save_loc)

print("Done")


# %%
