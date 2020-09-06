# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

from src.read.paths import DATA_DIR
import joblib
from vocalseg.dynamic_thresholding import *
from avgn.dataset import DataSet
from avgn.utils.hparams import HParams
import warnings
import pandas as pd
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython

# %% [markdown]
# # Segment songs into syllables
# Using dynamic thresholding


# %%
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
get_ipython().run_line_magic("matplotlib", "inline")
warnings.filterwarnings(action="once")


# %% [markdown]
# ### Create dataset

# %%

DATASET_ID = "GRETI_HQ"
# datetime identifier for output folder
DT_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DT_ID


hparams = HParams(
    n_fft=1024,
    win_length_ms=4,
    hop_length_ms=3,
    mel_lower_edge_hertz=1200,
    mel_upper_edge_hertz=9000,
    butter_lowcut=1200,
    butter_highcut=9000,
    ref_level_db=30,
    min_level_db=-30,
    n_jobs=-2,
    verbosity=1,
    nex=-1,
)


# %%
# create a dataset object
dataset = DataSet(DATASET_ID, hparams=hparams)


# %%
# Check a sample json

dataset.sample_json


# %%
# Segmentation parameters

parameters = {
    "n_fft": 1024,
    "hop_length_ms": 3,
    "win_length_ms": 15,
    "ref_level_db": 30,
    "pre": 0.4,
    "min_level_db": -26,
    "min_level_db_floor": -15,
    "db_delta": 7,
    "silence_threshold": 0.2,
    "min_silence_for_spec": 0.001,
    "max_vocal_for_spec": (0.4,),
    "min_syllable_length_s": 0.03,
    "spectral_range": [1200, 9000],
}


# There needs to be a silence of at least min_silence_for_spec length,
# and a syllable no longer than max_vocal_for_spec length

# %%

# Plot one example

rate, data = load_wav(dataset.sample_json["wav_loc"])
butter_min = dataset.sample_json["lower_freq"]
butter_max = dataset.sample_json["upper_freq"]
data = butter_bandpass_filter(data, butter_min, butter_max, rate)
data = librosa.util.normalize(data)
plt.plot(data)

# Plot one example

results = dynamic_threshold_segmentation(data, rate, **parameters)

plot_segmentations(
    results["spec"],
    results["vocal_envelope"],
    results["onsets"],
    results["offsets"],
    hop_length_ms=3,
    rate=rate,
    figsize=(15, 5),
)

plt.show()


# %%
indvs = np.array(["_".join(list(i)) for i in dataset.json_indv])
np.unique(indvs)

for indv in tqdm(np.unique(indvs), desc="individuals"):
    print(indv)
    indv_keys = np.array(list(dataset.data_files.keys()))[indvs == indv][20:25]

    joblib.Parallel(n_jobs=1, verbose=0)(
        joblib.delayed(segment_spec_custom)(
            key, dataset.data_files[key], **parameters, plot=True
        )
        for key in tqdm(indv_keys, desc="files", leave=False)
    )


#%%


# %% [markdown]
# ### Generate for full dataset

# %%
nex = -1
for indv in tqdm(np.unique(indvs), desc="individuals"):
    print(indv)
    indv_keys = np.array(list(dataset.data_files.keys()))[indvs == indv]

    joblib.Parallel(n_jobs=-2, verbose=1)(
        joblib.delayed(segment_spec_custom)(
            key, dataset.data_files[key], **parameters, save=True
        )
        for key in tqdm(indv_keys, desc="files", leave=False)
    )
