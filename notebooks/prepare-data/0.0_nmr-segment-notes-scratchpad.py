# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from avgn.utils.json import NoIndent, NoIndentEncoder
import joblib
import librosa
from avgn.dataset import DataSet
from avgn.utils.hparams import HParams
from avgn.utils.paths import most_recent_subdirectory, ensure_dir  # , DATA_DIR #! ADD
from vocalseg.dynamic_thresholding import plot_segmented_spec, plot_segmentations
from vocalseg.dynamic_thresholding import dynamic_threshold_segmentation
from avgn.signalprocessing.filtering import butter_bandpass_filter
from avgn.utils.audio import load_wav, read_wav
import warnings
from datetime import datetime
import pandas as pd
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython

# %% [markdown]
# ### segment waveform into individual syllables using dynamic thresholding


# %%
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
get_ipython().run_line_magic("matplotlib", "inline")
warnings.filterwarnings(action="once")


# %% [markdown]
# ### Load data in original format

# %%

DATASET_ID = "GRETI_HQ"
# datetime identifier for output folder
DT_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DT_ID

# %% [markdown]
# ### create dataset

# %%
hparams = HParams(
    n_fft=4096,
    mel_lower_edge_hertz=1900,
    mel_upper_edge_hertz=9000,
    # butter_lowcut=500,
    # butter_highcut=12000,
    ref_level_db=20,
    min_level_db=-100,
    win_length_ms=4,
    hop_length_ms=1,
    n_jobs=-2,
    verbosity=1,
    nex=-1,
)


# %%
# create a dataset object
dataset = DataSet(DATASET_ID, hparams=hparams)

# TODO: modify functions so that they bandpass per .wav AFTER those below
# TODO: threshold are rejected, etc


# %%
dataset.sample_json


# %%
# segmentation parameters
n_fft = 1024
hop_length_ms = 1
win_length_ms = 5
ref_level_db = 15
pre = 0.97
min_level_db = -120
min_level_db_floor = -20
db_delta = 5
silence_threshold = 0.01
min_silence_for_spec = 0.001
max_vocal_for_spec = (0.225,)
min_syllable_length_s = 0.025
butter_min = 500
butter_max = 8000
spectral_range = [500, 8000]


# %%


# %%
rate, data = load_wav(dataset.sample_json["wav_loc"])

np.min(data), np.max(data)

data = data / np.max(np.abs(data))

# filter data
data = butter_bandpass_filter(data, butter_min, butter_max, rate)


# %%
plt.plot(data)


# %%
# segment
results = dynamic_threshold_segmentation(
    data,
    rate,
    n_fft=n_fft,
    hop_length_ms=hop_length_ms,
    win_length_ms=win_length_ms,
    min_level_db_floor=min_level_db_floor,
    db_delta=db_delta,
    ref_level_db=ref_level_db,
    pre=pre,
    min_silence_for_spec=min_silence_for_spec,
    max_vocal_for_spec=max_vocal_for_spec,
    min_level_db=min_level_db,
    silence_threshold=silence_threshold,
    verbose=True,
    min_syllable_length_s=min_syllable_length_s,
    spectral_range=spectral_range,
)


# %%
plot_segmentations(
    results["spec"],
    results["vocal_envelope"],
    results["onsets"],
    results["offsets"],
    hop_length_ms,
    rate,
    figsize=(15, 5),
)
plt.show()


# %% [markdown]
# ### segment and plot
# - for each json, load the wav file - segment the file into start and end times
# - plot the segmentation
# - add to the JSON


# %%
warnings.filterwarnings(
    "ignore", message="'tqdm_notebook' object has no attribute 'sp'"
)


def segment_spec_custom(key, df, save=False, plot=False):
    # load wav
    rate, data = load_wav(df.data["wav_loc"])
    # filter data
    data = butter_bandpass_filter(data, butter_min, butter_max, rate)

    # segment
    results = dynamic_threshold_segmentation(
        data,
        rate,
        n_fft=n_fft,
        hop_length_ms=hop_length_ms,
        win_length_ms=win_length_ms,
        min_level_db_floor=min_level_db_floor,
        db_delta=db_delta,
        ref_level_db=ref_level_db,
        pre=pre,
        min_silence_for_spec=min_silence_for_spec,
        max_vocal_for_spec=max_vocal_for_spec,
        min_level_db=min_level_db,
        silence_threshold=silence_threshold,
        verbose=True,
        min_syllable_length_s=min_syllable_length_s,
        spectral_range=spectral_range,
    )
    if results is None:
        return
    if plot:
        plot_segmentations(
            results["spec"],
            results["vocal_envelope"],
            results["onsets"],
            results["offsets"],
            hop_length_ms,
            rate,
            figsize=(15, 3),
        )
        plt.show()

    # save the results
    json_out = (
        DATA_DIR
        / "processed"
        / (DATASET_ID + "_segmented")
        / DT_ID
        / "JSON"
        / (key + ".JSON")
    )

    json_dict = df.data.copy()

    json_dict["indvs"][list(df.data["indvs"].keys())[0]]["syllables"] = {
        "start_times": NoIndent(list(results["onsets"])),
        "end_times": NoIndent(list(results["offsets"])),
    }

    json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)
    # save json
    if save:
        ensure_dir(json_out.as_posix())
        print(json_txt, file=open(json_out.as_posix(), "w"))

    # print(json_txt)

    return results


# %%
indvs = np.array(["_".join(list(i)) for i in dataset.json_indv])
np.unique(indvs)


# %%
nex = 10
for indv in tqdm(np.unique(indvs), desc="individuals"):
    print(indv)
    indv_keys = np.array(list(dataset.data_files.keys()))[indvs == indv][:nex]

    joblib.Parallel(n_jobs=1, verbose=0)(
        joblib.delayed(segment_spec_custom)(key, dataset.data_files[key], plot=True)
        for key in tqdm(indv_keys, desc="files", leave=False)
    )

# %% [markdown]
# ### Generate for full dataset

# %%
nex = -1
for indv in tqdm(np.unique(indvs), desc="individuals"):
    print(indv)
    indv_keys = np.array(list(dataset.data_files.keys()))[indvs == indv]

    joblib.Parallel(n_jobs=-1, verbose=1)(
        joblib.delayed(segment_spec_custom)(key, dataset.data_files[key], save=True)
        for key in tqdm(indv_keys, desc="files", leave=False)
    )


# %%
