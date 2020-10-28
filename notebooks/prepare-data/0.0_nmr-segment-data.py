# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# null.tpl[markdown]
# # 0.0 Segment raw data and prepare syllables for analysis
#
# > **Note:** this works for the recordings of a given year only
#
# ### This notebook does the following:
#  - Segments raw recordings into manually defined songs
#  > saved as new .wav files and .json files with metadata
#  - Segments songs into syllables
#  > saved as on-off times in each .json file
#

# %%
# Reload modules automatically
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")


# %%
import numpy as np
import pandas as pd
import src
import glob
import joblib
from os import fspath

import warnings

warnings.filterwarnings(action="once")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")

from src.vocalseg.dynamic_thresholding import *
from src.avgn.dataset import DataSet
from src.avgn.utils.hparams import HParams
from src.avgn.utils.paths import most_recent_subdirectory
from tqdm.autonotebook import tqdm
from src.greti.read.paths import DATA_DIR, RESOURCES_DIR
from src.greti.audio.segmentation import *
from IPython import get_ipython
from IPython.display import display, HTML, display_html


# %%
# Set year
year = "2020"


# %%
# import recorded nestboxes
files_path = DATA_DIR / "raw" / year
filelist = np.sort(list(files_path.glob("**/*.WAV")))
recorded_nestboxes = pd.DataFrame(set([file.parent.name for file in filelist]))

# import the latest brood data downloaded from https://ebmp.zoo.ox.ac.uk/broods
brood_data_path = RESOURCES_DIR / "brood_data" / year
list_of_files = glob.glob(fspath(brood_data_path) + "/*.csv")
latest_file = max(list_of_files, key=os.path.getctime)
greti_nestboxes = pd.DataFrame(
    (pd.read_csv(latest_file).query('Species == "g"').filter(["Pnum"]))["Pnum"].str[5:]
)
# get those in both lists
recorded_gretis = [
    i
    for i in recorded_nestboxes.values.tolist()
    if i in greti_nestboxes.values.tolist()
]

print("You recorded a total of " + str(len(filelist)) + " hours of audio.")
print(
    "You recorded "
    + str(len(recorded_gretis))
    + " out of a total of "
    + str(len(greti_nestboxes))
    + " great tits that bred this year"
)

# null.tpl[markdown]
# # ### Segment raw recordings into bouts
# #  - Songs manually defined in AviaNZ - for now
# #
# # > `batch_segment_bouts()` usis multiprocessing. If you run into problems, use `batch_segment_bouts_single()` (much slower).

# # %%
# origin = DATA_DIR / "raw" / year  # Folder to segment
# DT_ID = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Unique name for output folder
# subset = "GRETI_HQ"  # Name of label to select
# DATASET_ID = "GRETI_HQ_2020"  # Name of output dataset
# threshold = 5000  # Amplitude threshold


# # %%
# batch_segment_songs(
#     origin, DATA_DIR, DT_ID, DATASET_ID, subset=subset, threshold=threshold
# )

# null.tpl[markdown]
# # - Let's check how many songs have been exported:

# # %%

# all_songs_path = most_recent_subdirectory(DATA_DIR / "processed" / DATASET_ID)
# all_songs_list = np.sort(list(all_songs_path.glob("**/*.wav")))
# print("There are " + str(len(all_songs_list)) + " songs")

# null.tpl[markdown]
# # # Syllable segmentation
# null.tpl[markdown]
# # ### Create dataset

# %%

# Dataset label
DATASET_ID = "GRETI_HQ_2020"

# datetime identifier for output folder
DT_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DT_ID

# Define parameters
hparams = HParams(
    n_fft=1024,
    win_length_ms=4,
    hop_length_ms=3,
    mel_lower_edge_hertz=1200,
    mel_upper_edge_hertz=10000,
    butter_lowcut=1200,
    butter_highcut=10000,
    ref_level_db=30,
    min_level_db=-30,
    n_jobs=-2,
    verbosity=1,
    nex=-1,
)


# %%
# create a dataset object
dataset = DataSet(DATASET_ID, hparams=hparams)

# Check a sample json
dataset.sample_json


# ### Define parameters

# %%
# Segmentation parameters
parameters = {
    "n_fft": 1024,
    "hop_length_ms": 3,
    "win_length_ms": 10,
    "ref_level_db": 30,
    "pre": 0.5,
    "min_level_db": -25,
    "min_level_db_floor": -20,
    "db_delta": 8,
    "silence_threshold": 0.2,
    "min_silence_for_spec": 0.001,
    "max_vocal_for_spec": (0.4,),
    "min_syllable_length_s": 0.03,
    "spectral_range": [1200, 10000],
}

# There needs to be a silence of at least min_silence_for_spec length,
# and a syllable no longer than max_vocal_for_spec length


# ### Check a sample song

# %%
rate, data = load_wav(dataset.sample_json["wav_loc"])
butter_min = dataset.sample_json["lower_freq"]
butter_max = dataset.sample_json["upper_freq"]
data = butter_bandpass_filter(data, butter_min, butter_max, rate)
data = librosa.util.normalize(data)

plt.figure(figsize=(10.05, 2))
plt.plot(data)

results = dynamic_threshold_segmentation(data, rate, **parameters)

plot_segmentations(
    results["spec"],
    results["vocal_envelope"],
    results["onsets"],
    results["offsets"],
    hop_length_ms=3,
    rate=rate,
    figsize=(10, 3),
)

plt.show()

null.tpl[markdown]
# ### Test segmentation in a subset of the data

# %%
indvs = np.array(["_".join(list(i)) for i in dataset.json_indv])
np.unique(indvs)
len(np.unique(indvs))


# %%

# for indv in tqdm(np.unique(indvs), desc="individuals"):
#     print(indv)
#     indv_keys = np.array(list(dataset.data_files.keys()))[indvs == indv][20:25]

#     joblib.Parallel(n_jobs=1, verbose=0)(
#         joblib.delayed(segment_spec_custom)(
#             key, dataset.data_files[key], **parameters, DT_ID=DT_ID, DATASET_ID=DATASET_ID, plot=True
#         )
#         for key in tqdm(indv_keys, desc="files", leave=False)
#     )

null.tpl[markdown]
# ### Segment full dataset

# %%
nex = -1
for indv in tqdm(np.unique(indvs), desc="individuals"):
    print(indv)
    indv_keys = np.array(list(dataset.data_files.keys()))[indvs == indv]

    joblib.Parallel(n_jobs=-2, verbose=1)(
        joblib.delayed(segment_spec_custom)(
            key,
            dataset.data_files[key],
            **parameters,
            DT_ID=DT_ID,
            DATASET_ID=DATASET_ID,
            save=True
        )
        for key in tqdm(indv_keys, desc="files", leave=True)
    )


# %%

