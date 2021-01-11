
#%%
import numpy as np
import pandas as pd
from src.greti.read.paths import DATA_DIR
from src.avgn.utils.paths import ensure_dir, most_recent_subdirectory
from random import randrange
import librosa
from src.vocalseg.utils import butter_bandpass_filter
from tqdm.autonotebook import tqdm

#%%

DATASET_ID = "GRETI_HQ_2020_segmented"
YEAR = "2020"
cluster_labels = "hdbscan_labels_fixed"

dfs_dir = DATA_DIR / "indv_dfs" / DATASET_ID
indv_dfs = pd.read_pickle(dfs_dir / (f"{DATASET_ID}_labelled_checked.pickle"))


all_birds = list(indv_dfs.keys())

exclude = ['CP34', 'EX38A', 'SW119', 'B11', 'MP20', 'B161', 'EX34', 'CP28', 'B49', 'SW116'] # Data for these are too noisy, judged from UMAP scatterplot in previous step
indvs = [indv for indv in all_birds if indv not in exclude]

butter_lowcut=1200
butter_highcut=10000


out_dir = DATA_DIR / 'processed' / f'{DATASET_ID}_notes'
ensure_dir(out_dir)

#%%
for indv in tqdm(indvs, desc="Saving wav files for each note", leave=True):
    for label in indv_dfs[indv][cluster_labels].unique():
        if label != -1:
            nrows = len(indv_dfs[indv][indv_dfs[indv][cluster_labels] == label])
            index = randrange(nrows) # choose one for now

            data = indv_dfs[indv][indv_dfs[indv][cluster_labels] == label].iloc[index]
            wav_loc = most_recent_subdirectory(DATA_DIR /'processed'/ DATASET_ID.replace('_segmented', ''), only_dirs=True) / 'WAV' / data.key

            # Load and trim audio
            y, sr = librosa.load(wav_loc, sr = 32000)
            y = y[int(data.start_time * sr) : int(data.end_time * sr)]
            # y = butter_bandpass_filter(
            #     y, butter_lowcut, butter_highcut, sr, order=5
            # )
            # Save note audio
            out_filedir = out_dir / f'{data.key[:-4]}-L{label}-I{index}.wav'
            librosa.output.write_wav(out_filedir, y, sr)

# %%

#TODO: with sequences inferred, export syllable dataset in order, then read in R etc
