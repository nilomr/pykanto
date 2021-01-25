# %%
import ast
from src.greti.read.paths import safe_makedir
import collections
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib2 import Path
import seaborn as sns
from joblib import Parallel, delayed
from src.avgn.utils.paths import ensure_dir
from src.greti.read.paths import DATA_DIR
from src.greti.sequencing.seqfinder import find_syllable_sequences
from src.greti.write.save_syllables import find_sublist, get_song_dir, is_a_in_x, save_syllable_audio, trim_or_extend_songtype, save_note_audio
from tqdm.autonotebook import tqdm
import librosa
import os

%load_ext autoreload
%autoreload 2

# %%

DATASET_ID = "GRETI_HQ_2020_segmented"
YEAR = "2020"
cluster_labels = "hdbscan_labels_fixed"

dfs_dir = DATA_DIR / "indv_dfs" / DATASET_ID
indv_dfs = pd.read_pickle(dfs_dir / (f"{DATASET_ID}_labelled_checked.pickle"))


all_birds = list(indv_dfs.keys())

exclude = [
    "CP34",
    "EX38A",
    "SW119",
    "B11",
    "MP20",
    "B161",
    "EX34",
    "CP28",
    "B49",
    "SW116",
]  # Data for these are too noisy, judged from UMAP scatterplot in previous step

indvs = [indv for indv in all_birds if indv not in exclude]

butter_lowcut = 1200
butter_highcut = 10000

out_dir = DATA_DIR / "processed" / DATASET_ID.replace('_segmented', "")
safe_makedir(out_dir)


# %%

# First make a dictionary with each bird's inferred repertoire
# NOTE: There is some stochasticity here - not much, but might want to make deterministic.
# Slight variations probably come from cases where two alternative song types are found the same number of times.
syllable_type_dict = {}
for indv in tqdm(indvs, desc="Inferring song type repertoires", leave=True):
    final_dict, n_seqs, _, _ = find_syllable_sequences(
        indv_dfs, indv, cluster_labels, min_freq=1, min_songs=1, double_note_threshold=0.5)
    syllable_type_dict[indv] = [ast.literal_eval(
        key) for key in list(final_dict.keys())]

# %%
# Plot distribution of repertoire size
sns.set_style("dark")
repsize = [len(d) for indv, d in syllable_type_dict.items()]
sns.countplot(repsize, color='grey')
# %%

# Check maximum sequence length:
seqlen = [len(seq) for indv, d in syllable_type_dict.items() for seq in d]
ax, fig = plt.subplots(figsize=(3, 5))
sns.countplot(seqlen, color='grey')
for n, freq in dict(sorted(collections.Counter(seqlen).items(),
                           key=lambda item: item[1],
                           reverse=True)).items():
    print(f'{n} notes = {freq} cases')

# %%
# Save syllables and their metadata
for indv, repertoire in tqdm(syllable_type_dict.items(),
                             desc="Saving syllable audio and metadata",
                             leave=True):
    save_syllable_audio(DATASET_ID,
                        indv_dfs,
                        indv,
                        repertoire,
                        cluster_labels,
                        out_dir,
                        shuffle_songs=True,
                        max_seqlength=3,
                        max_n_sylls=10)
# %%

# Save individual notes
for indv, repertoire in tqdm(syllable_type_dict.items(),
                             desc="Saving note audio and metadata",
                             leave=True):
    save_note_audio(DATASET_ID,
                    indv_dfs,
                    indv,  # dict keys
                    # dict values (lists of lists, where sublists are song types)
                    repertoire,
                    cluster_labels,
                    out_dir,
                    shuffle_songs=True,  # Useful to avoud using very similar examples
                    max_seqlength=3,  # There is no warrantee that this will work with greater lenghts as is
                    max_n_sylls=10,
                    frontpad=0.004,
                    endpad=0.03  # the segmentation algo shaves very closely - add some padding to each note
                    )

# %% Save metadata as csv to read with R (for individual notes)

DATASET_ID = "GRETI_HQ_2020_notes"
out_dir_notes_json = DATA_DIR / "processed" / DATASET_ID / 'JSON'

# Prepare metadata
json_files = [pos_json for pos_json in os.listdir(
    out_dir_notes_json) if pos_json.endswith('.json')]
dict_list = []
for index, js in tqdm(enumerate(json_files)):
    with open(os.path.join(out_dir_notes_json, js)) as json_file:
        dict_list.append(json.load(json_file))
syllables_df = pd.DataFrame(dict_list)
syllables_df['key'] = syllables_df['syll_wav_loc'].apply(
    lambda x: Path(x).stem)
syllables_df[['silence_1', 'silence_2']] = (syllables_df['silences']
                                            .transform([lambda x:x[0], lambda x:x[1]])
                                            .set_axis(['silence_1', 'silence_2'],
                                                      axis=1,
                                                      inplace=False)
                                            )

# Save
out_dir = (DATA_DIR / "note_dfs" / DATASET_ID / str(DATASET_ID + ".csv"))
ensure_dir(out_dir)
syllables_df.to_csv(out_dir, index=False)

# %%

# %%


DATASET_ID,
indv_dfs,
indv,  # dict keys
# dict values (lists of lists, where sublists are song types)
repertoire,
cluster_labels,
out_dir,
shuffle_songs = True,  # Useful to avoud using very similar examples
max_seqlength = 3  # There is no warrantee that this will work with greater lenghts as is
max_n_sylls = 10
frontpad = 0.006
endpad = 0.03  # the segmentation algo shaves very closely - add some padding to each note

# Prepare paths
out_dir_notes_wav = Path(str(out_dir) + '_notes') / 'WAV'
out_dir_notes_json = Path(str(out_dir) + '_notes') / 'JSON'
safe_makedir(out_dir_notes_wav)
safe_makedir(out_dir_notes_json)

# Get list of unique files (one per song)
songs = np.unique(indv_dfs[indv].key.values.tolist())
if shuffle_songs:
    random.shuffle(songs)
songtype_counter = {}
for song in songs:
    # Get song sequence
    sequence = indv_dfs[indv][indv_dfs[indv]
                              ['key'] == song][cluster_labels].tolist()
    for songtype in repertoire:
        # Trim or extend syllable
        songtype = trim_or_extend_songtype(max_seqlength, songtype)
        typestring = ''.join(str(e) for e in songtype)

        # Check which song type is present in the sequence
        if is_a_in_x(songtype, sequence):
            # Extract a sequence (of max_seqlength length)
            indexes = find_sublist(songtype, sequence)
            for index in indexes:
                # Get syllable times
                starts = indv_dfs[indv][indv_dfs[indv]
                                        ['key'] == song].start_time.values.tolist()
                ends = indv_dfs[indv][indv_dfs[indv]
                                      ['key'] == song].end_time.values.tolist()

                # Get IOIs, silences, etc, build dictionary
                substarts = [starts[i]
                             for i in [index[0], index[0]+1, index[1]]]
                subends = [ends[i]
                           for i in [index[0], index[0]+1, index[1]]]
                subdurs = [y - x for x, y in zip(substarts, subends)]
                subIOIs = [y - x for x, y in zip(substarts, substarts[1:])]
                subsilences = [y - x for x,
                               y in zip(subends, substarts[1:])]

                # Add to songtype counter dictionary
                if typestring in songtype_counter:
                    songtype_counter[typestring] += 1
                    # Stop if max number reached
                    if songtype_counter[typestring] > max_n_sylls:
                        break
                else:
                    songtype_counter[typestring] = 0

                wav_loc = get_song_dir(DATASET_ID, song)

                # Fetch, trim, and save audio
                y, sr = librosa.load(wav_loc, sr=32000)

                for syll in range(max_seqlength):

                    out_filedir = out_dir_notes_wav / \
                        f"{indv}-{typestring}-{songtype_counter[typestring]}-{syll}.wav"

                    # Get note audio
                    ynote = y[int((substarts[syll] - frontpad) * sr)                              : int((subends[syll] + endpad) * sr)]
                    librosa.output.write_wav(out_filedir, ynote, sr, norm=True)

                    # Save dictionary
                    syllable_dict = {'bird': indv,
                                     'start_time': substarts[0], 'end_time': subends[-1],
                                     'starts': substarts, 'ends': subends,
                                     'total_duration': subends[-1] - substarts[0],
                                     'note_index': syll,
                                     'durations': subdurs, 'IOIs': subIOIs, 'silences': subsilences,
                                     'rate': sr,
                                     'position': index, 'sequence': songtype,
                                     'song_wav_loc': str(wav_loc),
                                     'syll_wav_loc': str(out_filedir)}

                    out_filejson = out_dir_notes_json / \
                        f"{indv}-{typestring}-{songtype_counter[typestring]}-{syll}.json"

                    json.dump(syllable_dict, open(out_filejson,
                                                  'w', encoding="utf8"), sort_keys=True)
