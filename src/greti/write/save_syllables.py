import json
import random
import numpy as np
from pathlib2 import Path
from src.avgn.utils.paths import most_recent_subdirectory
from src.greti.read.paths import DATA_DIR, safe_makedir
import librosa


def is_a_in_x(A, X):
    for i in range(len(X) - len(A) + 1):
        if A == X[i:i+len(A)]:
            return True
    return False


def find_sublist(sl, l):
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind+sll] == sl:
            results.append((ind, ind+sll-1))
    return results


def trim_or_extend_songtype(max_seqlength, songtype):
    """Repeat first note or remove last"""
    if len(songtype) < max_seqlength:
        songtype.append(songtype[0])
    elif len(songtype) > max_seqlength:
        songtype = songtype[:max_seqlength]
    return songtype


def get_song_dir(DATASET_ID, song):
    wav_loc = (
        most_recent_subdirectory(
            DATA_DIR / "processed" /
            DATASET_ID.replace("_segmented", ""),
            only_dirs=True
        ) / "WAV" / song
    )
    return wav_loc


def save_syllable_audio(DATASET_ID,
                        indv_dfs,
                        indv,  # dict keys
                        # dict values (lists of lists, where sublists are song types)
                        repertoire,
                        cluster_labels,
                        out_dir,
                        shuffle_songs=True,  # Useful to avoud using very similar examples
                        max_seqlength=3,  # There is no warrantee that this will work with greater lenghts as is
                        max_n_sylls=10
                        ):
    # Prepare paths
    out_dir_syllables_wav = Path(str(out_dir) + '_syllables') / 'WAV'
    out_dir_syllables_json = Path(str(out_dir) + '_syllables') / 'JSON'
    safe_makedir(out_dir_syllables_wav)
    safe_makedir(out_dir_syllables_json)

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
                    out_filedir = out_dir_syllables_wav / \
                        f"{indv}-{typestring}-{songtype_counter[typestring]}.wav"

                    # Fetch, trim, and save audio
                    y, sr = librosa.load(wav_loc, sr=32000)
                    y = y[int(substarts[0] * sr): int(subends[-1] * sr)]
                    librosa.output.write_wav(out_filedir, y, sr, norm=True)

                    # Save dictionary
                    syllable_dict = {'bird': indv,
                                     'start_time': substarts[0], 'end_time': subends[-1],
                                     'starts': substarts, 'ends': subends,
                                     'total_duration': subends[-1] - substarts[0],
                                     'durations': subdurs, 'IOIs': subIOIs, 'silences': subsilences,
                                     'rate': sr,
                                     'position': index, 'sequence': songtype,
                                     'song_wav_loc': str(wav_loc),
                                     'syll_wav_loc': str(out_filedir)}

                    out_filejson = out_dir_syllables_json / \
                        f"{indv}-{typestring}-{songtype_counter[typestring]}.json"

                    json.dump(syllable_dict, open(out_filejson,
                                                  'w', encoding="utf8"), sort_keys=True)


def save_note_audio(DATASET_ID,
                    indv_dfs,
                    indv,  # dict keys
                    # dict values (lists of lists, where sublists are song types)
                    repertoire,
                    cluster_labels,
                    out_dir,
                    shuffle_songs=True,  # Useful to avoud using very similar examples
                    max_seqlength=3,  # There is no warrantee that this will work with greater lenghts as is
                    max_n_sylls=10,
                    frontpad=0.006,
                    endpad=0.03  # the segmentation algo shaves very closely - add some padding to each note
                    ):
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
                        ynote = y[int(substarts[syll] * sr): int(subends[syll] * sr)]
                        librosa.output.write_wav(
                            out_filedir, ynote, sr, norm=True)

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
