import collections
from src.avgn.utils.general import flatten
import numpy as np
import ast
import string
import re
from collections.abc import Iterable


def list_sequences(indv_dfs, indv, label):
    """Returns a lists of lists where each list contains the sequence of notes in one song
    """
    labs = indv_dfs[indv][label].values
    sequence_ids = np.array(indv_dfs[indv]["syllables_sequence_id"])
    sequences = [labs[sequence_ids == i] for i in np.unique(sequence_ids)]
    # Substitute labels
    sequences_newlabels = [list(seq) for seq in sequences]

    return sequences_newlabels, labs


def collapse_palindromic_keys(tally, seq_str):
    # Take the one that appears first in sequence most often
    remove = []
    for k, v in tally.items():
        if (k and k[::-1] in tally.keys()) and (k != k[::-1]):
            how_many_first = {pattrn: sum(
                [seq.startswith(pattrn) for seq in seq_str]) for pattrn in [k, k[::-1]]}
            minkey = min(how_many_first, key=lambda key: how_many_first[key])
            remove.append(minkey)

    no_palindromes = dict((k, tally[k])
                          for k in tally.keys() if k not in set(remove))
    return no_palindromes


def remove_long_ngrams(minus_palindromes):
    # Remove collapsible sequences
    remove = [
        seq2
        for seq2 in minus_palindromes
        for seq in minus_palindromes
        if seq != seq2
        and seq in seq2
        and not [
            substr2 for substr2 in seq2 if substr2 not in seq
        ]  # contains no new notes
    ]
    return [seq for seq in minus_palindromes if seq not in remove]


def dict_keys_to_symbol(labs):
    # Change keys from int to symbol: this is necessary to be able to search for patterns where int labels >9
    unique_labs = np.unique(labs)
    nlabs = len(unique_labs)
    sym_list = list(string.ascii_uppercase[0:nlabs])
    symbolic_dict = {
        lab: (symbol if lab != -1 else 'Z')
        for lab, symbol in zip(unique_labs, sym_list)
    }
    return symbolic_dict


def duplicate_1grams(result):
    # Duplicate 1-grams
    result = [seq * 2 if len(seq) == 1 else seq for seq in set(result)]
    return result


def get_seq_frequencies(result, seq_str, by_song=True):
    """Returns a sorted dictionary of sequence frequencies in the entire 
    output of an individual indv, by number of songs in which it appears 
    or by total occurrences
    Args:
        result (dict): A dictionary with repeated sequencies for each song
    Returns:
        dict: Sorted dictionary
    """
    if by_song:
        tally = {seq: sum(value == seq for value in result.values())
                 for k, seq in result.items()}
    else:
        unique_seqs = set([seq for seq in list(result.values())])
        tally = {}
        for seq in unique_seqs:
            n = 0
            for element in seq_str:
                n += element.count(seq)
            tally[seq] = n
        tally = dict(sorted(tally.items(), key=lambda x: x[1], reverse=True))
    return tally


def get_mostfreq_pattern(seq_str_dict):
    # Find repeating motifs; get most frequent in each song
    pattern = re.compile(r"(.+?)(?=\1)")
    end_dict = {}
    for key, seq in seq_str_dict.items():
        subseqs = pattern.findall(seq)
        subseqs2 = duplicate_1grams(subseqs)
        if len(subseqs2) == 0:
            continue
        persong_count = {s: seq.count(s) for s in subseqs2}
        # if one_per_song: #TODO: add option to return > 1 seqs per song
        end_dict[key] = max(persong_count, key=persong_count.get)
    return end_dict


def dict_keys_to_int(counts, sym_dict):
    # Change keys back to int matching use in syllable_df
    sym_dict_r = {v: k for k, v in sym_dict.items()}
    return {
        str([sym_dict_r[element] for element in key]): value
        for key, value in counts.items()
    }


def collapse_slidable_seqs(tally, seq_str):
    """
    Removes sequences that are contained in other sequences if repeated, 
    keeping those that serve as the start of a song most frerquently.
    """
    conflict = []
    for k in tally.keys():
        reference = k*4
        for k2 in tally.keys():
            if k2 != k and k2 in reference:
                conflict.append([k, k2])

    def to_remove(seq_str, conflictlist):
        """Which sequences to remove based on frequency as song start.
        """
        outer_remove = []
        for item in conflictlist:
            how_many_first = {pattrn: sum(
                [seq.startswith(pattrn) for seq in seq_str]) for pattrn in item}
            maxkey = max(how_many_first, key=lambda key: how_many_first[key])
            inner_remove = [key for key in item if key != maxkey]
            outer_remove.append(inner_remove[0])
        return outer_remove

    if len(conflict) == 0:
        return tally
    elif len(conflict) == 1:
        outer_remove = to_remove(seq_str, conflict)
    else:
        newconflict = []
        for item in conflict:
            for item2 in conflict:
                if item == item2[::-1] and item[::-1] not in newconflict and item2[::-1] not in newconflict:
                    newconflict.append(item)
        outer_remove = to_remove(seq_str, newconflict)

    no_sliders = dict((k, tally[k])
                      for k in tally.keys() if k not in outer_remove)
    return no_sliders


def get_missegment_index(indv_dfs, cluster_labels, note_label, indv, file_key, threshold=0.5):
    """
    Threshold as ratio between note and interval duration, eg: 0.5 means that sequences 
    where all notes are the same and the inter note interval is less than half the duration of a note are removed)
    """
    df = indv_dfs[indv][(indv_dfs[indv].key == file_key) & (
        indv_dfs[indv][cluster_labels] == note_label)]

    starts = df.start_time
    ends = df.end_time
    duration = ends - starts
    interval = [t1 - t2 for t1, t2 in zip(starts[1:], ends)]

    try:
        short_seqs = [i for i, (inter, dur) in enumerate(
            zip(interval, duration)) if inter < dur*threshold]
        if len(short_seqs) < 1:
            short_seqs = False
    except:
        short_seqs = False

    return short_seqs, duration, interval


def remove_bad_syllables(indv_dfs, indv, cluster_labels, final_dict, threshold=0.4):
    """Remove same-type notes that are adjacent and separated by an interval 
    shorter than a fraction (threshold) of the note duration.
    See `get_missegment_index()`.
    """
    try:
        note_label = [ast.literal_eval(key)[0] for key in final_dict.keys() if len(
            set(ast.literal_eval(key))) == 1]
    except:
        note_label = False
        print('No same-same sequences to remove')

    if note_label is not False or note_label == 0:
        for label in note_label if isinstance(note_label, Iterable) else [note_label]:
            index_list = [item for sublist in list(
                final_dict.values()) for item in sublist]

            for song in index_list:
                file_key = indv_dfs[indv][indv_dfs[indv]
                                          .syllables_sequence_id == song].key[0]
                missegment, _, _ = get_missegment_index(
                    indv_dfs, cluster_labels,  label, indv, file_key, threshold=threshold)

                if missegment:
                    final_dict = {
                        k: v for k, v in final_dict.items() if k != f'[{label}, {label}]'}
                    break

    return final_dict


def remove_repeat_songs(final_dict):
    """Removes songs that are called more than once, keeping them under the sequence type with most songs. 
    WARNING: This WILL give priority to sequences contained in longer 
    sequences (e.g., 15 > 215): always remove the former before running.

    Args:
        final_dict (dict): A dictionary, {sequence : [syllables_sequence_id]}
    Returns:
        dict : Same dict minus relevant entries
        complete (bool): Whether the process is done (for use in loop)
    """
    nseqs = list(flatten(final_dict.values()))
    duplicates = [k for k, v in collections.Counter(nseqs).items() if v > 1]

    keep = []
    repeats = []
    for k, v in final_dict.items():
        if any(i in v for i in duplicates):
            for k1, v1 in final_dict.items():
                if any(i in v for i in v1) and k != k1:
                    repeats.append(k)
                    if len(v) > len(v1):
                        keep.append(k)
                    else:
                        keep.append(k1)
    remove = [i for i in repeats if i not in keep]
    final_dict_2 = {k: v for k, v in final_dict.items() if k not in remove}
    complete = True

    # Check if any still repeated
    nseqs = list(flatten(final_dict_2.values()))
    duplicates = [k for k, v in collections.Counter(nseqs).items() if v > 1]
    for k, v in final_dict_2.items():
        if any(i in v for i in duplicates):
            complete = False
    return final_dict_2, complete


def find_syllable_sequences(
    indv_dfs,
    indv,
    cluster_labels,
    remove_noise=True,
    remove_redundant=True,
    collapse_palindromes=True,
    collapse_subsequences=True,
    remove_double_count_notes=True,
    remove_double_count_songs=True,
    use_n_songs=True,
    min_freq=1,
    min_songs=1,
    double_note_threshold=0.5  # half of the note duration
):
    sequences_newlabels, labs = list_sequences(
        indv_dfs, indv, cluster_labels)
    sym_dict = dict_keys_to_symbol(labs)
    sequences = [[sym_dict[b] for b in i] for i in sequences_newlabels]

    # Find n-grams
    # Convert lists to strings
    seq_str = ["".join(map(str, seq)) for seq in sequences]
    seq_str_dict = {key: seq for key, seq in enumerate(seq_str)}

    # Find repeating motifs; get most frequent in each song
    result = get_mostfreq_pattern(seq_str_dict)

    # Frequencies for each combination
    if use_n_songs:  # Whether to count total occurrences or total number of songs where it appears
        tally = get_seq_frequencies(result, seq_str)
    else:
        tally = get_seq_frequencies(result, seq_str, by_song=False)

    if remove_noise:
        # Remove sequences with noise (-1 labels)
        tally = {key: n for key, n in tally.items() if 'Z' not in key}

    if remove_redundant:
        # Remove sequences already contained in other, shorter sequences
        # (where the longest of the pair does not have new notes)
        tally = {k: v for k, v in tally.items() if k in remove_long_ngrams([
            key for key in tally.keys()])}

    # Remove absolute infrequent combinations (can help get rid of noise)
    tally = {k: v for k, v in tally.items() if v >= min_freq}

    # if collapse_palindromes:  # Take the one that appears first in sequence most often
    #     tally = collapse_palindromic_keys(tally, seq_str)
    # This is made redundant by the more general collapse_slidable_seqs()

    if collapse_subsequences:
        tally = collapse_slidable_seqs(tally, seq_str)

    # Build dictionary of songs containing each sequence (allows duplicates)
    song_dict = {}
    for sequence, n in tally.items():
        seqlist = []
        for key, songseq in seq_str_dict.items():
            if sequence in songseq:
                seqlist.append(key)
        song_dict[sequence] = seqlist

    # Remove sequences if present in fewer than min_songs
    song_dict = {k: v for k, v in song_dict.items() if len(v) >= min_songs}

    # Symbols to original labels
    final_dict = dict_keys_to_int(song_dict, sym_dict)

    # Remove double-counted notes?
    if remove_double_count_notes:
        final_dict = remove_bad_syllables(
            indv_dfs, indv, cluster_labels, final_dict, threshold=double_note_threshold)

    # Remove double-counted songs? #!EXPERIMENTAL
    if remove_double_count_songs:
        for _ in range(10):
            final_dict, state = remove_repeat_songs(final_dict)
            if state:
                break

    return final_dict, len(sequences_newlabels), seq_str, tally
