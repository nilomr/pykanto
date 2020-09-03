from tqdm.autonotebook import tqdm
from vocalseg.utils import _normalize, spectrogram, norm, plot_spec
from vocalseg.dynamic_thresholding import dynamic_threshold_segmentation
import numpy as np
from scipy import ndimage, signal
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

cmap = matplotlib.colors.ListedColormap(np.random.rand(256, 3))
cmap.set_bad(color=(0, 0, 0, 0))


def continuity_segmentation(
    vocalization,
    rate,
    min_level_db=-80,
    min_level_db_floor=-40,
    db_delta=5,
    n_fft=1024,
    hop_length_ms=1,
    win_length_ms=5,
    ref_level_db=20,
    pre=0.97,
    spectral_range=None,
    verbose=False,
    mask_thresh_std=1,
    neighborhood_time_ms=5,
    neighborhood_freq_hz=500,
    neighborhood_thresh=0.5,
    min_syllable_length_s=0.1,
    min_silence_for_spec=0.1,
    silence_threshold=0.05,
    max_vocal_for_spec=1.0,
    temporal_neighbor_merge_distance_ms=0.0,
    overlapping_element_merge_thresh=np.inf,
    min_element_size_ms_hz=[0, 0],  # ms, hz
    figsize=(20, 5),
):
    """
    segments song into continuous elements

    Arguments:
        vocalization {[type]} -- waveform of song
        rate {[type]} -- samplerate of datas

    Keyword Arguments:
        min_level_db {int} -- default dB minimum of spectrogram (threshold anything below) (default: {-80})
        min_level_db_floor {int} -- highest number min_level_db is allowed to reach dynamically (default: {-40})
        db_delta {int} -- delta in setting min_level_db (default: {5})
        n_fft {int} -- FFT window size (default: {1024})
        hop_length_ms {int} -- number audio of frames in ms between STFT columns (default: {1})
        win_length_ms {int} -- size of fft window (ms) (default: {5})
        ref_level_db {int} -- reference level dB of audio (default: {20})
        pre {float} -- coefficient for preemphasis filter (default: {0.97})
        spectral_range {[type]} -- spectral range to care about for spectrogram (default: {None})
        verbose {bool} -- display output (default: {False})
        mask_thresh_std {int} -- standard deviations above median to threshold out noise (higher = threshold more noise) (default: {1})
        neighborhood_time_ms {int} -- size in time of neighborhood-continuity filter (default: {5})
        neighborhood_freq_hz {int} -- size in Hz of neighborhood-continuity filter (default: {500})
        neighborhood_thresh {float} -- threshold number of neighborhood time-frequency bins above 0 to consider a bin not noise (default: {0.5})
        min_syllable_length_s {float} -- shortest expected length of syllable (default: {0.1})
        min_silence_for_spec {float} -- shortest expected length of silence in a song (used to set dynamic threshold) (default: {0.1})
        silence_threshold {float} -- threshold for spectrogram to consider noise as silence (default: {0.05})
        max_vocal_for_spec {float} -- longest expected vocalization in seconds  (default: {1.0})
        temporal_neighbor_merge_distance_ms {float} -- longest distance at which two elements should be considered one (default: {0.0})
        overlapping_element_merge_thresh {float} -- proportion of temporal overlap to consider two elements one (default: {np.inf})
        min_element_size_ms_hz {list} --  smallest expected element size (in ms and HZ). Everything smaller is removed. (default: {[0, 0]})
        figsize {tuple} -- size of figure for displaying output (default: {(20, 5)})

    Returns:
        results -- a dictionary with results of segmentation
    """

    def plot_interim(spec, cmap=plt.cm.afmhot, zero_nan=False):
        fig, ax = plt.subplots(figsize=figsize)
        if zero_nan:
            spec = spec.copy()
            spec[spec == 0] = np.nan
        plot_spec(
            spec,
            fig=fig,
            ax=ax,
            rate=rate,
            hop_len_ms=hop_length_ms,
            show_cbar=False,
            cmap=cmap,
        )
        plt.show()

    results = dynamic_threshold_segmentation(
        vocalization,
        rate,
        n_fft=n_fft,
        hop_length_ms=hop_length_ms,
        win_length_ms=win_length_ms,
        ref_level_db=ref_level_db,
        pre=pre,
        min_level_db=min_level_db,
        db_delta=db_delta,
        silence_threshold=silence_threshold,
        verbose=verbose,
        spectral_range=spectral_range,
        min_syllable_length_s=min_syllable_length_s,
        min_silence_for_spec=min_silence_for_spec,
        max_vocal_for_spec=max_vocal_for_spec,
    )
    if results is None:
        return None

    spec = results["spec"]

    # bin width in Hz
    if spectral_range is None:
        spec_bin_hz = (rate / 2) / np.shape(spec)[0]
    else:
        spec_bin_hz = (spectral_range[1] - spectral_range[0]) / np.shape(spec)[0]

    if verbose:
        plot_interim(spec, cmap=plt.cm.Greys)

    ### create a mask
    mask = mask_spectrogram(spec, mask_thresh_std)

    if verbose:
        plot_interim(mask)

    # Create a smoothing filter for the mask in time and frequency
    continuity_filter = make_continuity_filter(
        neighborhood_freq_hz, neighborhood_time_ms, spec_bin_hz, hop_length_ms
    )
    print(np.shape(continuity_filter))
    ### remove non-continuous regions of the mask
    # apply filter
    mask = signal.fftconvolve(
        (1 - mask.astype("float32")), continuity_filter, mode="same"
    )
    # threshold filter
    mask = mask < neighborhood_thresh

    if verbose:
        plot_interim(mask)

    # find continous elements
    elements = segment_mask(mask)

    if verbose:
        plot_interim(elements, cmap=cmap, zero_nan=True)

    # get element timing
    unique_elements, syllable_start_times, syllable_end_times = get_syllable_timing(
        elements, hop_length_ms
    )
    print("unique elements: {}".format(len(unique_elements)))
    # merge elements that are nearby to each other
    if temporal_neighbor_merge_distance_ms > 0:
        elements = merge_temporal_neighbors(
            elements,
            unique_elements,
            syllable_start_times,
            syllable_end_times,
            temporal_neighbor_merge_distance_ms,
        )

        if verbose:
            plot_interim(elements, cmap=cmap, zero_nan=True)
            unique_elements = np.unique(elements[elements != 0].astype(int))
            print("unique elements: {}".format(len(unique_elements)))

    # no reason to merge overlapping if already merging neighbords
    elif overlapping_element_merge_thresh <= 1.0:
        # merge elements that are overlapping in time by some amount
        elements = merge_overlapping_elements(
            elements,
            unique_elements,
            syllable_start_times,
            syllable_end_times,
            overlapping_element_merge_thresh,
        )
        if verbose:
            plot_interim(elements, cmap=cmap, zero_nan=True)
            unique_elements = np.unique(elements[elements != 0].astype(int))
            print("unique elements: {}".format(len(unique_elements)))

    # remove elements that are
    if np.product(min_element_size_ms_hz) > 0:
        min_element_size = int(
            np.product(
                (
                    min_element_size_ms_hz[0] / hop_length_ms,
                    min_element_size_ms_hz[1] / spec_bin_hz,
                )
            )
        )
        if min_element_size > 0:
            elements = remove_small_elements(elements, min_element_size)

    # randomize label values since they are temporally/frequency continuous
    elements = randomize_labels(elements)
    if verbose:
        plot_interim(elements, cmap=cmap, zero_nan=True)
        unique_elements = np.unique(elements[elements != 0].astype(int))
        print("unique elements: {}".format(len(unique_elements)))

    results["elements"] = elements

    return results


def remove_small_elements(elements, min_element_size):
    """ remove elements that are below some threshold size
    """
    # get unique points
    unique_elements = np.unique(elements[elements != 0].astype(int))

    print(min_element_size)
    for element in unique_elements:
        # if the size of the cluster is smaller than the minimum, remove it
        if np.sum(elements == element) < min_element_size:
            elements[elements == element] = 0

    return elements


def merge_temporal_neighbors(
    elements,
    unique_elements,
    syllable_start_times,
    syllable_end_times,
    temporal_neighbor_merge_distance_ms,
):
    """
    merge elements that are within temporal_neighbor_merge_distance_ms
     ms of each other
    
    Arguments:
        elements {[type]} -- [description]
        unique_elements {[type]} -- [description]
        syllable_start_times {[type]} -- [description]
        syllable_end_times {[type]} -- [description]
        temporal_neighbor_merge_distance_ms {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    merged_elements = {}
    for element, st, et in tqdm(
        zip(unique_elements, syllable_start_times, syllable_end_times),
        total=len(unique_elements),
        desc="merging temporal neighbors",
        leave=False,
    ):
        # if this element has already been merged, ignore it
        if element in merged_elements.keys():
            element = merged_elements[element]
        # get elements that start between the beginning of this element and the
        #    end of this element plus temporal_neighbor_merge_distance_ms
        overlapping_syllables = np.where(
            (syllable_start_times > st)
            & (syllable_start_times < et + (temporal_neighbor_merge_distance_ms))
        )[0]
        # print(overlapping_syllables)
        if len(overlapping_syllables) > 0:
            for overlapping_syllable in overlapping_syllables:
                syll_name = unique_elements[overlapping_syllable]
                merged_elements[syll_name] = element
                elements[elements == syll_name] = element
            # remove from lists
            unique_elements = np.delete(unique_elements, overlapping_syllables)
            syllable_start_times = np.delete(
                syllable_start_times, overlapping_syllables
            )
            syllable_end_times = np.delete(syllable_end_times, overlapping_syllables)

    return elements


def merge_overlapping_elements(
    elements,
    unique_elements,
    syllable_start_times,
    syllable_end_times,
    overlapping_element_merge_thresh,
):
    """
    merge elements that are overlapping by at least overlapping_element_merge_thresh
    
    Arguments:
        elements {[type]} -- [description]
        unique_elements {[type]} -- [description]
        syllable_start_times {[type]} -- [description]
        syllable_end_times {[type]} -- [description]
        overlapping_element_merge_thresh {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    # sort syllables by length
    sort_mask = np.argsort(syllable_end_times - syllable_start_times)
    syllable_end_times = syllable_end_times[sort_mask]
    syllable_start_times = syllable_start_times[sort_mask]
    unique_elements = unique_elements[sort_mask]

    # loop through elements
    for element, st, et in tqdm(
        zip(unique_elements, syllable_start_times, syllable_end_times),
        total=len(unique_elements),
        desc="merging temporally overlapping elements",
        leave=False,
    ):
        # elements have to be overlapped at least this length to merge
        overlap_thresh = (et - st) * overlapping_element_merge_thresh

        # get elements that
        # # c1: start befre et - overlap_thresh and end after et,
        #   c2: start before st and end after st + overlap_thresh
        #   c3: or start after st and before et and are longer than overlap_thresh
        #   c4: fully overlap syllable
        c1 = (syllable_start_times < (et - overlap_thresh)) & (syllable_end_times > et)
        c2 = (syllable_start_times < (st)) & (
            syllable_end_times > (st + overlap_thresh)
        )
        c3 = ((syllable_start_times > (st)) & (syllable_end_times < et)) & (
            (syllable_end_times - syllable_start_times) > overlap_thresh
        )
        c4 = (syllable_start_times < st) & (syllable_end_times > et)

        # get list of overlapping elements
        overlapping_syllables = np.where(c1 | c2 | c3 | c4)[0]

        # print(overlapping_syllables)
        if len(overlapping_syllables) > 0:
            # get the longest syllable
            overlapping_syllable = overlapping_syllables[-1]
            syll_name = unique_elements[overlapping_syllable]

            # change all elements to that element
            elements[elements == element] = syll_name
            # remove from lists
            el = np.where(unique_elements == element)[-1]
            unique_elements = np.delete(unique_elements, el)
            syllable_start_times = np.delete(syllable_start_times, el)
            syllable_end_times = np.delete(syllable_end_times, el)

    return elements


def randomize_labels(elements):
    unique_elements = np.unique(elements[elements != 0].astype(int))
    perm = np.random.permutation(unique_elements)
    el_dict = {i: j for i, j in zip(unique_elements, perm)}
    for el, val in el_dict.items():
        elements[elements == el] = val
    return elements


def mask_spectrogram(spec, mask_thresh_std):
    """
    masks low power noise in a spectrogram
    
    Arguments:
        spec {[type]} -- [description]
        mask_thresh_std {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    return (
        (
            spec.T
            < (np.median(spec, axis=1) + mask_thresh_std * np.std(spec, axis=1)) + 1e-5
        )
        .astype("float32")
        .T
    )


def make_continuity_filter(
    neighborhood_freq_hz, neighborhood_time_ms, spec_bin_hz, hop_length_ms
):
    """
     Generate a filter for continuous elements
    
    Arguments:
        neighborhood_freq_hz {[type]} -- [description]
        neighborhood_time_ms {[type]} -- [description]
        spec_bin_hz {[type]} -- [description]
        hop_length_ms {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    n_bin_freq = int(neighborhood_freq_hz / spec_bin_hz)
    n_bin_time = int(neighborhood_time_ms / hop_length_ms)
    return np.ones((n_bin_freq, n_bin_time)) / np.product((n_bin_freq, n_bin_time))


def segment_mask(mask):
    """
    segments a binary spectrogram mask into individual elements
    
    Arguments:
        mask {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    elements, _ = ndimage.label(mask == False)
    elements = np.ma.masked_where(elements == 0, elements)
    elements = np.array(elements.data).astype("float32")
    return elements


def get_syllable_timing(elements, hop_length_ms):
    """
    gets length of elements of each mask type
    
    Arguments:
        elements {[type]} -- [description]
        hop_length_ms {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    # get unique points
    unique_elements = np.unique(elements[elements != 0].astype(int))

    # get the time coverage of each element
    total_coverage = [
        np.sum(elements == i, axis=0)
        for i in tqdm(unique_elements, desc="element coverage", leave=False)
    ]

    # get the start and end times of each syllable
    syllable_start_times, syllable_end_times = np.array(
        [
            np.where(i > 0)[0][np.array([0, -1])] + np.array([0.0, 1.0])
            for i in tqdm(total_coverage, desc="element length", leave=False)
        ]
    ).T * float(hop_length_ms)

    sort_mask = np.argsort(syllable_start_times)
    syllable_start_times = syllable_start_times[sort_mask]
    syllable_end_times = syllable_end_times[sort_mask]
    unique_elements = unique_elements[sort_mask]

    return unique_elements, syllable_start_times, syllable_end_times


def plot_labelled_elements(elements, spec, background="white", figsize=(30, 5)):
    """ plots a spectrogram with colormap labels
    """
    unique_elements = np.unique(elements[elements != 0].astype(int))
    pal = np.random.permutation(
        sns.color_palette("rainbow", n_colors=len(unique_elements))
    )

    new_spec = np.zeros(list(np.shape(elements)) + [4])
    # fill spectrogram with colored regions
    for el, pi in tqdm(
        zip(unique_elements, pal), total=len(unique_elements), leave=False
    ):

        if background == "black":

            cdict = {
                "red": [(0, pi[0], pi[0]), (1, 1, 1)],
                "green": [(0, pi[1], pi[1]), (1, 1, 1)],
                "blue": [(0, pi[2], pi[2]), (1, 1, 1)],
                "alpha": [(0, 0, 0), (0.25, 0.5, 0.5), (1, 1, 1)],
            }
        else:
            cdict = {
                "red": [(0, pi[0], pi[0]), (1, 0, 0)],
                "green": [(0, pi[1], pi[1]), (1, 0, 0)],
                "blue": [(0, pi[2], pi[2]), (1, 0, 0)],
                "alpha": [(0, 0, 0), (1, 1, 1)],
            }
        cmap = LinearSegmentedColormap("CustomMap", cdict)

        new_spec[elements == el] = cmap(spec[elements == el])

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(background)
    ax.imshow(new_spec, interpolation=None, aspect="auto", origin="lower")

    return new_spec
