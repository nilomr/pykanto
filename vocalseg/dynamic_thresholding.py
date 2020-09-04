from tqdm import tqdm
from vocalseg.utils import _normalize, spectrogram_nn, norm
import numpy as np
from scipy import ndimage
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib import gridspec
from vocalseg.utils import plot_spec


def contiguous_regions(condition):
    """
    Compute contiguous region of binary value (e.g. silence in waveform) to 
        ensure noise levels are sufficiently low
    
    Arguments:
        condition {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    idx = []
    i = 0
    while i < len(condition):
        x1 = i + condition[i:].argmax()
        try:
            x2 = x1 + condition[x1:].argmin()
        except:
            x2 = x1 + 1
        if x1 == x2:
            if condition[x1] == True:
                x2 = len(condition)
            else:
                break
        idx.append([x1, x2])
        i = x2
    return idx


def dynamic_threshold_segmentation(
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
    silence_threshold=0.05,
    min_silence_for_spec=0.1,
    max_vocal_for_spec=1.0,
    min_syllable_length_s=0.1,
    spectral_range=None,
    verbose=False,
):
    """
    computes a spectrogram from a waveform by iterating through thresholds
         to ensure a consistent noise level
    
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
        min_syllable_length_s {float} -- shortest expected length of syllable (default: {0.1})
        min_silence_for_spec {float} -- shortest expected length of silence in a song (used to set dynamic threshold) (default: {0.1})
        silence_threshold {float} -- threshold for spectrogram to consider noise as silence (default: {0.05})
        max_vocal_for_spec {float} -- longest expected vocalization in seconds  (default: {1.0})
        spectral_range {[type]} -- spectral range to care about for spectrogram (default: {None})
        verbose {bool} -- display output (default: {False})
    
    
    Returns:
        [results] -- [dictionary of results]
    """

    # does the envelope meet the standards necessary to consider this a bout
    envelope_is_good = False

    # make a copy of the hyperparameters

    # make a copy of the original spectrogram
    spec_orig = spectrogram_nn(
        vocalization,
        rate,
        n_fft=n_fft,
        hop_length_ms=hop_length_ms,
        win_length_ms=win_length_ms,
        ref_level_db=ref_level_db,
        pre=pre,
    )
    fft_rate = 1000 / hop_length_ms

    if spectral_range is not None:
        spec_bin_hz = (rate / 2) / np.shape(spec_orig)[0]
        spec_orig = spec_orig[
            int(spectral_range[0] / spec_bin_hz) : int(spectral_range[1] / spec_bin_hz),
            :,
        ]

    # loop through possible thresholding configurations starting at the highest
    for _, mldb in enumerate(
        tqdm(
            np.arange(min_level_db, min_level_db_floor, db_delta),
            leave=False,
            disable=(not verbose),
        )
    ):
        # set the minimum dB threshold
        min_level_db = mldb
        # normalize the spectrogram
        spec = norm(_normalize(spec_orig, min_level_db=min_level_db))

        # subtract the median
        spec = spec - np.median(spec, axis=1).reshape((len(spec), 1))
        spec[spec < 0] = 0

        # get the vocal envelope
        vocal_envelope = np.max(spec, axis=0) * np.sqrt(np.mean(spec, axis=0))
        # normalize envelope
        vocal_envelope = vocal_envelope / np.max(vocal_envelope)

        # Look at how much silence exists in the signal
        onsets, offsets = onsets_offsets(vocal_envelope > silence_threshold) / fft_rate
        onsets_sil, offsets_sil = (
            onsets_offsets(vocal_envelope <= silence_threshold) / fft_rate
        )

        # if there is a silence of at least min_silence_for_spec length,
        #  and a vocalization of no greater than max_vocal_for_spec length, the env is good
        if len(onsets_sil) > 0:
            # frames per second of spectrogram

            # longest silences and periods of vocalization
            max_silence_len = np.max(offsets_sil - onsets_sil)
            max_vocalization_len = np.max(offsets - onsets)
            if verbose:
                print("longest silence", max_silence_len)
                print("longest vocalization", max_vocalization_len)

            if max_silence_len > min_silence_for_spec:
                if max_vocalization_len < max_vocal_for_spec:
                    envelope_is_good = True
                    break
        if verbose:
            print("Current min_level_db: {}".format(min_level_db))

    if not envelope_is_good:
        return None

    onsets, offsets = onsets_offsets(vocal_envelope > silence_threshold) / fft_rate

    # threshold out short syllables
    length_mask = (offsets - onsets) >= min_syllable_length_s

    return {
        "spec": spec,
        "vocal_envelope": vocal_envelope.astype("float32"),
        "min_level_db": min_level_db,
        "onsets": onsets[length_mask],
        "offsets": offsets[length_mask],
    }


def onsets_offsets(signal):
    """
    [summary]
    
    Arguments:
        signal {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    elements, nelements = ndimage.label(signal)
    if nelements == 0:
        return np.array([[0], [0]])
    onsets, offsets = np.array(
        [
            np.where(elements == element)[0][np.array([0, -1])] + np.array([0, 1])
            for element in np.unique(elements)
            if element != 0
        ]
    ).T
    return np.array([onsets, offsets])


import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


def plot_segmented_spec(
    spec, onsets, offsets, hop_length_ms, background="black", figsize=(30, 5)
):
    """ plot spectrogram with colormap labels
    """
    pal = np.random.permutation(sns.color_palette("hsv", n_colors=len(onsets)))
    fft_rate = 1000 / hop_length_ms
    new_spec = np.zeros(list(np.shape(spec)) + [4])
    for onset, offset, pi in zip(onsets, offsets, pal):
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

        start_frame = int(onset * fft_rate)
        stop_frame = int(offset * fft_rate)
        new_spec[:, start_frame:stop_frame, :] = cmap(spec[:, start_frame:stop_frame])

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(background)
    ax.imshow(new_spec, interpolation=None, aspect="auto", origin="lower")


def plot_segmentations(
    spec, vocal_envelope, onsets, offsets, hop_length_ms, rate, figsize=(30, 5)
):
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])
    gs.update(hspace=0.0)  # set the spacing between axes.
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    plot_spec(spec, fig, ax1, rate=rate, hop_len_ms=hop_length_ms, show_cbar=False)
    ax0.plot(vocal_envelope, color="k")
    ax0.set_xlim([0, len(vocal_envelope)])
    ax1.xaxis.tick_bottom()
    ylmin, ylmax = ax1.get_ylim()
    ysize = (ylmax - ylmin) * 0.1
    ymin = ylmax - ysize

    patches = []
    for onset, offset in zip(onsets, offsets):
        ax1.axvline(onset, color="#FFFFFF", ls="dashed", lw=0.75)
        ax1.axvline(offset, color="#FFFFFF", ls="dashed", lw=0.75)
        patches.append(Rectangle(xy=(onset, ymin), width=offset - onset, height=ysize))

    collection = PatchCollection(patches, color="white", alpha=0.5)
    ax1.add_collection(collection)
    ax0.axis("off")
    return fig


#


# ### segment and plot
# - for each json, load the wav file - segment the file into start and end times
# - plot the segmentation
# - add to the JSON

import json
import librosa
from avgn.signalprocessing.filtering import butter_bandpass_filter
from avgn.utils.audio import load_wav, read_wav
from avgn.utils.json import NoIndent, NoIndentEncoder
from avgn.utils.paths import most_recent_subdirectory, ensure_dir
from src.read.paths import DATA_DIR
from datetime import datetime

DT_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DATASET_ID = "GRETI_HQ"


def segment_spec_custom(
    key,
    df,
    n_fft=1024,
    hop_length_ms=3,
    win_length_ms=15,
    ref_level_db=30,
    pre=0.5,
    min_level_db=-30,
    min_level_db_floor=-10,
    db_delta=7,
    silence_threshold=0.3,
    min_silence_for_spec=0.001,
    max_vocal_for_spec=(0.4,),
    min_syllable_length_s=0.03,
    spectral_range=[1200, 9000],
    save=False,
    plot=False,
    DATA_DIR=DATA_DIR,
    DT_ID=DT_ID,
    DATASET_ID=DATASET_ID,
):

    # load wav
    rate, data = load_wav(df.data["wav_loc"])

    # bandpass based on manual selections in AviaNZ
    butter_min = df.data["lower_freq"]
    butter_max = df.data["upper_freq"]

    data = butter_bandpass_filter(data, butter_min, butter_max, rate)
    data = librosa.util.normalize(data)

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
