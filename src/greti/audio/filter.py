# Dereverberation using NARA-WPE

# @InProceedings{Drude2018NaraWPE,
#   Title     = {{NARA-WPE}: A Python package for weighted prediction error dereverberation in {Numpy} and {Tensorflow} for online and offline processing},
#   Author    = {Drude, Lukas and Heymann, Jahn and Boeddeker, Christoph and Haeb-Umbach, Reinhold},
#   Booktitle = {13. ITG Fachtagung Sprachkommunikation (ITG 2018)},
#   Year      = {2018},
#   Month     = {Oct},
# }

#! TODO


# The following two functions are a version of code by
# (c) 2011 James Robert, http://jiaaro.com

import numpy as np
from scipy.signal import butter, sosfilt
from src.avgn.visualization.spectrogram import visualize_spec

# import pydub


def _mk_butter_filter(freq, type, order):
    """[Defines ]
    
    Arguments:
        freq {[type]}  --   [The cutoff frequency for highpass and lowpass filters. For
                            band filters, a list of [low_cutoff, high_cutoff]]
        type {[type]}  --   ["lowpass", "highpass", or "band"]
        order {[type]} --   [nth order butterworth filter (default: 5th order). The
                            attenuation is -6dB/octave beyond the cutoff frequency (for 1st
                            order). A Higher order filter will have more attenuation, each level
                            adding an additional -6dB (so a 3rd order butterworth filter would
                            be -18dB/octave).]
    
    Returns:
        [function] -- [filters a mono audio segment]
    """

    def filter_fn(seg):

        assert seg.channels == 1

        nyq = 0.5 * seg.frame_rate
        try:
            freqs = [f / nyq for f in freq]
        except TypeError:
            freqs = freq / nyq

        sos = butter(order, freqs, btype=type, output="sos")
        y = sosfilt(sos, seg.get_array_of_samples())

        return seg._spawn(y.astype(seg.array_type))

    return filter_fn


def band_pass_filter(seg, low_cutoff_freq, high_cutoff_freq, order):
    filter_fn = _mk_butter_filter(
        [low_cutoff_freq, high_cutoff_freq], "band", order=order
    )
    return seg.apply_mono_filter_to_each_channel(filter_fn)


def dereverberate(spec, echo_range=50, echo_reduction=2, hop_length_ms=3, plot=False):
    """Function to reduce reverberation in a spectrogram. This is similar to the implementation in Luscinia
    by Robert Lachlan (https://rflachlan.github.io/Luscinia/).

    Args:
        spec (array): Spectrogram to process
        echo_range (int, optional): Time range for amplitude reduction. Defaults to 50.
        echo_reduction (int, optional): Amount of reduction. Defaults to 2.
        hop_length_ms (int, optional): Hop lenght. Defaults to 3.
        plot (bool, optional): Wether to plot a comparison with and without echo reduction. Defaults to False.

    Returns:
        array: an spectrogram with reduced reverberation
    """

    nbins = len(spec[0])

    echo_range_2 = int(echo_range / hop_length_ms)
    if echo_range_2 > nbins:
        echo_range_2 = nbins

    newspec = []

    for row in spec:

        # position = 0
        newrow = []
        for colindex, amplitude in enumerate(row):
            anterior = row[(colindex - echo_range_2) : colindex]
            if colindex < echo_range_2:
                posterior = row[colindex : (colindex + echo_range_2)]
                newrow.append(amplitude - echo_reduction * (max(posterior) - amplitude))
            elif (len(anterior) > 0) and (max(anterior) > amplitude):
                newrow.append(amplitude - echo_reduction * (max(anterior) - amplitude))
            else:
                newrow.append(amplitude)

        newspec.append(newrow)

    newspec = np.asarray(newspec)

    if plot is True:
        # newspec[newspec < 0] = 0
        # visualize_spec(spec)
        visualize_spec(newspec)

    return newspec
