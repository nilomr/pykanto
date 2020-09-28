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

from scipy.signal import butter, sosfilt

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


# convert to mono somewhere here!!!

test_audio = pydub.AudioSegment.from_wav(
    "/home/nilomr/Music/trimmed.wav"
)  # can you do without pydub?
test_audio_bandpass = band_pass_filter(test_audio, 3000, 9000, order=12)
test_audio_bandpass.export("/home/nilomr/Music/bandpasseddupdup.wav", format="wav")
