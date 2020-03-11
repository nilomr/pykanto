# -*- coding: utf-8 -*-

# # Source of lines 26-62:
# https://github.com/jiaaro/pydub/blob/0908a38fad57cba3a2e7eb4934c1d542ab56d7c5/pydub/scipy_effects.py
# Copyright (c) 2011 James Robert, http://jiaaro.com

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from scipy.signal import butter, sosfilt
import pydub 

def _mk_butter_filter(freq, type, order):
    """
    Args:
        freq: The cutoff frequency for highpass and lowpass filters. For
            band filters, a list of [low_cutoff, high_cutoff]
        type: "lowpass", "highpass", or "band"
        order: nth order butterworth filter (default: 5th order). The
            attenuation is -6dB/octave beyond the cutoff frequency (for 1st
            order). A Higher order filter will have more attenuation, each level
            adding an additional -6dB (so a 3rd order butterworth filter would
            be -18dB/octave).
    Returns:
        function which can filter a mono audio segment
    """
    def filter_fn(seg):
        assert seg.channels == 1

        nyq = 0.5 * seg.frame_rate
        try:
            freqs = [f / nyq for f in freq]
        except TypeError:
            freqs = freq / nyq

        sos = butter(order, freqs, btype=type, output='sos')
        y = sosfilt(sos, seg.get_array_of_samples())

        return seg._spawn(y.astype(seg.array_type))

    return filter_fn


def band_pass_filter(seg, low_cutoff_freq, high_cutoff_freq, order=5):
    filter_fn = _mk_butter_filter([low_cutoff_freq, high_cutoff_freq], 'band', order=order)
    return seg.apply_mono_filter_to_each_channel(filter_fn)


test_audio = pydub.AudioSegment.from_wav("/home/nilomr/Music/trimmed.wav")
test_audio_bandpass = band_pass_filter(test_audio, 2500, 9000, order = 12)
test_audio_bandpass.export("/home/nilomr/Music/bandpassed.wav", format = "wav")


# To split files:
# https://stackoverflow.com/questions/51060460/split-wav-file-in-python-with-different-start-and-end-duration
# https://stackoverflow.com/questions/55669405/how-would-i-be-able-to-split-an-audio-file-into-multiple-wav-files-using-timpooi
# https://unix.stackexchange.com/questions/420820/split-audio-into-several-pieces-based-on-timestamps-from-a-text-file-with-sox-or

# Split on silence:
# https://stackoverflow.com/questions/23730796/using-pydub-to-chop-up-a-long-audio-file
