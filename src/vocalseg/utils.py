import warnings

import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter

### General


def int16tofloat32(data):
    return np.array(data / 32768).astype("float32")


def norm(x, _type="zero_one"):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


### Filtering


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    if highcut > int(fs / 2):
        warnings.warn("Highcut is too high for bandpass filter. Setting to nyquist")
        highcut = int(fs / 2)
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


### Spectrogramming


def spectrogram(
    y,
    fs,
    n_fft=1024,
    hop_length_ms=1,
    win_length_ms=5,
    ref_level_db=20,
    pre=0.97,
    min_level_db=-50,
):
    return _normalize(
        spectrogram_nn(
            y,
            fs,
            n_fft=n_fft,
            hop_length_ms=hop_length_ms,
            win_length_ms=win_length_ms,
            ref_level_db=ref_level_db,
            pre=pre,
        ),
        min_level_db=min_level_db,
    )


def spectrogram_nn(y, fs, n_fft, hop_length_ms, win_length_ms, ref_level_db, pre):
    D = _stft(preemphasis(y, pre), fs, n_fft, hop_length_ms, win_length_ms)
    S = _amp_to_db(np.abs(D)) - ref_level_db
    return S


def preemphasis(x, pre):
    return signal.lfilter([1, -pre], [1], x)


def _stft(y, fs, n_fft, hop_length_ms, win_length_ms):
    return librosa.stft(
        y=y,
        n_fft=n_fft,
        hop_length=int(hop_length_ms / 1000 * fs),
        win_length=int(win_length_ms / 1000 * fs),
    )


def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def _normalize(S, min_level_db):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)


### viz

import matplotlib.pyplot as plt


def plot_spec(
    spec,
    fig=None,
    ax=None,
    rate=None,
    hop_len_ms=None,
    cmap=plt.cm.gray,
    show_cbar=True,
    spectral_range=None,
    time_range=None,
    figsize=(20, 6),
):
    """plot spectrogram
    
    [description]
    
    Arguments:
        spec {[type]} -- [description]
        fig {[type]} -- [description]
        ax {[type]} -- [description]
    
    Keyword Arguments:
        cmap {[type]} -- [description] (default: {plt.cm.afmhot})
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    extent = [0, np.shape(spec)[1], 0, np.shape(spec)[0]]
    if rate is not None:
        extent[3] = rate / 2
    if hop_len_ms is not None:
        extent[1] = (np.shape(spec)[1] * hop_len_ms) / 1000
    if spectral_range is not None:
        extent[2] = spectral_range[0]
        extent[3] = spectral_range[1]
    if time_range is not None:
        extent[0] = time_range[0]
        extent[1] = time_range[1]

    spec_ax = ax.matshow(
        spec,
        interpolation=None,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        extent=extent,
    )
    if show_cbar:
        cbar = fig.colorbar(spec_ax, ax=ax)
        return spec_ax, cbar
    else:
        return spec_ax
