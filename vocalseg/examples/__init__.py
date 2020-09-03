from scipy.io import wavfile
import os

FP = os.path.dirname(os.path.abspath(__file__))


def starling():
    rate, data = wavfile.read(os.path.join(FP, "starling.wav"))
    return rate, data


def mouse():
    rate, data = wavfile.read(os.path.join(FP, "mouse_usv.wav"))
    return rate, data


def canary():
    rate, data = wavfile.read(os.path.join(FP, "canary.wav"))
    return rate, data


def bengalese_finch():
    rate, data = wavfile.read(os.path.join(FP, "bengalese_finch.wav"))
    return rate, data
