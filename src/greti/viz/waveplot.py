import pathlib2
from pathlib2 import Path
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


folderpath = DATA_DIR / "raw/2020/MP25/"
filepaths = list(folderpath.rglob("*.WAV"))


x, sr = librosa.load(filepaths[0], duration=20)
plt.figure(figsize=(15, 5))
librosa.display.waveplot(x, sr, alpha=0.8)


import wave
import numpy as np
import matplotlib.pyplot as plt

signal_wave = wave.open(str(filepaths[0]), "r")
sample_rate = 4800
sig = np.frombuffer(signal_wave.readframes(sample_rate), dtype=np.int16)
sig = sig[0:4000]

plt.figure(1)

plot_a = plt.subplot(211)
plot_a.plot(sig)
plot_a.set_xlabel("sample rate * time")
plot_a.set_ylabel("energy")

plot_b = plt.subplot(212)
plot_b.specgram(sig, NFFT=1024, Fs=sample_rate, noverlap=900)
plot_b.set_xlabel("Time")
plot_b.set_ylabel("Frequency")

plt.show()
