from datetime import datetime

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import get_ipython
from joblib import Parallel, delayed
from tqdm.autonotebook import tqdm

from src.avgn.dataset import DataSet
from src.avgn.signalprocessing.create_spectrogram_dataset import *
from src.avgn.utils.hparams import HParams
from src.avgn.utils.paths import ensure_dir, most_recent_subdirectory
from src.avgn.visualization.spectrogram import draw_spec_set
from src.greti.read.paths import *

DATASET_ID = "GRETI_HQ_2020_segmented"


# Create data
N = 500
x = np.random.rand(N)
y = np.random.rand(N)
colors = (0, 0, 0)
area = np.pi * 3

# Plot
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.title("Pene")
plt.xlabel("x")
plt.ylabel("y")

fig_out = FIGURE_DIR / (
    "{}_scatter".format("caca")
    + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    + ".png"
)

ensure_dir(fig_out)
plt.savefig(
    fig_out, dpi=300, bbox_inches="tight", pad_inches=0.3, transparent=False,
)
plt.close()
