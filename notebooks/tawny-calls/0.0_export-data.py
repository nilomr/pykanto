# %%

import math
import statistics

import matplotlib.pyplot as plt
import pandas as pd
import pyproj
from joblib import Parallel, delayed
from mizani.breaks import date_breaks
from mizani.formatters import date_format
from plotnine import *
from tqdm.autonotebook import tqdm

from src.avgn.dataset import DataSet
from src.avgn.signalprocessing.create_spectrogram_dataset import create_label_df
from src.avgn.utils.hparams import HParams
from src.avgn.utils.paths import ensure_dir
from src.greti.read.paths import DATA_DIR, FIGURE_DIR, RESOURCES_DIR

# %%
DATASET_ID = "TAWNY_2020"
year = "2020"

# %% [markdown]
# ### create dataset
# %%

20 * math.log10(5000 / 32767)

# %%

hparams = HParams(
    n_fft=1024,
    win_length_ms=15,
    hop_length_ms=3,
    mel_lower_edge_hertz=1200,
    mel_upper_edge_hertz=15000,
    butter_lowcut=1200,
    butter_highcut=15000,
    ref_level_db=30,
    min_level_db=-30,
    mask_spec=False,
)


# %%
# create a dataset object
dataset = DataSet(DATASET_ID, hparams=hparams)
dataset.sample_json
len(dataset.data_files)


# %%
metadata = []
for key in tqdm(dataset.data_files.keys(), leave=False):
    metadata.append(pd.DataFrame(dataset.data_files[key].data))
metadata = pd.concat(metadata)

# %%

date_counts = []
for nestbox in metadata["nestbox"].unique():
    n = metadata.nestbox.str.contains(nestbox).sum()
    date = min(metadata[metadata.nestbox == nestbox]["date"])
    dB = 20 * math.log10(
        statistics.mean(metadata[metadata.nestbox == nestbox]["max_amplitude"]) / 32767
    )
    date_counts.append([nestbox, n, dB, date])

date_counts = pd.DataFrame(date_counts, columns=["nestbox", "count", "dB", "date"])
date_counts["date"] = pd.to_datetime(date_counts["date"])
date_counts = date_counts[date_counts.date != "2020-03-29"]


# %%

coords_file = RESOURCES_DIR / "nestboxes" / "nestbox_coords.csv"
tmpl = pd.read_csv(coords_file)

nestboxes = tmpl[tmpl["nestbox"].isin(date_counts.nestbox.unique())]

# To convert BNG to WGS84:


bng = pyproj.Proj(init="epsg:27700")
webmercator = pyproj.Proj(init="epsg:3857")
wgs84 = pyproj.Proj(init="epsg:4326")


def convertCoords(row):
    x2, y2 = pyproj.transform(bng, wgs84, row["x"], row["y"])
    return pd.Series([x2, y2])


nestboxes[["longitude", "latitude"]] = nestboxes[["x", "y"]].apply(
    convertCoords, axis=1
)


plt.scatter(nestboxes["longitude"], nestboxes["latitude"])

#  Add to syllable_df
data_clean = pd.merge(date_counts, nestboxes, how="inner")

# %%
# Save
filename = RESOURCES_DIR / "TAWNY" / year / "calls_dataset.csv"
ensure_dir(filename)
data_clean.to_csv(filename)


# %%


for nestbox in metadata["nestbox"].unique():
    print(statistics.mean(metadata[metadata.nestbox == nestbox]["max_amplitude"]))

# %%

# Plot count vs date
plot = (
    ggplot(date_counts, aes(x="date", y="count"))
    + geom_point()
    + geom_smooth(method="lm", se=True, alpha=0.3, span=0.9)
    + scale_y_log10(breaks=[1, 10, 100], labels=[1, 10, 100])
    + scale_x_datetime(breaks=date_breaks("7 days"), labels=date_format("%d %b"))
    + theme(
        figure_size=(7, 7),
        panel_grid_major_x=element_blank(),
        panel_grid_major_y=element_blank(),
        panel_grid_minor_x=element_blank(),
        panel_grid_minor_y=element_blank(),
    )
    + labs(
        title="Call count vs date recorded, 2020. n = {}\n".format(len(date_counts)),
        x="\nDate",
        y="Call count (log scale)",
    )
)
# ggsave(plot, filename=str(FIGURE_DIR / "count_vs_date_TAWNY_2020.png"), res=500)

fig = plot.draw()
fig.show()

# %%

(
    ggplot(date_counts, aes(x="nestbox", y="dB"))
    + geom_bar(stat="identity")
    # + geom_smooth(method="lm", se=True, alpha=0.3, span=0.9)
    # + scale_y_log10(breaks=[1, 10, 100], labels=[1, 10, 100])
    # + scale_x_datetime(breaks=date_breaks("7 days"), labels=date_format("%d %b"))
    + theme(
        figure_size=(7, 7),
        panel_grid_major_x=element_blank(),
        panel_grid_major_y=element_blank(),
        panel_grid_minor_x=element_blank(),
        panel_grid_minor_y=element_blank(),
    )
    + labs(
        title="Call count vs date recorded, 2020. n = {}\n".format(len(date_counts)),
        x="\nDate",
        y="Call count (log scale)",
    )
)
# ggsave(plot, filename=str(FIGURE_DIR / "count_vs_date_TAWNY_2020.png"), res=500)

# fig = plot.draw()
# fig.show()


# %%
