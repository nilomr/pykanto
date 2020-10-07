# %%
import glob
import string
from datetime import datetime
from os import fspath

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
import seaborn as sns
import src
from IPython import get_ipython
from IPython.display import display
from joblib import Parallel, delayed
from mizani.breaks import date_breaks
from mizani.formatters import date_format
from plotnine import *
from src.avgn.dataset import DataSet
from src.avgn.utils.hparams import HParams
from src.avgn.utils.paths import ensure_dir
from src.greti.read.paths import *
from tqdm import tqdm

get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")


plt.rcParams["axes.grid"] = False


# %%

# ### get data

DATASET_ID = "GRETI_HQ_2020_segmented"
YEAR = "2020"

# save_loc = DATA_DIR / "syllable_dfs" / DATASET_ID / "{}.pickle".format(DATASET_ID)

save_loc = DATA_DIR / "embeddings" / DATASET_ID / "full_dataset.pickle"
syllable_df = pd.read_pickle(save_loc)


# %%
# Create a dataset object

dataset = DataSet(DATASET_ID)
len(dataset.data_files)


# %%
# Make dataframe with all metadata
metadata = []
for key in tqdm(dataset.data_files.keys(), leave=False):
    metadata.append(pd.DataFrame(dataset.data_files[key].data))
metadata = pd.concat(metadata)


# %%
# Count the number of syllables per nest

syllable_n = pd.Series(
    [len(syllable_df[syllable_df.indv == ind]) for ind in syllable_df.indv.unique()]
)

# %%
# Get data for each individual (with cluster info)

DATASET_ID = "GRETI_HQ_2020_segmented"
YEAR = "2020"

all_indv_dfs = pd.concat(
    [
        pd.read_pickle(i)
        for i in list((DATA_DIR / "indv_dfs" / DATASET_ID).glob("*.pickle"))
    ]
)

# %%
# Count the number of song types per bird

grouped = all_indv_dfs.groupby("indv")

type_counts = grouped.apply(
    lambda x: len(x["hdbscan_labels"].unique()[x["hdbscan_labels"].unique() >= 0])
)
# type_counts=type_counts[type_counts <= 0]


# %%

# Get number of songs per nest
date_counts = []
for nestbox in metadata["nestbox"].unique():
    n = metadata.nestbox.str.contains(nestbox).sum()
    date = min(metadata[metadata.nestbox == nestbox]["date"])
    date_counts.append([nestbox, n, date])

date_counts = pd.DataFrame(date_counts, columns=["nestbox", "song_count", "date"])
date_counts["date"] = pd.to_datetime(date_counts["date"])
date_counts = date_counts[date_counts.date != "2020-03-29"]  # remove early test

# import the latest brood data downloaded from https://ebmp.zoo.ox.ac.uk/broods
brood_data_path = RESOURCES_DIR / "brood_data" / "2020"
list_of_files = glob.glob(fspath(brood_data_path) + "/*.csv")
latest_file = max(list_of_files, key=os.path.getctime)
greti_nestboxes = pd.DataFrame(
    (pd.read_csv(latest_file, dayfirst=True).query('Species == "g"'))
)
greti_nestboxes["nestbox"] = greti_nestboxes["Pnum"].str[5:]
greti_nestboxes["Lay date"] = pd.to_datetime(greti_nestboxes["Lay date"], dayfirst=True)
# Merge
date_counts = pd.merge(date_counts, greti_nestboxes, on="nestbox", how="outer")
# Add column = how long after egg laying onset was nest recorded?
date_counts["difference"] = (date_counts["Lay date"] - date_counts["date"]).dt.days

# %%
# prepare and save full dataset

cols = ["nestbox", "n", "syll_type_n"]
data = []

for ind in syllable_df.indv.unique():
    try:
        syll_type_n = int(float(type_counts[ind]))
    except:
        syll_type_n = 0
    n = int(len(syllable_df[syllable_df.indv == ind]))
    zipped = zip(cols, [ind, n, syll_type_n])
    a_dictionary = dict(zipped)
    data.append(a_dictionary)

syllable_info = pd.DataFrame(columns=cols)
syllable_info = syllable_info.append(data, True)
# syllable_info[syllable_info.syll_type_n > 0].shape[0]

GRETI_dataset_2020 = pd.merge(date_counts, syllable_info, on="nestbox", how="outer")
out_dir = DATA_DIR / "resources" / DATASET_ID / ("full_dataset" + ".csv")
ensure_dir(out_dir)
GRETI_dataset_2020.to_csv(out_dir, index=False)

#####################
#####################
#####################
#####################
#####################
#####################

#%%
# Convert coordinates for mapbox map
import pyproj

coords_file = RESOURCES_DIR / "nestboxes" / "nestbox_coords.csv"
coordinates = pd.read_csv(coords_file)


bng = pyproj.Proj(init="epsg:27700")
webmercator = pyproj.Proj(init="epsg:3857")
wgs84 = pyproj.Proj(init="epsg:4326")


def convertCoords(row):
    x2, y2 = pyproj.transform(bng, wgs84, row["x"], row["y"])
    return pd.Series([x2, y2])


coordinates[["longitude", "latitude"]] = coordinates[["x", "y"]].apply(
    convertCoords, axis=1
)

coords_out = RESOURCES_DIR / "nestboxes" / "nestbox_coords_transformed.csv"
coordinates.to_csv(coords_out, index=False)

# %%
# plot some example syllable spectrograms

bird = "MP66"

sample_specs = syllable_df[syllable_df.indv == bird].spectrogram.values
sample_specs = np.invert(sample_specs)
draw_spec_set(
    sample_specs, cmap="bone", maxrows=7, colsize=15, zoom=3, facecolour="#f2f1f0"
)

fig_out = (
    FIGURE_DIR
    / YEAR
    / "examples"
    / (
        "{}_sample_syllables_".format(bird)
        + str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
        + ".png"
    )
)
ensure_dir(fig_out)

plt.savefig(
    fig_out, dpi=300, bbox_inches="tight", pad_inches=0.3, transparent=False,
)
# plt.show()
plt.close()


# %%
## Plot frequency distribution of song counts

freq_data = metadata["nestbox"].value_counts()


nbins = 25
fig = plt.figure(figsize=(10, 3))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], hspace=0, wspace=0.15)
axes = [fig.add_subplot(gs[i]) for i in range(2)]


# histogram of song counts
axes[0].hist(freq_data, bins=nbins, color="#8fa1ca")
# histogram of song counts on log scale.

axes[1].hist(freq_data, bins=nbins, color="#8fa1ca")
for ax in axes:
    ax.set_facecolor("#f2f1f0")
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis=u"both", which=u"both", length=0)
    for tick in ax.get_xaxis().get_major_ticks():
        tick.set_pad(8.0)
        tick.label1 = tick._get_text1()
plt.yscale("log")

fig.suptitle("Frequency distribution of song per bird", fontsize=15, y=1.03)
axes[0].set_ylabel("Number of birds", labelpad=13, fontsize=11)
axes[0].set_xlabel("Number of songs", labelpad=13, fontsize=11, x=1.04)
plt.annotate("n = {}".format(len(freq_data)), xy=(0.1, 0.9), xycoords="axes fraction")
plt.annotate("n = {}".format(len(freq_data)), xy=(-1.06, 0.9), xycoords="axes fraction")

labels = string.ascii_uppercase[0 : len(axes)]

for ax, labels in zip(axes, labels):
    bbox = ax.get_tightbbox(fig.canvas.get_renderer())
    fig.text(
        0.92,
        0.95,
        labels,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
        transform=ax.transAxes,
    )

# plt.show()

fig_out = (
    FIGURE_DIR
    / YEAR
    / "population"
    / (
        "Frequency_distribution_song_n_"
        + str(datetime.now().strftime("%Y-%m-%d_%H:%M"))
        + ".png"
    )
)
ensure_dir(fig_out)
plt.savefig(
    fig_out, dpi=300, bbox_inches="tight", pad_inches=0.3, transparent=False,
)

plt.close()


# %%
# Plot frequency distribution of syllable counts and types

nbins = 25
fig = plt.figure(figsize=(10, 3))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], hspace=0, wspace=0.15)
axes = [fig.add_subplot(gs[i]) for i in range(2)]


# histogram of syllable types
axes[0].hist(type_counts, bins=15, color="#8fa1ca")
# histogram of syllable counts on log scale.

axes[1].hist(syllable_n, bins=nbins, color="#8fa1ca")
for ax in axes:
    ax.set_facecolor("#f2f1f0")
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis=u"both", which=u"both", length=0)
    for tick in ax.get_xaxis().get_major_ticks():
        tick.set_pad(8.0)
        tick.label1 = tick._get_text1()
plt.yscale("log")

fig.suptitle("Frequency distribution of note types and counts", fontsize=15, y=1.03)
axes[0].set_ylabel("Number of birds", labelpad=13, fontsize=11)
axes[0].set_xlabel("Number of note types", labelpad=13, fontsize=11)
axes[1].set_xlabel("Number of notes", labelpad=13, fontsize=11)
plt.annotate("n = {}".format(len(syllable_n)), xy=(0.1, 0.9), xycoords="axes fraction")
plt.annotate(
    "n = {}".format(len(type_counts)), xy=(-1.11, 0.9), xycoords="axes fraction"
)

labels = string.ascii_uppercase[0 : len(axes)]

for ax, labels in zip(axes, labels):
    bbox = ax.get_tightbbox(fig.canvas.get_renderer())
    fig.text(
        0.92,
        0.95,
        labels,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
        transform=ax.transAxes,
    )

# plt.show()

fig_out = (
    FIGURE_DIR
    / YEAR
    / "population"
    / (
        "Frequency_distribution_note_types_"
        + str(datetime.now().strftime("%Y-%m-%d_%H:%M"))
        + ".png"
    )
)
ensure_dir(fig_out)
plt.savefig(
    fig_out, dpi=300, bbox_inches="tight", pad_inches=0.3, transparent=False,
)

plt.close()


# %%

# Number of song types versus date

(
    ggplot(GRETI_dataset_2020, aes(x="date", y="n"))
    + geom_point()
    # + geom_smooth(method="lm", se=True, alpha=0.3, span=0.9)
    # + scale_y_log10(breaks=[1, 10, 100, 1000], labels=[1, 10, 100, 1000])
    + scale_x_datetime(breaks=date_breaks("7 days"), labels=date_format("%d %b"))
    + theme(
        figure_size=(7, 7),
        panel_grid_major_x=element_blank(),
        panel_grid_major_y=element_blank(),
        panel_grid_minor_x=element_blank(),
        panel_grid_minor_y=element_blank(),
    )
    + labs(
        title="Song count vs date recorded, 2020. n = {}\n".format(len(date_counts)),
        x="\nDate",
        y="Song count (log scale)",
    )
)


# %%
# Plot count vs date
plot = (
    ggplot(date_counts, aes(x="date", y="song_count"))
    + geom_point()
    + geom_smooth(method="loess", se=True, alpha=0.3, span=0.9)
    + scale_y_log10(breaks=[1, 10, 100, 1000], labels=[1, 10, 100, 1000])
    + scale_x_datetime(breaks=date_breaks("7 days"), labels=date_format("%d %b"))
    + theme(
        figure_size=(7, 7),
        panel_grid_major_x=element_blank(),
        panel_grid_major_y=element_blank(),
        panel_grid_minor_x=element_blank(),
        panel_grid_minor_y=element_blank(),
    )
    + labs(
        title="Song count vs date recorded, 2020. n = {}\n".format(len(date_counts)),
        x="\nDate",
        y="Song count (log scale)",
    )
)
ggsave(plot, filename=str(FIGURE_DIR / "count_vs_date_2020.png"), res=500)


# %%
# Plot count vs how long after laydate

plot = (
    ggplot(date_counts, aes(x="difference", y="song_count"))
    + geom_point()
    + geom_smooth(method="loess", se=True, alpha=0.3, span=0.9)
    + scale_y_log10(breaks=[1, 10, 100, 1000], labels=[1, 10, 100, 1000])
    + scale_x_reverse()
    # + scale_x_datetime(breaks=date_breaks("7 days"), labels=date_format("%d %b"))
    + theme(
        figure_size=(7, 7),
        panel_grid_major_x=element_blank(),
        panel_grid_major_y=element_blank(),
        panel_grid_minor_x=element_blank(),
        panel_grid_minor_y=element_blank(),
    )
    + labs(
        title="Song count vs lag (lay date - recording date), 2020. n = {}\n".format(
            len(date_counts)
        ),
        x="\nDifference (days)",
        y="Song count (log scale)",
    )
)

ggsave(plot, filename=str(FIGURE_DIR / "count_vs_lag_2020.png"), res=500)

# %%
# Plot count vs laydate

plot = (
    ggplot(date_counts, aes(x="Lay date", y="song_count"))
    + geom_point()
    + geom_smooth(method="loess", se=True, alpha=0.3, span=0.9)
    + scale_y_log10(breaks=[1, 10, 100, 1000], labels=[1, 10, 100, 1000])
    + scale_x_datetime(breaks=date_breaks("7 days"), labels=date_format("%d %b"))
    + theme(
        figure_size=(7, 7),
        panel_grid_major_x=element_blank(),
        panel_grid_major_y=element_blank(),
        panel_grid_minor_x=element_blank(),
        panel_grid_minor_y=element_blank(),
    )
    + labs(
        title="Song count vs lay date, 2020. n = {}\n".format(len(date_counts)),
        x="\nLay date",
        y="Song count (log scale)",
    )
)

ggsave(plot, filename=str(FIGURE_DIR / "count_vs_laydate_2020.png"), res=500)


# %%

# Plot syllable durations

fig_dims = (10, 4)
# sns.set_palette("cubehelix")
fig, ax = plt.subplots(figsize=fig_dims)
fig.suptitle("Note duration", fontsize=15, y=1.01)

for indv in np.unique(syllable_df.indv):
    plot = sns.distplot(
        (
            syllable_df[syllable_df.indv == indv]["end_time"]
            - syllable_df[syllable_df.indv == indv]["start_time"]
        ),
        label=False,
        rug=False,
        hist=False,
        kde_kws={"alpha": 0.3, "color": "#8ea0c9", "clip": (0.0, 0.4)},
        ax=ax,
    )

ax.grid(False)
ax.set_facecolor("#f2f1f0")
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.get_legend().set_visible(False)
ax.tick_params(axis=u"both", which=u"both", length=0)
for tick in ax.get_xaxis().get_major_ticks():
    tick.set_pad(8.0)
    tick.label1 = tick._get_text1()

plot.set_xlabel("Duration (s)", fontsize=11, labelpad=13)
plot.set_ylabel("Density", fontsize=11, labelpad=13)
plot.tick_params(labelsize=11)

plt.annotate(
    "n = {}".format(len(np.unique(syllable_df.indv))),
    xy=(0.03, 0.9),
    xycoords="axes fraction",
)


# plt.show()

fig_out = (
    FIGURE_DIR
    / YEAR
    / "population"
    / (
        "syllable_duration_pd_"
        + str(datetime.now().strftime("%Y-%m-%d_%H:%M"))
        + ".png"
    )
)
ensure_dir(fig_out)
plt.savefig(
    fig_out, dpi=300, bbox_inches="tight", pad_inches=0.3, transparent=False,
)

plt.close()

#%%

# Plot time of day per song


metadata["hour"] = pd.to_datetime(metadata["datetime"]).dt.hour
song_datetimes = metadata.filter(["nestbox", "hour"])

song_datetimes["hour"].value_counts()

# song_datetimes.pivot(index="nestbox", columns="datetime", values="datetime")

times_of_day = song_datetimes["hour"].value_counts()

sns.distplot(song_datetimes["hour"], kde=False, bins=20, kde_kws={"bw": 0.4})


# %%

