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
fig = plt.figure(figsize=(10, 4))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], hspace=0, wspace=0.15)
axes = [fig.add_subplot(gs[i]) for i in range(2)]


# histogram of syllable types
axes[0].hist(type_counts, bins=15, color="#787878")
# histogram of syllable counts on log scale.

axes[1].hist(syllable_n, bins=nbins, color="#787878")
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

plt.yticks([1, 10, 100], ["1", "10", "100"])

plt.text(
    x=0.5,
    y=0.88,
    s="Frequency distribution of note types and counts",
    fontsize=15,
    fontweight="bold",
    ha="center",
    transform=fig.transFigure,
)

plt.subplots_adjust(top=0.8, wspace=0.3)

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

# Plot syllable durations

fig_dims = (10, 4)
fig, ax = plt.subplots(figsize=fig_dims)

for indv in np.unique(syllable_df.indv):
    plt.xlim(0, 0.4)
    plot = sns.distplot(
        (
            syllable_df[syllable_df.indv == indv]["end_time"]
            - syllable_df[syllable_df.indv == indv]["start_time"]
        ),
        label=False,
        rug=False,
        hist=False,
        kde_kws={"alpha": 0.3, "color": "#787878", "clip": (0.0, 0.4)},
        ax=ax,
    )

ax.grid(False)
ax.set_facecolor("#f2f1f0")
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.get_legend().set_visible(False)

plt.xticks(
    np.arange(0, 0.45, 0.05),
    ["0", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4"],
)

ax.tick_params(axis=u"both", which=u"both", length=0)
for tick in ax.get_xaxis().get_major_ticks():
    tick.set_pad(8.0)
    tick.label1 = tick._get_text1()

plt.text(
    x=0.5,
    y=0.94,
    s="Distribution of note durations",
    fontsize=15,
    fontweight="bold",
    ha="center",
    transform=fig.transFigure,
)
plt.text(
    x=0.5,
    y=0.88,
    s="Each line corresponds to a different bird",
    fontsize=11,
    color="#575757",
    ha="center",
    transform=fig.transFigure,
)

plt.subplots_adjust(top=0.8, wspace=0.3)
plot.set_xlabel("Duration (s)", fontsize=11, labelpad=13)
plot.set_ylabel("Density", fontsize=11, labelpad=13)
plot.tick_params(labelsize=11)

plt.annotate(
    "n = {}".format(len(np.unique(syllable_df.indv))),
    xy=(0.91, 0.9),
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

#%%

# Plot time of day per song

fig_dims = (10, 4)
fig, ax = plt.subplots(figsize=fig_dims)

for nestbox in np.unique(song_datetimes.nestbox):

    counts = (
        song_datetimes[song_datetimes.nestbox == nestbox]["hour"]
        .value_counts()
        .rename_axis("time")
        .reset_index(name="count")
    )

    plot = sns.lineplot(
        x=counts["time"], y=counts["count"], label=False, color="#787878", ax=ax,
    )


plt.setp(plot.lines, alpha=0.6)
ax.grid(False)
ax.set_facecolor("#f2f1f0")
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.get_legend().set_visible(False)
# ax.set(yscale="log")


ax.tick_params(axis=u"both", which=u"both", length=0)
for tick in ax.get_xaxis().get_major_ticks():
    tick.set_pad(8.0)
    tick.label1 = tick._get_text1()

plt.text(
    x=0.5,
    y=0.94,
    s="Number of songs recorded per hour",
    fontsize=15,
    fontweight="bold",
    ha="center",
    transform=fig.transFigure,
)
plt.text(
    x=0.5,
    y=0.88,
    s="Each line corresponds to a different bird",
    fontsize=11,
    color="#575757",
    ha="center",
    transform=fig.transFigure,
)

plt.subplots_adjust(top=0.8, wspace=0.3)
plot.set_xlabel("Time of the day (am)", fontsize=11, labelpad=13)
plot.set_ylabel("Number of songs", fontsize=11, labelpad=13)
plot.tick_params(labelsize=11)

plt.annotate(
    "n = {}".format(len(np.unique(syllable_df.indv))),
    xy=(0.91, 0.9),
    xycoords="axes fraction",
)

fig_out = (
    FIGURE_DIR
    / YEAR
    / "population"
    / ("time_plot" + str(datetime.now().strftime("%Y-%m-%d_%H:%M")) + ".png")
)
ensure_dir(fig_out)
plt.savefig(
    fig_out, dpi=300, bbox_inches="tight", pad_inches=0.3, transparent=False,
)

plt.close()


# %%

# Plot cumulative curves per bird

## Build dictionary of syllable types per song

## Parellelised code; only run in cluster


def build_note_dictionary(empty_dict=song_dict, df=all_indv_dfs, metadata=metadata):
    song_loc = [
        Path(song).name for song in metadata[metadata.nestbox == indv].wav_loc.tolist()
    ]
    song_dict[indv] = {}
    song_n = 0
    for song in song_loc:
        song_n += 1
        labels = all_indv_dfs[
            (all_indv_dfs.indv == indv) & (all_indv_dfs.key == str(song))
        ].hdbscan_labels.tolist()
        song_dict[indv].update({song_n: labels})


song_dict = {}

with Parallel(n_jobs=-2, verbose=3) as parallel:
    parallel(
        delayed(build_note_dictionary)(
            empty_dict=song_dict, df=all_indv_dfs, metadata=metadata
        )
        for indv in np.unique(all_indv_dfs.indv)
    )

# %%

song_dict = {}

for indv in np.unique(all_indv_dfs.indv):
    song_loc = [
        Path(song).name for song in metadata[metadata.nestbox == indv].wav_loc.tolist()
    ]
    song_dict[indv] = {}
    song_n = 0

    for song in song_loc:
        song_n += 1
        labels = all_indv_dfs[
            (all_indv_dfs.indv == indv) & (all_indv_dfs.key == str(song))
        ].hdbscan_labels.tolist()

        song_dict[indv].update({song_n: labels})


## Cumulative new songs

data = []

for bird, songs in song_dict.items():
    new = 0
    for number, labels in songs.items():
        new_list = []

        for label in labels:

            previous_labs = [
                b_labels
                for b_number, b_labels in song_dict[bird].items()
                if b_number < number
            ]  # the rest of songs for the current bird
            previous_labs_flat = [item for sublist in previous_labs for item in sublist]

            if label in previous_labs_flat:
                new_list.append(0)
            else:
                new_list.append(1)

        if 1 in new_list:
            new += 1

        temp_dict = {"bird": bird, "song_order": number, "cumulative_n": new}
        data.append(dict(temp_dict))

        # print(dict(zip(bird, number, new)))

        # [key for key in song_dict[bird].keys()][0:number-1]

cumulative_df = pd.DataFrame(data)


#%%

## Plot cumulative 'curves'

fig_dims = (8, 4)
fig, ax = plt.subplots(figsize=fig_dims)


for bird in np.unique(cumulative_df.bird):
    x_data = cumulative_df[cumulative_df.bird == bird]["song_order"]
    y_data = cumulative_df[cumulative_df.bird == bird]["cumulative_n"]
    # y_data_norm = (y_data - np.min(y_data)) / (np.max(y_data) - np.min(y_data))

    plot = sns.lineplot(
        x=x_data, y=y_data, label=False, color="#787878", linewidth=2.5, ax=ax,
    )

# ax.set(xscale="log")
ax.grid(False)
ax.set_facecolor("#f2f1f0")
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.get_legend().set_visible(False)

ax.xaxis.set_ticks(np.arange(0, 1100, 100))


ax.tick_params(axis=u"both", which=u"both", length=0)
for tick in ax.get_xaxis().get_major_ticks():
    tick.set_pad(8.0)
    tick.label1 = tick._get_text1()


plt.setp(plot.lines, alpha=0.1)
plot.set_xlabel("Number of songs recorded", fontsize=11, labelpad=13)
plot.set_ylabel("New songs", fontsize=11, labelpad=13)
plot.tick_params(labelsize=11)


plt.text(
    x=0.5,
    y=0.94,
    s="Cumulative new song types",
    fontsize=15,
    fontweight="bold",
    ha="center",
    transform=fig.transFigure,
)
plt.text(
    x=0.5,
    y=0.88,
    s="Each line corresponds to a different bird",
    fontsize=11,
    color="#575757",
    ha="center",
    transform=fig.transFigure,
)

plt.subplots_adjust(top=0.8, wspace=0.3)

plt.annotate(
    "n = {}".format(len(np.unique(cumulative_df.bird))),
    xy=(0.9, 0.93),
    xycoords="axes fraction",
)


fig_out = (
    FIGURE_DIR
    / YEAR
    / "population"
    / ("cumulative_plot" + str(datetime.now().strftime("%Y-%m-%d_%H:%M")) + ".png")
)
ensure_dir(fig_out)
plt.savefig(
    fig_out, dpi=300, bbox_inches="tight", pad_inches=0.3, transparent=False,
)

plt.close()

# %%
