# %%

import string
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import src
from IPython import get_ipython
from joblib import Parallel, delayed
from plotnine import *
from src.avgn.utils.hparams import HParams
from src.avgn.utils.paths import ensure_dir
from src.greti.read.paths import *

get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

plt.rcParams["axes.grid"] = False


# %%
# ### Import data

DATASET_ID = "GRETI_HQ_2020_segmented"
YEAR = "2020"

syll_dir = DATA_DIR / "embeddings" / DATASET_ID / "full_dataset.pickle"
syllable_df = pd.read_pickle(syll_dir)

dat_dir = (
    DATA_DIR / "resources" / DATASET_ID / ("{}_nest_data".format(DATASET_ID) + ".csv")
)
GRETI_dataset_2020 = pd.read_csv(dat_dir)

meta_dir = (
    DATA_DIR
    / "processed"
    / DATASET_ID
    / "metadata"
    / "{}_metadata.pickle".format(DATASET_ID)
)
metadata = pd.read_pickle(meta_dir)

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

type_counts = GRETI_dataset_2020.syll_type_n.dropna()

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

# %%

## Build dictionary of syllable types per song

# Load data

loc = (
    DATA_DIR / "syllable_dfs" / DATASET_ID / "{}_with_labels.pickle".format(DATASET_ID)
)
all_indv_dfs = pd.read_pickle(loc)


#%%

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
# ensure_dir(fig_out)
# plt.savefig(
#     fig_out, dpi=300, bbox_inches="tight", pad_inches=0.3, transparent=False,
# )

# plt.close()
plt.show()

# %%

# Plot cumulative repertoire for each bird

for bird in np.unique(cumulative_df.bird)[:20]:

    fig_dims = (5, 5)
    fig, ax = plt.subplots(figsize=fig_dims)

    x_data = cumulative_df[cumulative_df.bird == bird]["song_order"]
    y_data = cumulative_df[cumulative_df.bird == bird]["cumulative_n"]
    # y_data_norm = (y_data - np.min(y_data)) / (np.max(y_data) - np.min(y_data))

    plot = sns.lineplot(
        x=x_data, y=y_data, label=False, color="#1f1f1f", linewidth=4, ax=ax,
    )

    # ax.set(xscale="log")
    ax.grid(False)
    ax.set_facecolor("#f2f1f0")
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_legend().set_visible(False)

    # ax.xaxis.set_ticks(np.arange(0, 1100, 100))

    ax.tick_params(axis=u"both", which=u"both", length=0)
    for tick in ax.get_xaxis().get_major_ticks():
        tick.set_pad(8.0)
        tick.label1 = tick._get_text1()

    plt.setp(plot.lines, alpha=0.1)
    plot.set_xlabel("Number of songs recorded", fontsize=11, labelpad=13)
    plot.set_ylabel("New songs", fontsize=11, labelpad=13)
    plot.tick_params(labelsize=11)

    plt.text(
        x=0.127,
        y=0.88,
        s="Cumulative new song types ({})".format(bird),
        fontsize=14,
        fontweight="bold",
        ha="left",
        transform=fig.transFigure,
    )

    plt.subplots_adjust(top=0.8, wspace=0.3)

    plt.annotate(
        "n = {}".format(len(x_data)), xy=(0.83, 0.06), xycoords="axes fraction",
    )

    plt.show()

# %%
nsongs = []

for bird in np.unique(cumulative_df.bird):
    cumulative_df[cumulative_df.bird == bird]
    nsong = max(cumulative_df[cumulative_df.bird == bird]["song_order"])
    nsongs.append(nsong)

sum(1 for bird in nsongs if bird >= 40)

# %%
