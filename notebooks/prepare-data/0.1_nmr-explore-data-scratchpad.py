# %%
import glob
from os import fspath

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
import seaborn as sns
from IPython import get_ipython
from IPython.display import display
from joblib import Parallel, delayed
from mizani.breaks import date_breaks
from mizani.formatters import date_format
from plotnine import *
from tqdm import tqdm

import src
from src.avgn.dataset import DataSet
from src.avgn.utils.hparams import HParams
from src.greti.read.paths import *

get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

# %%


# %%

# ### get data

DATASET_ID = "GRETI_HQ_2020_segmented"
YEAR = "2020"

# save_loc = DATA_DIR / "syllable_dfs" / DATASET_ID / "{}.pickle".format(DATASET_ID)

save_loc = DATA_DIR / "syllable_dfs" / DATASET_ID / "full_dataset.pickle"
syllable_df = pd.read_pickle(save_loc)


# %%
# Create a dataset object

# hparams = HParams(
#     num_mel_bins=64,
#     n_fft=1024,
#     win_length_ms=15,
#     hop_length_ms=3,
#     mel_lower_edge_hertz=1200,
#     mel_upper_edge_hertz=10000,
#     butter_lowcut=1200,
#     butter_highcut=10000,
#     ref_level_db=30,
#     min_level_db=-30,
#     mask_spec=True,
#     n_jobs=-2,
#     verbosity=1,
#     nex=-1,
# )


# , hparams=hparams


dataset = DataSet(DATASET_ID)
dataset.sample_json
len(dataset.data_files)

# %%
# Make dataframe with all metadata
metadata = []
for key in tqdm(dataset.data_files.keys(), leave=False):
    metadata.append(pd.DataFrame(dataset.data_files[key].data))
metadata = pd.concat(metadata)


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
# Count the number of syllables per nest

syllable_n = pd.Series(
    [len(syllable_df[syllable_df.indv == ind]) for ind in syllable_df.indv.unique()]
)


# %%
# Plot frequency distribution of syllable counts

nbins = 60
fig = plt.figure(figsize=(10, 4))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], hspace=0, wspace=0.1)
axes = [fig.add_subplot(gs[i]) for i in range(2)]
fig.suptitle("Frequency distribution of syllable counts\n".format(indv), fontsize=15)
# histogram on linear scale
axes[0].hist(syllable_n, bins=nbins)
# histogram on log scale.
# Use non-equal bin sizes, such that they look equal on log scale.
logbins = np.geomspace(syllable_n.min(), syllable_n.max(), nbins)
axes[1].hist(syllable_n, bins=nbins)
for ax in axes:
    ax.set_facecolor("#f2f1f0")
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis=u"both", which=u"both", length=0)
plt.yscale("log")
plt.show()


# %%
## Plot frequency distribution of song segment count

freq_data = metadata["nestbox"].value_counts()

fig_dims = (7, 4)
style.use("ggplot")
# sns.set_palette("cubehelix")

fig, ax = plt.subplots(figsize=fig_dims)
sns.set_context(rc={"patch.linewidth": 0.0})

plot = sns.distplot(
    freq_data,
    bins=len(freq_data) + 1,
    kde=False,
    norm_hist=False,
    color="#242424",
    hist_kws=dict(alpha=1),
)

ax.grid(False)
plot.axes.set_title(
    "Songs per nestbox. n = {}, 2020".format(len(freq_data)), fontsize=15, pad=15
)
plot.set_xlabel("Number of songs", fontsize=15, labelpad=15)
plot.set_ylabel("Count", fontsize=15, labelpad=15)
plot.tick_params(labelsize=11)
plt.legend(loc=1, prop={"size": 11})

figure = plot.get_figure()
figure.savefig(
    str(FIGURE_DIR / "songs_per_nestbox_2020.png"), dpi=100, bbox_inches="tight"
)


# %%

# Plot songs per day recorded

date_counts = []
for nestbox in metadata["nestbox"].unique():
    n = metadata.nestbox.str.contains(nestbox).sum()
    date = min(metadata[metadata.nestbox == nestbox]["date"])
    date_counts.append([nestbox, n, date])

date_counts = pd.DataFrame(date_counts, columns=["nestbox", "count", "date"])
date_counts["date"] = pd.to_datetime(date_counts["date"])
date_counts = date_counts[date_counts.date != "2020-03-29"]  # remove early test

# Add column = how long after egg laying onset was nestbox recorded?

# import the latest brood data downloaded from https://ebmp.zoo.ox.ac.uk/broods
brood_data_path = RESOURCES_DIR / "brood_data" / "2020"
list_of_files = glob.glob(fspath(brood_data_path) + "/*.csv")
latest_file = max(list_of_files, key=os.path.getctime)
greti_nestboxes = pd.DataFrame(
    (
        pd.read_csv(latest_file, dayfirst=True)
        .query('Species == "g"')
        .filter(["Pnum", "Lay date", "Father"])
    )
)
greti_nestboxes["nestbox"] = greti_nestboxes["Pnum"].str[5:]
greti_nestboxes["Lay date"] = pd.to_datetime(greti_nestboxes["Lay date"], dayfirst=True)

date_counts = pd.merge(date_counts, greti_nestboxes, on="nestbox")
date_counts["difference"] = (date_counts["Lay date"] - date_counts["date"]).dt.days


# %%

# How many with >50 songs?
print("How many with >50 songs? " + str(sum(i > 50 for i in freq_data)))

# How many that layed eggs?
print("How many that layed eggs? " + str(len(date_counts)))

# How many ID?
print("How many ID'd? " + str(len(date_counts.dropna())))

# How many ID and > 50 songs?
print(
    "How many ID'd and > 50 songs? "
    + str(sum(i > 50 for i in date_counts.dropna()["count"]))
)


# %%
# Plot count vs date
plot = (
    ggplot(date_counts, aes(x="date", y="count"))
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
    ggplot(date_counts, aes(x="difference", y="count"))
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
    ggplot(date_counts, aes(x="Lay date", y="count"))
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
# Import (previously saved) syllable dataframe

save_loc = DATA_DIR / "syllable_dfs" / DATASET_ID / "{}.pickle".format(DATASET_ID)
syllable_df = pd.read_pickle(save_loc)

# %%

# Plot syllable durations

fig_dims = (10, 4)
# sns.set_palette("cubehelix")
fig, ax = plt.subplots(figsize=fig_dims)
sns.set_context(rc={"patch.linewidth": 0.0})


for indv in np.unique(syllable_df.indv[0]):
    plot = sns.distplot(
        (
            syllable_df[syllable_df.indv == indv]["end_time"]
            - syllable_df[syllable_df.indv == indv]["start_time"]
        ),
        label=indv,
        norm_hist=True,
        ax=ax,
    )

ax.grid(False)
plot.axes.set_title("Syllable duration", fontsize=15, pad=15)
plot.set_xlabel("Duration (s)", fontsize=15, labelpad=15)
plot.set_ylabel("Density", fontsize=15, labelpad=15)
plot.tick_params(labelsize=11)
plt.legend(loc=1, prop={"size": 11})
plt.show()

#%%

# Plot time of day per song


metadata["datetime"] = pd.to_datetime(metadata["datetime"])
song_datetimes = metadata.filter(["nestbox", "datetime"])

song_datetimes.pivot(index="nestbox", columns="datetime", values="values")

# %%
