
# %%
import subprocess
from matplotlib.animation import FuncAnimation
import datetime as dt
import glob
import json
import os
import re
import time
import wave

import audio_metadata
import datashader as ds
import datashader.transfer_functions as dtf
import librosa
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phate
import seaborn as sns
import umap
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D
from multiprocess import Pool, cpu_count
from pathlib2 import Path
from PIL import Image
from scipy import signal
from src.avgn.utils.json import NoIndent, NoIndentEncoder
from src.avgn.visualization.projections import scatter_projections
from src.greti.read.paths import DATA_DIR, FIGURE_DIR, safe_makedir
from src.greti.read.soundscape_data import batch_save_mean_chunks, normalise_01
from tqdm.auto import tqdm


def make_cat_palette(labs, color_palette):
    pal = sns.color_palette(color_palette, n_colors=len(np.unique(labs)))
    lab_dict = {lab: pal[i] for i, lab in enumerate(np.unique(labs))}
    colors = np.array([lab_dict[i] for i in labs])
    return colors, lab_dict


def selection_palette(labs, colors=['#e60000', '#0088ff'], rest_col='#636363', select_labels=['Bean Wood', 'Great Wood']):
    rest_col = sns.color_palette([rest_col])
    pal = sns.color_palette(colors)
    lab_dict = {lab: pal[i] for i, lab in enumerate(np.unique(select_labels))}
    for lab in np.unique(labs):
        if lab not in select_labels:
            lab_dict[lab] = rest_col[0]
    colors = np.array([lab_dict[i] for i in labs])
    return colors, lab_dict


# %% Load data
DATASET_ID = "SOUNDSCAPES_2020"
out_dir = DATA_DIR / "syllable_dfs" / DATASET_ID
m5_chunks_df = pd.read_pickle(
    out_dir / ("full_dataset" + ".pickle"))  # Full names
m5_chunks_df.replace({'section': {'MP': 'Marley Plantation', 'C': 'Bean Wood', 'O': 'Broad Oak',
                                  'EX': 'Extra', 'CP': 'Common Piece', 'P': 'Pasticks',
                                  'SW': 'Singing Way', 'W': 'Great Wood', 'B': 'Marley Wood'}}, regex=False,  inplace=True)


# Get relevant data
# same time across woods
# m5_chunks_df = m5_chunks_df[m5_chunks_df['hour'] == 4]

m5_chunks_df = m5_chunks_df[m5_chunks_df.section ==
                            'Marley Wood']  # MP on a day with a lot of data
m5_chunks_df = m5_chunks_df[m5_chunks_df.hour < 5]


# %% susbet for tests
# m5_chunks_df = m5_chunks_df.iloc[0:10000]
m5_chunks_df = m5_chunks_df.sample(frac=0.4, replace=True, random_state=1)
# Prepare spectrograms
# specs = [normalise_01(spec) for spec in m5_chunks_df.spec.values]
specs = [spec for spec in m5_chunks_df.spec.values]


# %% Project UMAP

umap_parameters = {
    "n_neighbors": 10,
    "min_dist": 0.25,
    "n_components": 3,
    "verbose": True,
    "init": "spectral",
    "low_memory": False,
}
fit = umap.UMAP(**umap_parameters)
m5_chunks_df["umap"] = list(fit.fit_transform(specs))

# %% PHATE

phate_parameters = {"n_jobs": -1, "knn": 10,
                    "n_pca": 100, "gamma": 0, 'n_components': 3}
phate_operator = phate.PHATE(**phate_parameters)  # **phate_parameters
m5_chunks_df["phate"] = list(phate_operator.fit_transform(np.array(specs)))

# %% 3D GIFs


def get_ax_lims(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    return xlim, ylim, zlim


def get_zoom_step_size(fraction, n_frames, ax):
    zoom_x = tuple((ax-(ax*fraction))/n_frames if ax >
                   0 else ((ax*fraction)-ax)/n_frames for ax in ax.get_xlim())
    zoom_y = tuple((ax-(ax*fraction))/n_frames if ax >
                   0 else ((ax*fraction)-ax)/n_frames for ax in ax.get_ylim())
    zoom_z = tuple((ax-(ax*fraction))/n_frames if ax >
                   0 else ((ax*fraction)-ax)/n_frames for ax in ax.get_zlim())
    return zoom_x, zoom_y, zoom_z


def set_zoom(fraction, ax):
    zoom_x = tuple((ax*fraction) if ax >
                   0 else (ax*fraction) for ax in ax.get_xlim())
    zoom_y = tuple((ax*fraction) if ax >
                   0 else (ax*fraction) for ax in ax.get_ylim())
    zoom_z = tuple((ax*fraction) if ax >
                   0 else (ax*fraction) for ax in ax.get_zlim())
    ax.set_xlim(zoom_x)
    ax.set_ylim(zoom_y)
    ax.set_zlim(zoom_z)
    return ax


def substract(tup):
    return tup[0] - tup[1]


def animate_zoom(ax, zoom_steps):
    current_lims = get_ax_lims(ax)
    new_lims = [[sum(x) if x[0] < 0 else substract(x)
                 for x in zip(current_lims[i], zoom_steps[i])] for i in range(3)]
    ax.set_xlim(new_lims[0])
    ax.set_ylim(new_lims[1])
    ax.set_zlim(new_lims[2])
    return ax, new_lims


# %%

# Paths and data
proj_str = 'phate'  # projection to use
fig_dir = FIGURE_DIR / DATASET_ID / "gifs" / "4am"
safe_makedir(fig_dir)
fp_in = str(fig_dir) + '/*.png'
fp_out = str(fig_dir / '4am.gif')
data = np.vstack(m5_chunks_df[proj_str])

# Settings
initial_point_size = 4
end_point_size = 8
cat_palette = "husl"
cont_palette = "RdYlBu_r"
textcol = '#b3b3b3'
by_time_of_day = True
by_area_of_woods, select_labs = True, False
by_nestbox = False
n_frames = 700
max_alpha = 0.1

# %%
# Plot figure

# Start figure
fig = plt.figure(figsize=(15, 15))
fig.patch.set_facecolor('k')
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('k')
pointsize = [initial_point_size] * n_frames


if by_time_of_day:
    labs = [x for x in list(m5_chunks_df.hour)]
    colors, lab_dict = make_cat_palette(labs, cont_palette)
    scat = ax.scatter(data[:, 0], data[:, 1], data[:, 2], zdir='z',
                      depthshade=True, s=pointsize, color=colors, alpha=max_alpha, marker='o', linewidth=0)

elif by_area_of_woods:
    labs = [x for x in list(m5_chunks_df.section)]
    if not select_labs:
        colors, lab_dict = make_cat_palette(labs, cat_palette)
    else:
        colors, lab_dict = selection_palette(
            labs, colors=['#e63b15', '#2081e3'], rest_col='#3b3a36', select_labels=['Bean Wood', 'Marley Plantation'])
    scat = ax.scatter(data[:, 0], data[:, 1], data[:, 2], zdir='z',
                      depthshade=True, s=pointsize, color=colors, alpha=max_alpha, marker='o', linewidth=0)

elif by_nestbox:
    labs = [x for x in list(m5_chunks_df.nestbox)]
    colors, lab_dict = make_cat_palette(labs, cat_palette)
    scat = ax.scatter(data[:, 0], data[:, 1], data[:, 2], zdir='z',
                      depthshade=True, s=pointsize, color=colors, alpha=max_alpha, marker='o', linewidth=0)

plt.axis('off')
fig.tight_layout()

# Set zoom
start_zoom = 1
end_zoom = 0.2
ax = set_zoom(start_zoom, ax)  # set initial zoom level
# end zoom level, as proportion of whole figure under initial zoom level
# after you set initial zoom level
zoom_steps = get_zoom_step_size(end_zoom, n_frames, ax)
pointsize_steps = (end_point_size - initial_point_size) / n_frames  # that
# Run this to check different zoom levels, etc #!remove before saving animation
# ax = set_zoom(end_zoom, ax)

# # Legend and title
# legend_elements = [
#     Line2D([0], [0], marker="o", linestyle="None", color=value, label=key)
#     for key, value in lab_dict.items()
# ]
# leg = ax.legend(handles=legend_elements, markerscale=1.3, labelspacing=1,
#                 facecolor='black', edgecolor=None, framealpha=0.3,
#                 loc='upper right', fontsize=12
#                 )
# leg.get_frame().set_linewidth(0.0)
# for text in leg.get_texts():
#     text.set_color(textcol)
# # fig.suptitle("A big long suptitle that runs into the title\n",
# #              y=0.85, fontweight='bold', color=textcol)

frame_duration = 1000/24
gif_lenght_s = frame_duration*n_frames/1000
fade_length_s = 4.5
n_transition_frames = fade_length_s*1000 / frame_duration

# Export images
for i in tqdm(range(0, n_frames)):
    azimuth = i * 100 / n_frames
    elevation = ((i - 0) / (n_frames)) * (90 + 90) - 90
    ax.view_init(elev=elevation, azim=azimuth)
    ax, new_lims = animate_zoom(ax, zoom_steps)
    pointsize = [p + pointsize_steps for p in pointsize]
    # if i <= n_transition_frames:
    #     alpha = (((i - 0) * max_alpha) / n_transition_frames) + 0
    # if i >= n_frames - n_transition_frames:
    #     alpha = (((i - (n_frames - n_transition_frames)) * - max_alpha) /
    #              n_transition_frames) + max_alpha

    scat.set_sizes(pointsize)
    # scat.set_alpha(alpha)
    # print(
    #     f'Iteration: {i}, elevation: {elevation}, pointsize: {pointsize[0]}, lims: {new_lims}')
    plt.savefig(fig_dir / str(str(i).zfill(3) + '.png'),
                bbox_inches="tight", pad_inches=0)

plt.close()


# %%
# Put gif together
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=1000/24, loop=0)

# %%
# Export images
for n in tqdm(range(0, 100)):
    ax.elev += 0.4
    ax.azim += 0.3
    ax.dist -= 0.08
    plt.savefig(fig_dir / str(str(n).zfill(3) + '.png'),
                bbox_inches="tight", pad_inches=0)

# Put gif together
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=50, loop=0)

# %% example

n_frames = 500
pointsize = 50

fig = plt.figure()
fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor((0.5, 0.5, 0.5))


gradient = np.linspace(0, 1, 2)
X, Y, Z = np.meshgrid(gradient, gradient, gradient)
colors = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1)
pointsize = [50] * n_frames
scat = ax.scatter(X, Y, Z, alpha=1.0, s=pointsize,
                  c=colors, marker='o', linewidth=0)
fig.set_size_inches(5, 5)
# plt.axis('off')

izoom = 0.2
# Set initial zoom
ylim = ax.get_ylim()
xlim = ax.get_xlim()
zlim = ax.get_zlim()
ax.set_ylim(ylim[0]-izoom, ylim[1]+izoom)
ax.set_xlim(xlim[0]-izoom, xlim[1]+izoom)
ax.set_zlim(zlim[0]-izoom, zlim[1]+izoom)


def update(i, fig, ax, zoom=- 0.002):
    azimuth = ((i * 360) / n_frames)
    elevation = ((i - 0) / (n_frames)) * (180 + 180) - 180
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    zlim = ax.get_zlim()
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_ylim(ylim[0]-zoom, ylim[1]+zoom)
    ax.set_xlim(xlim[0]-zoom, xlim[1]+zoom)
    ax.set_zlim(zlim[0]-zoom, zlim[1]+zoom)
    global pointsize
    pointsize = [p + 2 for p in pointsize]
    scat.set_sizes(pointsize)
    return fig, ax


anim = FuncAnimation(fig, update, frames=np.arange(
    0, 360, 1), repeat=True, fargs=(fig, ax))
anim.save(fp_out, dpi=80, writer='imagemagick', fps=24)


# %%
# n Compress?

cmd = 'magick convert %s.gif -fuzz 5%% -layers Optimize %s_r.gif' % (fn, fn)
subprocess.check_output(cmd)


# %%
# save this as an example of datashader, not useful here due to few data


labs = [(x.hour*60+x.minute+x.second/60) for x in list(m5_chunks_df.time)]
projection = pd.DataFrame(m5_chunks_df["umap"].tolist(), columns=['x', 'y'])
projection['time'] = labs


def bg(img): return dtf.set_background(img, "black")


cvs = ds.Canvas(plot_width=800, plot_height=800)
agg = cvs.points(projection, 'x', 'y', ds.max('time'))
bg(dtf.shade(agg, how='eq_hist', min_alpha=100))


# %%

agg = cvs.points(projection, 'x', 'y', 'time')
img = tf.shade(agg, cmap=['green', 'yellow', 'red'], how='eq_hist')

# %%
# Categorical labels

ddf.hvplot.points(x="easting", y="northing",

                  c="race",
                  cmap={'w': 'aqua', 'b': 'lime',  'a': 'red',
                        'h': 'fuchsia', 'o': 'yellow'}

                  aggregator=ds.count_cat("race"),
                  datashade=True,
                  crs=ccrs.GOOGLE_MERCATOR,
                  ).opts(bgcolor="black")

# %% 'legacy' code

# def update(i, fig, ax, zoom_steps):
#     azimuth = ((i * 360) / n_frames)
#     elevation = ((i - 0) / (n_frames)) * (180 + 180) - 180
#     ax.view_init(elev=elevation, azim=azimuth)
#     # deactivated this to check if it was the problem, it wasnt
#     ax, new_lims = animate_zoom(ax, zoom_steps)
#     global pointsize
#     # it wasnt the points either
#     pointsize = [p + pointsize_steps for p in pointsize]
#     scat.set_sizes(pointsize)
#     print(
#         f'Iteration: {i}, elevation: {elevation}, pointsize: {pointsize[0]}, lims: {new_lims}')
#     return fig, ax


# anim = FuncAnimation(fig, update, frames=n_frames, interval=1, repeat=False, fargs=(
#     fig, ax, zoom_steps))  # removed int(/) bit, not this either
# print(f'Saving animation as {fp_out}')
# anim.save(fp_out, dpi=80, writer='imagemagick', fps=24)

# # frames=np.arange(0, 360, 2)
# # TODO: reset point size manually
#
