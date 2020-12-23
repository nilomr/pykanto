import plotly.graph_objs as go
import plotly.offline as py
import pandas as pd
import numpy as np
from itertools import chain
from plotly.subplots import make_subplots
import random
from PIL import Image
import base64
from pathlib2 import Path
from io import BytesIO
import matplotlib.pyplot as plt
import re
import seaborn as sns
from src.avgn.visualization.network_graph import plot_network_graph
import librosa
from src.avgn.utils.paths import most_recent_subdirectory
from src.greti.read.paths import DATA_DIR
import matplotlib.patches as mpatches
from src.vocalseg.utils import (
    butter_bandpass_filter,
    plot_spec,
    spectrogram,
)


def prepare_interactive_data(
    indv_dfs, indv, pal_name, original_labels="hdbscan_labels_fixed"
):
    """This function prepares a dataframe of notes and its corresponding colour palette, which can then be plotted.

    Args:
        indv_dfs (dict) : Dictionary containing data for all birds
        indv (str): Code for bird
        pal_name (str): The name of a colour palette

    Returns:
        df, colour, palette: a dataframe of labels, a colour dictionary and a full palette for a given bird
    """

    labs = indv_dfs[indv][original_labels].values
    palette = sns.color_palette(pal_name, n_colors=len(np.unique(labs)))
    lab_dict = {lab: palette[i] for i, lab in enumerate(np.unique(labs))}

    lab_dict[-1] = (0.83137254902, 0.83137254902, 0.83137254902)

    x = np.array(list(indv_dfs[indv]["umap_viz"].values))[:, 0]
    y = np.array(list(indv_dfs[indv]["umap_viz"].values))[:, 1]
    # z = np.array(list(indv_dfs[indv]["umap_viz"].values))[:, 2]

    # colours = np.array([lab_dict[i] for i in labs])
    colour = {
        f"{lab}": f"rgb{tuple((np.array(color)*255).astype(np.uint8))}"
        for lab, color in lab_dict.items()
    }

    df = pd.DataFrame(
        data=np.column_stack((x.astype(np.object), y.astype(np.object), labs)),
        columns=["x", "y", "labs"],
    )
    df["labs"] = df["labs"].map(str)

    return df, colour, palette


def update_colours(new_df, colour, pal_name):
    """Add new, unused colours to a colour dictionary if there are new labels in the update dataframe. Max length of colour depends on palette length

    Args:
        new_df (dataframe): Data to plot (columns = 'x', 'y', 'labs')
        colour (dict): Existing dictionary of colours for each label
        pal_name (str): Name of palette to use
        palette: A full colour palette

    Returns:
        (colour) dict: Updated colour dictionary
    """

    label_list = new_df.labs.unique().tolist()
    label_list.sort(key=int)
    newpalette = sns.color_palette(pal_name, n_colors=20)
    newpalette = [
        f"rgb{tuple((np.array(col)*255).astype(np.uint8))}" for col in newpalette
    ]
    newpalette = [col for col in newpalette if col not in colour.values()]
    newlabs = [lab for lab in label_list if lab not in colour.keys()]

    newlab_dict = {lab: code for lab, code in zip(newlabs, newpalette)}
    newlab_dict["-1"] = "rgb(212, 212, 212)"

    colour.update(newlab_dict)

    return colour


def get_transition_df(indv_dfs, indv, viz_proj):
    """Prepare a dataframe with coordinates for every first-order transition. This wil be used to plot a network in the interactive scatterplot.

    Args:
        indv_dfs (dict) : Dictionary containing data for all birds
        indv (str): Which bird
        viz_proj (str): Which projection to use

    Returns:
        dataFrame: A dataframe containing coordinates for pairs of notes, with NA rows separating each pair.
    """

    # Prepare sequences
    projections = np.array(list(indv_dfs[indv][viz_proj].values))[:, 0:2]
    sequence_ids = np.array(indv_dfs[indv]["syllables_sequence_id"])

    # Build dataframe with each pair of points separated by a null row
    sequence_list = []
    for sequence in np.unique(sequence_ids):
        seq_mask = sequence_ids == sequence
        projection_seq = [i.tolist() for i in projections[seq_mask]]
        sequence_list.append(projection_seq)

    all_coords = pd.DataFrame(
        list(chain.from_iterable(sequence_list)), columns=("x", "y")
    )
    all_coords["id"] = all_coords.index
    tmp_df = (
        all_coords.iloc[1::2]
        .assign(id=lambda x: x["id"] + 1, y=np.nan)
        .rename(lambda x: x + 0.5)
    )
    all_coords_nas = (
        pd.concat([all_coords, tmp_df], sort=False).sort_index().reset_index(drop=True)
    )
    all_coords_nas.loc[all_coords_nas.isnull().any(axis=1), :] = np.nan

    return all_coords_nas


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(
        buf,
        dpi=100,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
        facecolor=fig.get_facecolor(),
    )
    buf.seek(0)
    img = Image.open(buf)
    return img


def pil2datauri(img):
    # converts PIL image to datauri
    data = BytesIO()
    img.save(data, "png")
    data64 = base64.b64encode(data.getvalue())
    return "data:img/png;base64," + data64.decode("utf-8")


def plot_sample_notes(
    indv_dfs,
    indv,
    labels,
    colour,
    original_labels="hdbscan_labels_fixed",
    reset_bird=False,
):
    """Make plot containing examples of each existing cluster label, with colour labels

    Args:
        indv_dfs (dict) : Dictionary containing data for all birds
        indv (str): Which bird
        labels (list): A list of labels to plot
        colour (dict): Colour dictionary with labels:colours

    Returns:
        PIL image: The plot, converted to PIL so that it can be embedded in a plotly figure
    """
    if reset_bird is True:
        original_labels = "hdbscan_labels"

    fig, ax = plt.subplots(nrows=len(labels), ncols=15, figsize=(10, 10))

    for row, label in zip(ax, labels):
        specs = indv_dfs[indv][
            indv_dfs[indv][original_labels] == label
        ].spectrogram.values

        if len(specs) >= 15:
            number = 15
        else:
            number = len(specs)

        specs_subset = [specs[i] for i in random.sample(range(len(specs)), number)]

        point_colour = colour[str(label)]
        point_colour = point_colour[point_colour.find("(") + 1 : point_colour.find(")")]
        point_colour = [
            val / 255.0 for val in tuple(map(int, point_colour.split(", ")))
        ]

        for i, (col, spec) in enumerate(zip(row, specs_subset)):
            if i == 0:
                col.add_artist(
                    plt.Circle((0.5, 0.5), 0.25, color=point_colour, alpha=1)
                )
                col.set_aspect("equal")
                col.axis("off")
            else:
                col.imshow(spec, cmap="Greys_r", aspect=2)
                col.axis("off")

    plt.subplots_adjust(wspace=0, hspace=0.1)

    fig.patch.set_facecolor("black")

    figure = fig2img(fig)
    plt.close()

    return figure


def plot_sample_labelled_song(
    DATASET_ID,
    indv_dfs,
    indv,
    colour,
    label,
    original_labels="hdbscan_labels_fixed",
    reset_bird=False,
):
    if reset_bird is True:
        original_labels = "hdbscan_labels"

    # Plot a randomly chosen song
    len_label = len(indv_dfs[indv].loc[indv_dfs[indv][original_labels] == label].key)
    index = random.sample(range(len_label), 1)[0]

    # load the wav
    key = indv_dfs[indv].loc[indv_dfs[indv][original_labels] == label].key.iloc[index]
    wav_dir = (
        most_recent_subdirectory(
            DATA_DIR / "processed" / DATASET_ID.replace("_segmented", ""),
            only_dirs=True,
        )
        / "WAV"
        / key
    )
    wav, rate = librosa.core.load(wav_dir, sr=None)

    # Bandpass
    data = butter_bandpass_filter(wav, 1200, 10000, rate)

    # Create the spectrogram
    spec = spectrogram(
        data,
        rate,
        n_fft=1024,
        hop_length_ms=3,
        win_length_ms=15,
        ref_level_db=30,
        min_level_db=-60,
    )

    # Trim or pad spectrogram
    if spec.shape[1] < 500:

        pad_left = 0
        pad_right = 500 - np.shape(spec)[1]
        spec = np.pad(
            spec, [(0, 0), (pad_left, pad_right)], "constant", constant_values=0
        )

    else:
        spec = spec[:, 0:500]

    # Narrower frequency band
    spec = spec[0:200]

    # clip to add contrast
    spec[spec < 0.5] = 0

    # Prepare label colours
    lab_colours = {
        int(lab): tuple([int(s) / 255 for s in re.findall(r"\b\d+\b", col)])
        for lab, col in colour.items()
    }

    # Plot the spectrogram with labels
    figure, ax = plt.subplots(figsize=(7, 3))
    plot_spec(spec, figure, ax, hop_len_ms=3, rate=rate, show_cbar=False, cmap="bone")
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ymin, ymax = ax.get_ylim()

    # Plot label rectangles

    for ix, row_2 in indv_dfs[indv][indv_dfs[indv].key == key].iterrows():
        # if row[original_labels] > -1:  # don't plot noise

        color = lab_colours[row_2[original_labels]]
        ax.add_patch(
            mpatches.Rectangle(
                [row_2.start_time, (ymax - (ymax - ymin) / 10)],
                row_2.end_time - row_2.start_time,
                (ymax - ymin) / 10,
                ec="none",
                color=color,
            )
        )

    ax.xaxis.tick_bottom()

    figure_img = fig2img(figure)

    plt.close()
    return figure_img


def plot_sample_songs_set(
    DATASET_ID,
    indv_dfs,
    indv,
    new_df,
    colour,
    original_labels="hdbscan_labels_fixed",
    reset_bird=False,
):

    if reset_bird is True:
        original_labels = "hdbscan_labels"

    label_list = new_df.labs.unique().tolist()
    labels = [int(i) for i in label_list]
    fig, ax = plt.subplots(nrows=len(labels), ncols=2, figsize=(4, 5))

    for row, label in zip(ax, labels):
        if label > -1:  # do not create a row for noise
            for col in row:
                figure_img = plot_sample_labelled_song(
                    DATASET_ID,
                    indv_dfs,
                    indv,
                    colour,
                    label,
                    original_labels=original_labels,
                    reset_bird=reset_bird,
                )
                col.imshow(figure_img, aspect=1)
                col.axis("off")

    plt.subplots_adjust(wspace=0, hspace=0)

    fig.patch.set_facecolor("black")

    figure_img_full = fig2img(fig)

    plt.close()

    return figure_img_full


def plot_directed_graph(
    indv_dfs,
    indv,
    viz_proj,
    colour,
    original_labels="hdbscan_labels_fixed",
    reset_bird=False,
):

    if reset_bird is True:
        original_labels = "hdbscan_labels"

    # Prepare necessary data
    projections = np.array(list(indv_dfs[indv][viz_proj].values))[:, 0:2]

    hdbscan_labs = indv_dfs[indv][original_labels]
    labs = hdbscan_labs.values

    sequence_ids = np.array(indv_dfs[indv]["syllables_sequence_id"])

    # Convert dictionary to palette, scaling colour values from 0 to 1
    net_palette_dict = {
        int(lab): tuple([int(s) / 255 for s in re.findall(r"\b\d+\b", col)])
        for lab, col in colour.items()
    }

    # Make plot
    fig, ax = plt.subplots(figsize=(10, 10))

    ax = plot_network_graph(
        labs,
        projections,
        sequence_ids,
        color_palette="tab20",
        pal_dict=net_palette_dict,
        min_cluster_samples=0,
        min_connections=0,
        facecolour="black",
        edge_width=0.1,
        edge_colour="white",
        point_size=300,
        arrowsize=30,
        ax=ax,
    )

    plt.subplots_adjust(wspace=0, hspace=0.1)

    fig.set_facecolor("black")
    ax.set_facecolor("black")

    figure = fig2img(fig)
    plt.close()

    return figure


def interactive_plot(
    DATASET_ID,
    indv_dfs,
    indv,
    pal_name,
    viz_proj,
    original_labels="hdbscan_labels_fixed",
    reset_bird=False,
):
    """Main function to make an interactive scatterplot with a) notes, coloured by label, and their transtions ; b) examples of each label

    Args:
        indv_dfs (dict) : Dictionary containing data for all birds
        indv (str): Bird to plot
        pal_name (str): Name of palette to use
        viz_proj (str): Which projection to use

    Returns:
        FigureWidget: An interactive plotly figure
    """

    if reset_bird is True:
        original_labels = "hdbscan_labels"

    # prepare data (scatterplot)
    new_df, colour, palette = prepare_interactive_data(
        indv_dfs, indv, pal_name, original_labels=original_labels
    )

    # get data ready (transition lines)
    all_coords_nas = get_transition_df(indv_dfs, indv, viz_proj)

    # Start plotting interactive fig
    fig = make_subplots(rows=1, cols=3)

    # Add transition lines
    fig.add_trace(
        go.Scatter(
            x=all_coords_nas.x,
            y=all_coords_nas.y,
            mode="lines",
            name="T",
            line=dict(color="rgba(255,255,255,0.7)", width=0.05),
        ),
        row=1,
        col=1,
    )
    fig.update_traces(connectgaps=False, marker=dict(size=5))

    # Add each label to scatterplot in a loop
    label_list = new_df.labs.unique().tolist()
    label_list.sort(key=int)
    for label in label_list:
        fig.add_trace(
            go.Scatter(
                x=new_df.loc[new_df.labs == label].x,
                y=new_df.loc[new_df.labs == label].y,
                mode="markers",
                name=label,
                marker=dict(size=5, color=colour[label]),
            ),
            row=1,
            col=1,
        )

    # Add sample songs with note labels
    example_image = plot_sample_songs_set(
        DATASET_ID,
        indv_dfs,
        indv,
        new_df,
        colour,
        original_labels=original_labels,
        reset_bird=reset_bird,
    )

    fig.add_layout_image(
        dict(
            source=example_image,
            xref="paper",
            yref="paper",
            x=-1.2,
            y=3.8,
            sizex=7,
            sizey=7,
            opacity=1,
            layer="above",
        ),
        row=1,
        col=2,
    )

    # Add directed graph
    example_graph = plot_directed_graph(
        indv_dfs,
        indv,
        viz_proj,
        colour,
        original_labels=original_labels,
        reset_bird=reset_bird,
    )

    fig.add_layout_image(
        dict(
            source=example_graph,
            xref="paper",
            yref="paper",
            x=-1.2,
            y=3.5,
            sizex=7,
            sizey=7,
            opacity=1,
            layer="above",
        ),
        row=1,
        col=3,
    )

    # Aesthetics
    fig.update_xaxes(
        showgrid=False, zeroline=False, visible=False, showticklabels=False
    )
    fig.update_yaxes(
        showgrid=False, zeroline=False, visible=False, showticklabels=False
    )
    fig.update_layout(
        autosize=False,
        width=1800,
        height=700,
        legend=dict(orientation="v"),
        legend_title_text="Toggle <br> element <br>",
        font_color="#cfcfcf",
        title_font_color="#cfcfcf",
        legend_title_font_color="#cfcfcf",
        title={
            "text": f"{indv}",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_range=(new_df.x.min() - 1, new_df.x.max() + 1),
        yaxis_range=(new_df.y.min() - 1, new_df.y.max() + 1),
        plot_bgcolor="black",
        paper_bgcolor="black",
    )

    # convert to figurewidget (listen for selections)
    fig = go.FigureWidget(fig)

    return fig, colour, new_df


def assign_new_label(
    DATASET_ID,
    indv_dfs,
    indv,
    fig,
    pal_name,
    colour,
    new_df,
    label_to_assign,
    relabel_noise=False,
):
    """Assign new labels to selection in current interactive plot. 
    It updates the global colour dictionary, fig, and dataframe.

    Args:
        label_to_assign (int): the numeric label to assign (e.g. -1 for noise, 5)

    """
    # Check and clean label imput
    label_to_assign = str(label_to_assign)
    input_num = re.sub("^-", "", label_to_assign)
    if not str.isdigit(input_num):
        raise SyntaxError(f"{input_num} is not an integral at heart")

    # Get real indexes of selected notes (and check if there are selected notes)
    selection = []
    for f in fig.data[1:]:
        for name, name_new_df in new_df.groupby("labs"):
            if f.name == name:
                try:
                    selection = selection + [
                        name_new_df.iloc[i].name for i in f.selectedpoints
                    ]
                except:
                    print("No selected points")
    if not selection:
        raise Exception("There are no selected points")

    # Re-label notes (with option to relabel noise)
    for index in selection:
        if relabel_noise is False:
            if new_df.loc[index, "labs"] > "-1":
                new_df.loc[index, "labs"] = label_to_assign
        elif relabel_noise is True:
            new_df.loc[index, "labs"] = label_to_assign

    # Update colour list to add new labels
    colour = update_colours(new_df, colour, pal_name)
    label_list = new_df.labs.unique().tolist()
    label_list.sort(key=int)

    # Clear existing data in figure
    fig.data = [fig.data[0]]

    # Add updated data
    for label in label_list:
        fig.add_trace(
            go.Scatter(
                x=new_df.loc[new_df.labs == label].x,
                y=new_df.loc[new_df.labs == label].y,
                mode="markers",
                name=label,
                marker=dict(size=3, color=colour[label]),
            ),
            row=1,
            col=1,
        )

    # Add new labels to main dataframe
    if len(indv_dfs[indv]["hdbscan_labels"]) == len(new_df):
        indv_dfs[indv]["hdbscan_labels_fixed"] = [int(i) for i in new_df.labs]
    else:
        raise Exception("Different length data structures")

    # Build plot with example note spectrograms and add it to plot

    example_image = plot_sample_songs_set(
        DATASET_ID,
        indv_dfs,
        indv,
        new_df,
        colour,
        original_labels="hdbscan_labels_fixed",
    )

    newimage = pil2datauri(example_image)
    fig.layout.images[0]["source"] = newimage

    # Plot directed graph and add to plot
    example_graph = plot_directed_graph(
        indv_dfs, indv, "umap_viz", colour
    )  # Note: can change projection used for visualisation if needed

    newimage1 = pil2datauri(example_graph)
    fig.layout.images[1]["source"] = newimage1

    fig.update_layout(
        title={
            "text": f"{indv}: Changed {len(selection)} points to label '{label_to_assign}'",
            "y": 0.935,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        }
    )

    # Return updated colour dictionary,
    # which will be input in the same function if further relabelling is needed.
    return colour


def check_new_bird(
    DATASET_ID,
    dfs_dir,
    indv_dfs,
    indv,
    indvs,
    i,
    pal_name,
    viz_proj,
    original_labels="hdbscan_labels",
    reset_bird=False,
):

    # Save the previous bird dataframe
    if i > -1:
        indv_dfs[indv].to_pickle(dfs_dir / (indv + "_labelled_checked.pickle"))

    if reset_bird is False or i == -1:
        i += 1

    if i >= len(indvs):
        raise Exception("End of list")

    already_checked = 0

    if reset_bird is False:
        while Path(dfs_dir / (indvs[i] + "_labelled_checked.pickle")).is_file() is True:
            i += 1
            already_checked += 1
            if i >= len(indvs):
                raise Exception("All birds have been checked")

    if already_checked > 0 and not i >= len(indvs):
        print(
            f"{already_checked} birds have already been checked, returning the next bird that hasn't"
        )

    indv = indvs[i]

    if "fig" in locals() or "fig" in globals():
        del fig

    fig, colour, new_df = interactive_plot(
        DATASET_ID,
        indv_dfs,
        indv,
        pal_name,
        viz_proj,
        original_labels=original_labels,
        reset_bird=reset_bird,
    )  # change to original_labels="hdbscan_labels_fixed" if you don't want to reset

    return fig, i, colour, new_df, indv
