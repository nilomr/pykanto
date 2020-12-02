import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py
import pandas as pd
import numpy as np
from ipywidgets import interactive, HBox, VBox, widgets
from src.avgn.visualization.projections import colorline
from itertools import chain
from plotly.subplots import make_subplots
import random
import matplotlib.transforms as mtrans
from PIL import Image
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import re
import seaborn as sns


def prepare_interactive_data(indv_dfs, indv, pal_name):
    """This function prepares a dataframe of notes and its corresponding colour palette, which can then be plotted.

    Args:
        indv_dfs (dict) : Dictionary containing data for all birds
        indv (str): Code for bird
        pal_name (str): The name of a colour palette

    Returns:
        df, colour, palette: a dataframe of labels, a colour dictionary and a full palette for a given bird
    """

    labs = indv_dfs[indv]["hdbscan_labels_fixed"].values
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

    print(newlabs, newpalette)

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
    sequence_pos = indv_dfs[indv]["syllables_sequence_pos"]

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
    fig.savefig(buf, dpi=100, bbox_inches="tight", pad_inches=0, transparent=True)
    buf.seek(0)
    img = Image.open(buf)
    return img


def pil2datauri(img):
    # converts PIL image to datauri
    data = BytesIO()
    img.save(data, "png")
    data64 = base64.b64encode(data.getvalue())
    return "data:img/png;base64," + data64.decode("utf-8")


def plot_sample_notes(indv_dfs, indv, labels, colour):
    """Make plot containing examples of each existing cluster label, with colour labels

    Args:
        indv_dfs (dict) : Dictionary containing data for all birds
        indv (str): Which bird
        labels (list): A list of labels to plot
        colour (dict): Colour dictionary with labels:colours

    Returns:
        PIL image: The plot, converted to PIL so that it can be embedded in a plotly figure
    """

    fig, ax = plt.subplots(nrows=len(labels), ncols=15, figsize=(10, 10))

    for row, label in zip(ax, labels):
        specs = indv_dfs[indv][
            indv_dfs[indv]["hdbscan_labels_fixed"] == label
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


def interactive_plot(indv_dfs, indv, pal_name, viz_proj):
    """Main function to make an interactive scatterplot with a) notes, coloured by label, and their transtions ; b) examples of each label

    Args:
        indv_dfs (dict) : Dictionary containing data for all birds
        indv (str): Bird to plot
        pal_name (str): Name of palette to use
        viz_proj (str): Which projection to use

    Returns:
        FigureWidget: An interactive plotly figure
    """

    # prepare data (scatterplot)
    new_df, colour, palette = prepare_interactive_data(indv_dfs, indv, pal_name)

    # get data ready (transition lines)
    all_coords_nas = get_transition_df(indv_dfs, indv, viz_proj)

    # Start plotting interactive fig
    fig = make_subplots(rows=1, cols=2)

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

    # Add image
    example_image = plot_sample_notes(
        indv_dfs, indv, [int(i) for i in label_list], colour
    )

    fig.add_layout_image(
        dict(
            source=example_image,
            xref="paper",
            yref="paper",
            x=-1.2,
            y=4,
            sizex=7,
            sizey=7,
            opacity=1,
            layer="above",
        ),
        row=1,
        col=2,
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
        width=1300,
        height=700,
        legend=dict(orientation="v"),
        legend_title_text="Toggle <br> element",
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

    return fig, colour, new_df, palette


def assign_new_label(
    indv_dfs, indv, fig, pal_name, palette, colour, new_df, label_to_assign
):
    """Assign new labels to selection in current interactive plot. It updates the global colour dictionary, fig, and dataframe.

    Args:
        label_to_assign (int): the numeric label to assign (e.g. -1 for noise, 5)

    """

    label_to_assign = str(label_to_assign)

    input_num = re.sub("^-", "", label_to_assign)

    if not str.isdigit(input_num):
        raise SyntaxError(f"{input_num} is not an integral at heart")

    selection = []
    for f in fig.data[1:]:
        for name, name_new_df in new_df.groupby("labs"):
            if f.name == name:
                try:
                    selection = selection + [
                        name_new_df.iloc[i].name for i in f.selectedpoints
                    ]
                    # print(name, len(name_new_df), f.name, len(f.selectedpoints))
                except:
                    print("No selected points")
                    # print(name, len(name_new_df), f.name, len(f.selectedpoints))
                    # print([i for i in f.selectedpoints])

    if not selection:
        raise Exception("There are no selected points")

    for index in selection:
        new_df.loc[index, "labs"] = label_to_assign

    colour = update_colours(new_df, colour, pal_name)

    label_list = new_df.labs.unique().tolist()
    label_list.sort(key=int)

    fig.data = [fig.data[0]]

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

    if len(indv_dfs[indv]["hdbscan_labels"]) == len(new_df):
        indv_dfs[indv]["hdbscan_labels_fixed"] = [int(i) for i in new_df.labs]
    else:
        raise Exception("Different length data structures")

    print(label_list, colour)
    example_image = plot_sample_notes(
        indv_dfs, indv, [int(i) for i in label_list], colour
    )
    newimage = pil2datauri(example_image)
    fig.layout.images[0]["source"] = newimage

    fig.update_layout(
        title={
            "text": f"{indv}: Changed {len(selection)} points to label '{label_to_assign}'",
            "y": 0.935,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        }
    )

    return colour

