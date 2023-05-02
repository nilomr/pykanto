# ─── DESCRIPTION ──────────────────────────────────────────────────────────────

"""
Source code for pykanto's interactive labelling app.
"""
# To run from terminal:
# set `debug = True`` in this script.
# bokeh serve --show /home/nilomr/projects/pykanto/pykanto/app

# ─── DEPENDENCIES ─────────────────────────────────────────────────────────────

import pickle
import re
import sys
import warnings
from itertools import cycle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pkg_resources
from bokeh.core.properties import field
from bokeh.layouts import row
from bokeh.models import (
    BoxSelectTool,
    CustomJSHover,
    Div,
    HoverTool,
    LassoSelectTool,
    Legend,
)
from bokeh.models.annotations import LegendItem
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import CrosshairTool
from bokeh.models.widgets import Button, Dropdown
from bokeh.palettes import Set3_12
from bokeh.plotting import curdoc, figure
from bokeh.themes import Theme
from bokeh.transform import factor_cmap

from pykanto.app.data import load_app_data
from pykanto.utils.io import load_dataset
from pykanto.utils.paths import ProjDirs

# ──── FUNCTIONS ───────────────────────────────────────────────────────────────


def get_markers(marker_types: List[str], mapping: np.ndarray) -> List[str]:
    """
    Maps a list of markers to labels.

    Args:
        marker_types (List[str]): List of marker types.
        mapping (np.ndarray): List of labels.

    Returns:
        List[str]: List of markers.
    """
    if "-1" not in labs:
        labs.insert(0, "-1")
    marker_dict = {
        lab: marker for marker, lab in zip(cycle(marker_types), labs)
    }
    markers = [marker_dict.get(e, "") for e in mapping]
    return markers


def prepare_legend(
    source: ColumnDataSource,
    palette: List[str],
    labs: List[str],
    grouping_labels: str = "auto_class",
) -> tuple[List[str], list[str], field]:
    """
    Prepares the legend for the scatterplot.

    Args:
        source (ColumnDataSource): Data source for the scatterplot.
        palette (List[str]): Colour palette for the scatterplot.
        labs (List[str]): List of labels.
        grouping_labels (str, optional): Column name for the labels. Defaults to
        "auto_class".

    Returns:
        tuple[List[str], list[str], field]: Tuple of the colour palette, marker
        types and the colour field.
    """
    # Build marker shapes
    marker_types = [
        "asterisk",
        "circle",
        "triangle",
        "plus",
        "square",
        "star",
        "diamond",
        "hex",
        "inverted_triangle",
        "plus",
        "square",
        "star",
        "circle",
        "triangle",
        "hex",
    ]
    source.data["markers"] = get_markers(
        marker_types, source.data[grouping_labels]
    )
    # Build colour palette for the scatterplot
    # Add red for -1
    if "-1" not in labs:
        labs.insert(0, "-1")
    palette.insert(0, "#d92400")
    palette = tuple(palette)
    colours = factor_cmap(
        field_name=grouping_labels, palette=palette, factors=labs
    )

    return palette, marker_types, colours


def build_legend(source, html_markers, span_mk_sizes, mk_colours):
    """
    Builds the legend for the main interactive plot.

    Args:
        source (ColumnDataSource): Data source for the scatterplot.
        html_markers (dict): Dictionary of HTML markers.
        span_mk_sizes (dict): Dictionary of marker sizes.
        mk_colours (dict): Dictionary of marker colours.

    Returns:
        str: HTML for the legend.
    """

    pre = '<div class="legend_wrapper">'
    img_htmls = [pre]

    for label in sorted(set(source.data["auto_class"]), key=float):
        idx = np.where(source.data["auto_class"] == label)[0][0]
        spec = source.data["spectrogram"][idx]
        marker = html_markers[source.data["markers"][idx]]
        span = span_mk_sizes[source.data["markers"][idx]]
        colour = mk_colours[label]
        space = "&nbsp;" if len(label) > 1 else ""
        img_html = f"""  
        <div class="legend_container">
            <img class='legend_image' src={spec} style="width:100%"/>
            <div class="legend_img_label_container">
                <div class="legend_img_label">
                    <span style="color:{colour}" ><{span}>{space}{marker}</{span}> {label}</span>
                </div>
            </div>
        </div>
    """
        img_htmls.append(img_html)

    img_htmls.append("</div>")
    legend_html = "".join(img_htmls)
    return legend_html


def update_feedback_text(indv_list, remaining_indvs):
    """
    Updates the feedback text for the user.

    Args:
        indv_list (list): List of individuals.
        remaining_indvs (list): List of remaining individuals.

    Returns:
        str: Feedback text.
    """
    text = (
        f"// <b>{len(indv_list) - len(remaining_indvs)} out of {len(indv_list)} done</b> // "
        "Use the 'Label' button to change the label of the selected points. "
        "Label data are automatically saved when you click 'Next'.<br>"
        "Points labelled '-1' (red asterisks) are considered noise."
    )
    return text


def parse_boolean(b):
    """
    Parses a boolean.

    Args:
        b (str): String to parse.

    Returns:
        bool: Boolean.
    """
    return b == "True"


def set_range(sdata, ax_percentage=5):
    """
    Sets the range of the scatterplot.

    Args:
        sdata (pd.DataFrame): Dataframe with the scatterplot data.
        ax_percentage (int, optional): Percentage of the axis to extend.
            Defaults to 5.

    Returns:
        tuple: Tuple of the x and y ranges.
    """

    xmin, xmax = min(sdata["umap_x"]), max(sdata["umap_x"])
    ymin, ymax = min(sdata["umap_y"]), max(sdata["umap_y"])
    xlen, ylen = abs(xmin - xmax), abs(ymin - ymax)
    xextra, yextra = (ax_percentage / 100) * xlen, (ax_percentage / 100) * ylen
    xstart, xend = xmin - xextra, xmax + xextra
    ystart, yend = ymin - yextra, ymax + yextra
    return (xstart, xend), (ystart, yend)


if "sphinx" in sys.modules:
    pass
else:
    # ──── SETTINGS ────────────────────────────────────────────────────────────────

    debug = False

    if debug:
        warnings.warn(
            "Debug is set to true. The app will not work "
            "unless served directly"
        )
        song_level = False
        DATASET_ID = "GREAT_TIT"
        DATA_PATH = Path(pkg_resources.resource_filename("pykanto", "data"))
        PROJECT = Path(DATA_PATH).parent
        RAW_DATA = DATA_PATH / "segmented" / "great_tit"
        DIRS = ProjDirs(PROJECT, RAW_DATA, DATASET_ID, mkdir=True)
        max_n_labs = 12
        palette = list(Set3_12)
        dataset_loc = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
        dataset = load_dataset(dataset_loc, DIRS)

    else:
        # Get dataset location from command line args
        dataset_loc = Path(sys.argv[1])
        # Maximum possible n of vocalisation types
        max_n_labs = int(float(sys.argv[2]))
        # Whether to use single units or their average per vocalisation
        song_level = parse_boolean(sys.argv[3])
        # Colour palette to use
        pal = sys.argv[4:]
        palette = [
            re.findall(r"'(.*?)'", colour, re.DOTALL)[0] for colour in pal
        ]

    # ──── MAIN ────────────────────────────────────────────────────────────────────

    datatype = "data" if song_level else "units"
    labtype = "voc_app_data" if song_level else "unit_app_data"
    checktype = "voc_check" if song_level else "unit_check"

    # Load dataset
    if dataset_loc.is_file():
        dataset = pickle.load(open(dataset_loc, "rb"))
    else:
        raise FileNotFoundError(dataset_loc)
    print(f"App loaded {dataset.DATASET_ID} successfully")
    # Exclude individuals without labels (those dropped by
    # :func:`~pykanto.signal.cluster.dim_reduction_and_cluster`
    # because len < min_sample)
    indv_list = np.unique(
        getattr(dataset, datatype).dropna(subset=["auto_class"])["ID"]
    )

    # Where to store labelling information?
    if checktype not in dataset.files:
        dataset.files[checktype] = False

    # Get individuals to do
    remaining_indvs = np.unique(
        dataset.files.dropna(subset=[checktype]).query(f"{checktype} == False")[
            "ID"
        ]
    )

    if len(remaining_indvs) == 0:
        raise KeyError(
            f"All IDs in {indv_list} have already been checked. "
            "Close the app."
        )

    source = load_app_data(dataset, labtype, remaining_indvs[0])

    # Generate labels, markers and colours

    # Build labels
    labs = [str(lab) for lab in range(0, max_n_labs)]

    palette, marker_types, colours = prepare_legend(source, palette, labs)

    lightgrey = "#9e9e9e"
    background_grey = "#2F2F2F"

    # Tools in app
    TOOLS = "pan, wheel_zoom, box_select, lasso_select, crosshair, tap, undo, redo, reset"

    # Build the scatterplot
    splot = figure(
        tools=TOOLS,
        active_scroll="wheel_zoom",
        height_policy="max",
        width_policy="max",
        max_height=550,
        min_height=300,
        max_width=930,
        min_width=650,
        toolbar_location="above",
        x_axis_location=None,
        y_axis_location=None,
        output_backend="webgl",
    )
    splot.title.text = f"{remaining_indvs[0]}: vocal repertoire"
    splot.select(BoxSelectTool).select_every_mousemove = False
    splot.select(LassoSelectTool).select_every_mousemove = False

    r = splot.scatter(
        source=source,
        x="umap_x",
        y="umap_y",
        size=13,
        fill_color=colours,
        line_color=colours,
        marker="markers",
        alpha=0.5,
        hover_color="white",
    )

    # Hover tool

    custom_hov = CustomJSHover(
        code="""
        special_vars.indices = special_vars.indices.slice(0,5)
        if (special_vars.indices.indexOf(special_vars.index) >= 0)
        {
            return " "
        }
        else
        {
            return " hidden "
        }
    """
    )

    splot.add_tools(
        HoverTool(
            tooltips=(
                """
    <div @spectrogram{custom}>
        <div class="hover_container">
            <img class='hover_img' src='@spectrogram' style='float: top; width:120px;height:120px;'/>
            <div class="image_label">
                <span style='font-size: 16px; '>@auto_class</span>
            </div>
        </div>
    </div>
    """
            ),
            formatters={"@spectrogram": custom_hov},
        )
    )

    # Legend
    # Create a dummy legend that can be updated easily

    dummy_data = ColumnDataSource(
        {
            "x": [0] * len(labs),
            "y": [0] * len(labs),
            "label": labs,
            "markers": get_markers(marker_types, labs),
        }
    )

    dummy_colours = factor_cmap(
        field_name="label", palette=palette, factors=labs
    )

    dummy_scatter = splot.scatter(
        source=dummy_data,
        x="x",
        y="y",
        size=0,
        fill_color=dummy_colours,
        line_color=dummy_colours,
        fill_alpha=1.0,
        marker="markers",
        name="dummy_scatter",
    )

    def get_legend_items() -> list[LegendItem]:
        """
        Returns a list of LegendItem objects for the legend.
        """
        dummy_labels = list(dict.fromkeys(source.data["auto_class"]))
        legend_items = [
            LegendItem(label=label, renderers=[dummy_scatter], index=i)
            for i, label in enumerate(labs)
            if label in dummy_labels
        ]
        return legend_items

    legend_items = get_legend_items()

    # Inactive legend
    # plot_legend = Legend(
    #     items=legend_items,
    #     location=(0, 0),
    #     border_line_alpha=0,
    #     background_fill_alpha=0,
    #     label_text_font_size='15px',
    #     label_text_color='white',
    #     spacing=5)

    # splot.add_layout(plot_legend, place='right')

    # Style tools and toolbar
    hover = splot.select_one(HoverTool)
    hover.point_policy = "follow_mouse"
    hover.show_arrow = False

    crosshair = splot.select_one(CrosshairTool)
    crosshair.line_color = lightgrey
    crosshair.line_alpha = 0.5

    lasso_overlay = splot.select_one(LassoSelectTool).overlay
    lasso_overlay.line_color = None
    lasso_overlay.fill_color = lightgrey
    lasso_overlay.fill_alpha = 0.2

    box_overlay = splot.select_one(BoxSelectTool).overlay
    box_overlay.line_color = None
    box_overlay.fill_color = lightgrey
    box_overlay.fill_alpha = 0.2

    splot.toolbar.logo = "grey"

    # ──── MAIN INTERFACE ───────────────────────────────────────────────────────────

    # Add label button
    label_button = Dropdown(
        label="Label", menu=labs, css_classes=["label_button"]
    )

    # Next button
    next_button = Button(label="Next", css_classes=["next_button"])

    # Close app button
    close_button = Button(label="Stop App", css_classes=["close_button"])

    # Instructions/feedback text
    text = update_feedback_text(indv_list, remaining_indvs)
    feedback_text = Div(text=text, css_classes=["help_text"])

    def update_class_examples_div():
        spec = source.data[source.data["auto_class"] == "0"][0]["spectrogram"][
            0
        ]

    # Panel with class examples
    # Prepare marker dictionary for legend

    html_markers = {
        "asterisk": "&#8270",
        "circle": "&#9679",
        "diamond": "&#9830",
        "hex": "&#11042",
        "inverted_triangle": "&#9660",
        "plus": "&#10010",
        "square": "&#9632",
        "star": "&#9733",
        "triangle": "&#9650",
    }

    span_mk_sizes = {
        "asterisk": "mark-asterisk",
        "circle": "mark-circle",
        "diamond": "mark-diamond",
        "hex": "mark-hex",
        "inverted_triangle": "mark-inverted_triangle",
        "plus": "mark-plus",
        "square": "mark-square",
        "star": "mark-star",
        "triangle": "mark-triangle",
    }

    mk_colours = {lab: col for lab, col in zip(labs, palette)}

    legend_html = build_legend(source, html_markers, span_mk_sizes, mk_colours)

    # Add right legend panel
    class_examples_div = Div(text=legend_html)

    # Interactive labelling and updating

    def next_plot(event):
        # Add previous individual to list of already checked
        indv = source.data["ID"][0]

        dataset.files.loc[
            (dataset.files[checktype] == False) & (dataset.files["ID"] == indv),
            checktype,
        ] = True

        # Save labels from last individual to dataset
        label_dict = {
            key: label
            for key, label in zip(
                source.data["index"], source.data["auto_class"]
            )
        }

        if "class_label" not in getattr(dataset, datatype).columns:
            getattr(dataset, datatype).insert(
                1, "class_label", pd.Series(label_dict)
            )
        else:
            getattr(dataset, datatype)["class_label"].update(
                pd.Series(label_dict)
            )

        dataset.save_to_disk(verbose=True)

        # Get next individual and update plot
        remaining_indvs = np.unique(
            dataset.files.dropna(subset=[checktype]).query(
                f"{checktype} == False"
            )["ID"]
        )

        if len(remaining_indvs) == 0:
            done_t = "Done. You can now stop the app."
            print(done_t)
            feedback_text.text = done_t
            return

        indv = remaining_indvs[0]
        source.data = dict(load_app_data(dataset, labtype, indv).data)
        labels = source.data["auto_class"]
        source.data["markers"] = get_markers(marker_types, labels)

        # Update legend
        # splot.legend.items = get_legend_items()
        splot.title.text = f"{indv}'s song repertoire"
        class_examples_div.text = build_legend(
            source, html_markers, span_mk_sizes, mk_colours
        )

        xrange, yrange = set_range(source.data, ax_percentage=5)
        splot.x_range.start = xrange[0]
        splot.x_range.end = xrange[1]
        splot.y_range.start = yrange[0]
        splot.y_range.end = yrange[1]

        # Update help text
        feedback_text.text = update_feedback_text(indv_list, remaining_indvs)

    def get_selections(attr, old, new):
        global indices
        indices = new

    def update_labels(event):
        if "indices" not in globals():
            return
        labels = source.data["auto_class"]
        labels[indices] = event.item
        source.data["auto_class"] = labels
        source.data["markers"] = get_markers(marker_types, labels)

        # Update legend
        # splot.legend.items = get_legend_items()
        class_examples_div.text = build_legend(
            source, html_markers, span_mk_sizes, mk_colours
        )

    def close_app(event):
        feedback_text.text = "You may now close this browser tab."
        curdoc().add_next_tick_callback(sys.exit)

    # Actions
    r.data_source.selected.on_change("indices", get_selections)
    next_button.on_click(next_plot)
    label_button.on_click(update_labels)
    close_button.on_click(close_app)

    # Build app
    curdoc().add_root(
        row(
            close_button,
            next_button,
            label_button,
            width=250,
            css_classes=["button_div"],
        )
    )
    curdoc().add_root(row(splot, class_examples_div, css_classes=["main_div"]))
    curdoc().add_root(feedback_text)
    curdoc().title = "Pykanto: interactive song labelling"

    json_theme = {
        "attrs": {
            "Figure": {
                "background_fill_color": background_grey,
                "border_fill_color": background_grey,
                "outline_line_color": "#444444",
            },
            "Axis": {
                "axis_line_color": None,
            },
            "Grid": {
                "grid_line_dash": [6, 4],
                "grid_line_alpha": 0.1,
            },
            "Title": {"text_color": lightgrey, "text_font_size": "20px"},
        }
    }

    curdoc().theme = Theme(json=json_theme)
