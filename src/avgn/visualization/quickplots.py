import string
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.avgn.utils.paths import ensure_dir, most_recent_subdirectory
from src.avgn.visualization.network_graph import plot_network_graph
from src.avgn.visualization.projections import (
    draw_projection_transitions,
    plot_label_cluster_transitions,
    scatter_projections,
    scatter_spec,
)
from src.avgn.visualization.spectrogram import draw_spec_set, plot_example_specs
from src.greti.read.paths import DATA_DIR, FIGURE_DIR
from tqdm.autonotebook import tqdm


def draw_projection_plots(
    syllable_df,
    label_column="syllables_labels",
    projection_column="umap",
    figsize=(30, 10),
):
    """ draws three plots of transitions
    """
    fig, axs = plt.subplots(ncols=3, figsize=figsize)
    # plot scatter
    ax = axs[0]
    scatter_projections(
        projection=np.array(list(syllable_df[projection_column].values)),
        labels=syllable_df[label_column].values,
        ax=ax,
    )
    ax.axis("off")

    # plot transitions
    ax = axs[1]
    draw_projection_transitions(
        projections=np.array(list(syllable_df[projection_column].values)),
        sequence_ids=syllable_df["syllables_sequence_id"],
        sequence_pos=syllable_df["syllables_sequence_pos"],
        ax=ax,
    )
    ax.axis("off")

    # plot network graph
    ax = axs[2]
    elements = syllable_df[label_column].values
    projections = np.array(list(syllable_df[projection_column].values))
    sequence_ids = np.array(syllable_df["syllables_sequence_id"])
    plot_network_graph(
        elements, projections, sequence_ids, color_palette="tab20", ax=ax
    )

    ax.axis("off")

    return ax


def quad_plot_syllables(
    indv_dfs, YEAR, viz_proj="umap_viz", palette="Set2", facecolour="#f2f1f0"
):

    """Make plot including scatterplot, 'raw' syllable transitions, 
    transition directed network and example spectrograms.

    Args:
        indv_dfs ([pd.DataFrame]): Dataframe with individual data
        facecolour (str, optional): Subplot background colour. Defaults to "#f2f1f0".
        viz_proj(str, optional): Which projection to plot. Defaults to "umap_viz".
        labels (str, optional): Which labels to use. Defaults to "hdbscan_labels".
    """

    for indv in tqdm(indv_dfs.keys()):

        f = plt.figure(figsize=(40, 10))
        gs = f.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1], hspace=0, wspace=0.2)
        axes = [f.add_subplot(gs[i]) for i in range(4)]

        f.suptitle("Syllable clusters and transitions for {}".format(indv), fontsize=30)

        hdbscan_labs = indv_dfs[indv]["hdbscan_labels"]
        labs = hdbscan_labs.values
        unique_labs = hdbscan_labs.unique()
        nlabs = len(unique_labs)

        proj = np.array(list(indv_dfs[indv][viz_proj].values))
        sequence_ids = np.array(indv_dfs[indv]["syllables_sequence_id"])
        specs = np.invert(indv_dfs[indv].spectrogram.values)
        specs = np.where(specs == 255, 242, specs)  # grey

        pal = sns.color_palette(palette, n_colors=nlabs)

        # Projection scatterplot, labeled by cluster
        scatter_projections(
            projection=proj,
            labels=labs,
            color_palette=pal,
            alpha=0.60,
            s=7,
            facecolour=facecolour,
            show_legend=False,
            range_pad=0.1,
            ax=axes[0],
        )

        # Draw lines between consecutive syllables
        draw_projection_transitions(
            projections=proj,
            sequence_ids=indv_dfs[indv]["syllables_sequence_id"],
            sequence_pos=indv_dfs[indv]["syllables_sequence_pos"],
            cmap=plt.get_cmap("ocean"),
            facecolour=facecolour,
            range_pad=0.1,
            alpha=0.05,
            ax=axes[1],
        )

        # Plot inferred directed network
        plot_network_graph(
            labs,
            proj,
            sequence_ids,
            color_palette=pal,
            min_cluster_samples=10,
            min_connections=0.07,
            facecolour=facecolour,
            ax=axes[2],
        )

        # Plot examples of each cluster
        plot_example_specs(
            specs=specs,
            labels=labs,
            clusters_to_viz=unique_labs[unique_labs >= 0],  # do not show 'noisy' points
            custom_pal=pal,
            cmap=plt.cm.bone,
            nex=nlabs,
            line_width=3,
            ax=axes[3],
        )

        labels = string.ascii_uppercase[0 : len(axes)]

        for ax, labels in zip(axes, labels):
            bbox = ax.get_tightbbox(f.canvas.get_renderer())
            f.text(
                0.03,
                0.97,
                labels,
                fontsize=25,
                fontweight="bold",
                va="top",
                ha="left",
                transform=ax.transAxes,
            )

        fig_out = (
            FIGURE_DIR
            / YEAR
            / "ind_repertoires"
            / (indv + "_" + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + ".png")
        )
        ensure_dir(fig_out)
        plt.savefig(
            fig_out, dpi=300, bbox_inches="tight", pad_inches=0.3, transparent=False,
        )
        # plt.show()
        plt.close()
