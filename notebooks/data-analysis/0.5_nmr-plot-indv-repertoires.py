# %%

from src.avgn.utils.general import save_fig
from src.avgn.visualization.network_graph import plot_network_graph
from src.avgn.visualization.projections import scatter_projections, scatter_spec
from src.avgn.visualization.quickplots import draw_projection_plots, quad_plot_syllables
from src.avgn.visualization.spectrogram import draw_spec_set


from matplotlib import colors
import matplotlib.pyplot as plt

# %%
# ### plot each individual's repertoire


for indv in tqdm(indv_dfs.keys()):
    labs = indv_dfs[indv]["hdbscan_labels"].values
    proj = np.array(list(indv_dfs[indv]["umap_viz"].values))
    specs = indv_dfs[indv].spectrogram.values

    scatter_spec(
        proj,
        specs=indv_dfs[indv].spectrogram.values,
        column_size=8,
        # x_range = [-5.5,7],
        # y_range = [-10,10],
        pal_color="hls",
        color_points=False,
        enlarge_points=20,
        range_pad=0.1,
        figsize=(10, 10),
        scatter_kwargs={
            "labels": labs,
            "alpha": 0.60,
            "s": 5,
            "color_palette": pal,
            "show_legend": False,
        },
        matshow_kwargs={"cmap": plt.cm.Greys},
        line_kwargs={"lw": 1, "ls": "solid", "alpha": 0.11},
        draw_lines=True,
        border_line_width=0,
        facecolour=facecolour,
    )

    fig_out = (
        FIGURE_DIR
        / YEAR
        / "ind_repertoires"
        / (
            "{}_repertoire_".format(indv)
            + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            + ".png"
        )
    )
    ensure_dir(fig_out)
    plt.savefig(
        fig_out, dpi=300, bbox_inches="tight", pad_inches=0.3, transparent=False,
    )
    plt.show()
    plt.close()


# %%

# Plot: scatter, transitions, examples, per nestbox
# Saves figures to FIGURE_DIR / year / "ind_repertoires" / (indv + ".png")

quad_plot_syllables(indv_dfs, YEAR, "umap_viz", palette=pal, facecolour=facecolour)
