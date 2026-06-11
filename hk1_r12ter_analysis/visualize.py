from itertools import product

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns

from .config import logger


def plot_correlations_vs_pvalues(
    data, correlation_statistic, neg_log10_pvalue, ax=None, cax=None, **kwargs
):
    """Plot the -log10(p-value) as a function of the correlation statistic."""

    ylim = kwargs.get("yaxis_kwargs", {}).get("lim")
    title = kwargs.get("title")
    fontdict = kwargs.get("fontdict", dict(fontsize="large"))

    ax = data.plot(
        kind="scatter",
        x=correlation_statistic,
        y=neg_log10_pvalue,
        ax=ax,
        c=neg_log10_pvalue,
        xlim=kwargs.get("xaxis_kwargs", {}).get("lim"),
        ylim=ylim,
        cmap=kwargs.get("cmap", "viridis"),
        vmin=kwargs.get("vmin", ylim[0] if ylim is not None else ylim),
        vmax=kwargs.get("vmax", ylim[1] if ylim is not None else ylim),
        marker=kwargs.get("marker", "o"),
        edgecolor=kwargs.get("edgecolor", "black"),
        linewidth=kwargs.get("linewidth", 1),
        s=kwargs.get("s", 50),
        zorder=kwargs.get("zorder", 5),
        colorbar=False,
        legend=None,
    )
    if kwargs.get("grid_kwargs"):
        ax.xaxis.grid(**kwargs.get("grid_kwargs"))
        ax.yaxis.grid(**kwargs.get("grid_kwargs"))

    if cax is None:
        cax = make_axes_locatable(ax).append_axes("left", size="5%", pad="0%")

    if kwargs.get("title"):
        title, title_fontdict = (
            (kwargs.get("title"), fontdict)
            if isinstance(kwargs.get("title"), str)
            else kwargs.get("title")
        )
        ax.set_title(title, fontdict=title_fontdict)

    for xy in ["x", "y"]:
        axis_kwargs = kwargs.get(f"{xy}axis_kwargs")
        if not axis_kwargs:
            continue
        getattr(ax, f"set_{xy}label")(
            axis_kwargs.get("label"), fontdict=axis_kwargs.get("fontdict", fontdict)
        )

        major_tick_multiple = axis_kwargs.get("major_tick_multiple")
        getattr(ax, f"{xy}axis").set_major_locator(mpl.ticker.MultipleLocator(major_tick_multiple))
        if axis_kwargs.get("minor_ticks"):
            getattr(ax, f"{xy}axis").set_minor_locator(
                mpl.ticker.MultipleLocator(major_tick_multiple, offset=major_tick_multiple / 2)
            )
        ax.tick_params(
            axis=xy,
            labelsize=axis_kwargs.get("tick_labelsize", fontdict["size"]),
            which="both",
            labelbottom=True,
            length=kwargs.get("tick_length", 5),
        )

    if cax:
        ax.get_figure().colorbar(
            mpl.cm.ScalarMappable(
                norm=mpl.colors.Normalize(vmin=ax.get_ylim()[0], vmax=ax.get_ylim()[-1]),
                cmap="viridis",
            ),
            cax=cax,
            orientation="vertical",
        )
        cax.yaxis.set_label_position("left")
        cax.set_yticks(ax.get_yticks(minor=False), minor=False)
        if kwargs.get("yaxis_kwargs", {}).get("minor_ticks"):
            cax.set_yticks(ax.get_yticks(minor=True), minor=True)

        cax.tick_params(
            axis="y",
            reset=True,
            labelsize=kwargs.get("yaxis_kwargs", {}).get(
                "tick_labelsize", fontdict.get("fontsize")
            ),
            labelright=False,
            right=False,
            which="both",
            length=kwargs.get("tick_length", 5),
        )
        cax.set_ylabel(
            ax.get_ylabel(), fontdict=kwargs.get("yaxis_kwargs", {}).get("fontdict", fontdict)
        )
        cax.set_ylim(ax.get_ylim())

        # Remove ylabel and yticks
        ax.set_ylabel(None)
        ax.tick_params(axis="y", labelleft=False, length=0, which="both")

    return ax, cax


def plot_violin_for_snp_phenotype(data, rsid, variable, subgroup, split=False, ax=None, **kwargs):
    """Create violin plots with boxplots and stripplots for metadata stratified by SNP phenotype."""
    # Applied to all
    linewidth = kwargs.get("linewidth", 1)
    min_zorder = kwargs.get("min_zorder", 3)

    violinplot_kwargs = kwargs.get("violinplot_kwargs", {}).copy()
    stripplot_kwargs = kwargs.get("stripplot_kwargs", {}).copy()
    boxplot_kwargs = kwargs.get("boxplot_kwargs", {}).copy()
    pointplot_kwargs = kwargs.get("pointplot_kwargs", {}).copy()
    legend_kwargs = kwargs.get("legend_kwargs", {}).copy()

    width = 0.8 if not split else 0.9
    hue = subgroup if bool(subgroup) else rsid
    ax = sns.violinplot(
        data=data,
        x=rsid,
        y=variable,
        ax=ax,
        hue=hue,
        edgecolor="xkcd:black",
        legend=bool(legend_kwargs),
        inner=None,
        split=split if bool(subgroup) else False,
        width=width,
        bw_adjust=violinplot_kwargs.get("bw_adjust", 1),
        linewidth=violinplot_kwargs.get("linewidth", linewidth),
        zorder=violinplot_kwargs.get("zorder", min_zorder),
        cut=violinplot_kwargs.get("cut", 2),
    )
    if legend_kwargs:
        handles, labels = ax.get_legend_handles_labels()
        labels = legend_kwargs.pop("labels", labels)
        labels = [labels[x] if isinstance(labels, dict) else x for x in labels]
        ax.legend(handles=handles, labels=labels, **legend_kwargs)

    if split:
        ax = sns.stripplot(
            data=data,
            x=rsid,
            y=variable,
            ax=ax,
            color=stripplot_kwargs.get("color", "xkcd:grey"),
            zorder=stripplot_kwargs.get("zorder", min_zorder),
            jitter=0.04,
            dodge=False,
            legend=False,
            s=stripplot_kwargs.get("s", 2),
        )
        ax = sns.boxplot(
            data=data,
            x=rsid,
            y=variable,
            ax=ax,
            legend=False,
            gap=width,
            width=width,
            color=boxplot_kwargs.get("color", "xkcd:lightgrey"),
            linecolor=boxplot_kwargs.get("linecolor", "xkcd:black"),
            linewidth=boxplot_kwargs.get("linewidth", linewidth),
            zorder=boxplot_kwargs.get("zorder", min_zorder),
            showfliers=False,
            capwidths=width / 4,
            capprops=boxplot_kwargs.get("capprops"),
            whiskerprops=boxplot_kwargs.get("whiskerprops"),
        )
        medians = data.groupby([rsid], as_index=False)[variable].median()
        ax = sns.pointplot(
            data=medians,
            x=rsid,
            y=variable,
            marker="o",
            color="black",
            linestyle="None",
            legend=False,
            zorder=pointplot_kwargs.get("zorder", min_zorder + 2),
            markersize=pointplot_kwargs.get("markersize", None),
        )
    else:
        # Set the seed for reproducibility
        np.random.seed(stripplot_kwargs.get("seed", 4))
        ax = sns.stripplot(
            data=data,
            x=rsid,
            y=variable,
            hue=hue,
            ax=ax,
            zorder=stripplot_kwargs.get("zorder", min_zorder),
            jitter=0.05 if bool(subgroup) else True,
            dodge=bool(subgroup),
            legend=False,
            s=stripplot_kwargs.get("s", 2),
            palette=stripplot_kwargs.get("palette", {x: "xkcd:black" for x in data[hue].unique()}),
        )
        ax = sns.boxplot(
            data=data,
            x=rsid,
            y=variable,
            ax=ax,
            hue=hue,
            legend=False,
            dodge=bool(subgroup),
            gap=width if bool(subgroup) else None,
            width=width / 4 if not bool(subgroup) else width,
            linecolor=boxplot_kwargs.get("linecolor", "xkcd:black"),
            linewidth=boxplot_kwargs.get("linewidth", linewidth),
            zorder=boxplot_kwargs.get("zorder", min_zorder),
            showfliers=False,
            capwidths=width / 4,
            capprops=boxplot_kwargs.get("capprops"),
            whiskerprops=boxplot_kwargs.get("whiskerprops"),
        )

        medians = data.groupby([rsid, subgroup] if subgroup else [rsid], as_index=False).median()
        ax = sns.pointplot(
            data=medians,
            x=rsid,
            y=variable,
            hue=hue,
            marker="o",
            dodge=0.4 if bool(subgroup) else None,
            linestyle="None",
            legend=False,
            zorder=pointplot_kwargs.get("zorder", min_zorder + 2),
            palette=pointplot_kwargs.get("palette", {x: "xkcd:black" for x in data[hue].unique()}),
            markersize=pointplot_kwargs.get("markersize", None),
        )

    palette = kwargs.get("palette")
    if palette:
        groups = data[rsid].unique()
        subgroups = data[subgroup].unique() if bool(subgroup) else []
        if split:
            for i, violin in enumerate(ax.findobj(mpl.collections.PolyCollection)):
                violin.set_facecolor(palette.get((groups[int(i / 2)], subgroups[i % 2])))
        else:
            violins = ax.findobj(mpl.collections.PolyCollection)
            # Reorder boxes to match violins
            if bool(subgroup):
                boxes = [x for x in (range(1, (len(groups) * len(subgroups) + 1)))]
                boxes = [x for x in boxes if x % 2 == 1] + [x for x in boxes if x % 2 == 0]
            else:
                boxes = [x for x in (range(1, len(groups) + 1))]

            boxes = [x[1] for x in sorted(zip(boxes, ax.findobj(mpl.patches.PathPatch)))]
            for i, (violin, box) in enumerate(zip(violins, boxes)):
                if subgroups and all([x in palette for x in product(groups, subgroups)]):
                    palette_key = (groups[int(np.floor(i / 2))], subgroups[i % len(subgroups)])
                elif groups and all([x in palette for x in groups]):
                    palette_key = groups[i]
                elif subgroups and all([x in palette for x in subgroups]):
                    palette_key = subgroups[i % len(subgroups)]
                else:
                    logger.warning(
                        "Palette keys do not match groups or subgroups. Skipping color assignment for violins and boxes."
                    )
                    continue

                violin.set_facecolor(palette[palette_key])
                box.set_facecolor(palette[palette_key])

        if ax.legend_:
            for i, rect in enumerate(ax.legend_.findobj(mpl.patches.Rectangle)):
                if subgroups and all([x in palette for x in product(groups, subgroups)]):
                    palette_key = (groups[int(np.floor(len(groups) - 1))], rect.get_label())
                else:
                    palette_key = rect.get_label()
                    palette_key = float(palette_key) if palette_key.isdigit() else palette_key

                rect.set_facecolor(palette[palette_key])

    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())

    return ax


def plot_hemolysis_pvalues_for_snp(data, x, y, hue, ax=None, **kwargs):
    """Create a point plot of hemolysis correlation p-values for given SNPs."""
    ax = sns.scatterplot(
        data,
        x=x,
        y=y,
        ax=ax,
        zorder=4,
        hue=hue,
        palette=kwargs.get("palette"),
        edgecolor=kwargs.get("edgecolor", "xkcd:black"),
        s=kwargs.get("s", 100),
    )

    fontdict = kwargs.get("fontdict")
    for xy in ["x", "y"]:
        axis_kwargs = kwargs.get(f"{xy}axis_kwargs")
        getattr(ax, f"set_{xy}lim")(axis_kwargs.get("lim", getattr(ax, f"get_{xy}lim")()))
        if not axis_kwargs:
            continue
        getattr(ax, f"set_{xy}label")(
            axis_kwargs.get("label"), fontdict=axis_kwargs.get("fontdict", fontdict)
        )
        major_tick_multiple = axis_kwargs.get("major_tick_multiple")
        if major_tick_multiple:
            getattr(ax, f"{xy}axis").set_major_locator(
                mpl.ticker.MultipleLocator(major_tick_multiple)
            )
            if axis_kwargs.get("minor_ticks"):
                getattr(ax, f"{xy}axis").set_minor_locator(
                    mpl.ticker.MultipleLocator(major_tick_multiple, offset=major_tick_multiple / 2)
                )
        ax.tick_params(
            axis=xy,
            labelsize=axis_kwargs.get("tick_labelsize", fontdict["size"]),
            which="both",
            labelbottom=True,
            length=kwargs.get("tick_length", 5),
            rotation=axis_kwargs.get("tick_rotation"),
        )
        for ticklabel in getattr(ax, f"get_{xy}ticklabels")():
            ticklabel.set_rotation_mode("anchor")
            ticklabel.set_horizontalalignment(axis_kwargs.get("ha", "center"))

        getattr(ax, f"{xy}axis").grid(
            visible=bool(axis_kwargs.get("which")) or bool(kwargs.get("grid_kwargs")),
            which=axis_kwargs.get("which", "major"),
            **kwargs.get("grid_kwargs", {}),
        )

    ax.legend(**kwargs.get("legend_kwargs", {}))
    return ax
