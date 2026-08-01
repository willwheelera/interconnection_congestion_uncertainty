"""Shared layout and loaders for the direct-* per-line plots.

The plot_direct_* scripts draw the same per-line panel grid -- congested lines first,
a two-entry header legend, inch-based header/footer/left bands, and sup-labels -- around
different panel bodies (box vs survivor curve).  panel_grid owns that scaffold so a layout
tweak lands in one place; each script supplies only its panel-draw callback, its labels, and
its legend handles.

Functions:
    load_direct_context  common parquet/meta loader for one case
    panel_grid           the unified per-line grid (returns fig, axs)
    box_legend           the direct-replicates + exact-truth legend handles
    box_panel            one line's box-per-budget panel
    box_draw             build the panel_grid callback for a box frame
"""

import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
import pyarrow.parquet as pq

from sampling.problem import REPO_ROOT, load_experiment, CASES
from figures.tail.inverse_cdf import COLORS
from figures.tail.line_exceed_strip import one_sigfig_yaxis


def load_direct_context(case, kind="samples"):
    """Common loader: (case_dir, frame, meta, target_cache_path, ngen, marginals p)."""
    case_dir = REPO_ROOT / "results" / case
    frame, meta = load_experiment(case_dir, kind)
    tc = case_dir / "target_cache.parquet"
    ngen = int(json.loads(pq.read_schema(tc).metadata[b"run_meta"])["ngen"])
    p = CASES[case].nominal_p()
    return case_dir, frame, meta, tc, ngen, p


def panel_grid(order_names, draw_panel, legend_handles, title, xlabel, ylabel, log_y=False):
    """Unified per-line panel grid; returns (fig, axs).

    order_names     ordered line ids, one panel each (congested first)
    draw_panel      callback(ax, name) that draws one line's panel
    legend_handles  figure legend entries
    log_y           shared log y axis (survivor curves) vs independent linear (boxes)
    """
    ncol = min(5, len(order_names))
    nrow = min(4, int(np.ceil(len(order_names) / ncol)))
    head, foot, lft = 0.95, 0.6, 0.7
    figw, figh = lft + ncol * 2.3 + 0.15, head + nrow * 2.0 + foot
    fig, axs = plt.subplots(nrow, ncol, figsize=(figw, figh), squeeze=False,
                            sharex=True, sharey=log_y)
    if log_y:
        axs[0, 0].set_yscale("log")
    for name, ax in zip(order_names, axs.ravel()):
        draw_panel(ax, name)
    for ax in axs.ravel()[len(order_names):]:
        ax.set_visible(False)

    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1 - 0.5 / figh), ncol=len(legend_handles),
               frameon=True, fontsize=8)
    fig.suptitle(title, y=1 - 0.22 / figh)
    fig.supxlabel(xlabel, y=0.45 * foot / figh)
    fig.supylabel(ylabel, x=0.3 * lft / figw)
    fig.subplots_adjust(left=lft / figw, right=0.99, bottom=foot / figh,
                        top=1 - head / figh, wspace=0.32, hspace=0.24)
    return fig, axs


def box_legend():
    """Legend handles shared by the two direct box plots."""
    return [Patch(facecolor=COLORS["direct"], alpha=0.55, label="direct replicates"),
            Line2D([0], [0], color="k", ls="--", label="exact truth")]


def box_panel(ax, sub, xorder, truth, name, ycol):
    """One line's panel: a direct box per budget cell plus the dashed exact truth."""
    sns.boxplot(data=sub, x="budget", y=ycol, order=xorder, color=COLORS["direct"],
                width=0.5, linewidth=0.8, fliersize=2, ax=ax,
                boxprops=dict(alpha=0.55), medianprops=dict(color="k"))
    ax.axhline(truth, color="k", ls="--", lw=1.0)
    ax.set_title(f"line {name}", fontsize=8, y=0.88)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=7)
    ax.tick_params(axis="x", length=0)
    hi = max(float(truth), float(sub[ycol].max()))          # keep truth line and fliers in frame
    ax.set_ylim(0, hi * 1.2 if hi > 0 else None)
    one_sigfig_yaxis(ax)


def box_draw(frame, xorder, truth_by_name, ycol):
    """Build the panel_grid callback that boxes one line's cells from a long frame."""
    def draw(ax, name):
        box_panel(ax, frame[frame["line"] == name], xorder, truth_by_name[name], name, ycol)
    return draw
