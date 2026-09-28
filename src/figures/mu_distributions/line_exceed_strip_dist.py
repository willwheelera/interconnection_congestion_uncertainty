"""Per-line jittered strip of the tail probability P[|mu_l| > c] across the four nominal laws.

The nominal-law transpose of line_exceed_strip.py.  That figure fixes one law and spreads
the sampling budgets along the x axis, one figure per law; here direct sampling is the only
method and all four nominal laws (Bernoulli, MRF-A, MRF-co, MRF-sub) share one figure.  Every
(law, line) pair gets its own panel: within a panel the sampling budget is the x axis and each
budget's replicate estimates are jittered points, read against that law's exact nominal-weighted
truth (the dashed line).  A line's panels share one y scale so its estimate can be read across
the four laws.

The grid has two orientations.  The default puts the laws down the rows and the congested lines
across the columns, matching the other per-line grids.  `--transpose` swaps them -- lines down
the rows and laws across the columns -- which is easier to read when the line ids are long.

Line order is by descending E[|mu_l|] under the baseline (Bernoulli) law -- the shared order
every per-line grid uses -- kept to lines that congest under some law.

Usage:
    uv run python -m src.figures.mu_distributions.line_exceed_strip_dist --case activsg2000_v12_lev3a3
    uv run python -m src.figures.mu_distributions.line_exceed_strip_dist --case activsg2000_v12_lev3a3 \
        --threshold 500 --budgets 640,2560
    uv run python -m src.figures.mu_distributions.line_exceed_strip_dist --case activsg2000_v12_lev3a3 \
        --transpose
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.sampling.problem import case_tag, create_problem
from src.sampling.distributions import DIST_NAMES, DIST_LABEL
from src.figures.style import DIST_COLOR, budget_colors
from src.figures.mu_distributions._line_stats import exceedance_prob, select_lines
from src.figures.mu_distributions._experiment_reads import direct_samples, direct_slice
from src.figures.mu_distributions._panel_layout import (FONT, LABEL_IN, LEGEND_IN, TITLE_IN,
                                           margin_grid, one_sigfig_yaxis, panel_title)

POINT_SIZE = 1.5                  # marker diameter in points
JITTER = 0.40                     # horizontal jitter of replicate points within a budget slot
JITTER_SEED = 0                   # fixed, so re-running moves no point
HEADROOM = 0.0                    # fraction of the widest column's range padded onto each side
ROW_ASPECT = 1.                   # transposed panels: wider than tall, to squeeze the vertical
LEFT_WIDE_IN = 0.66               # transposed left margin (in): room for the long "line" labels
TOP_WIDE_IN = 0.30                # transposed top margin (in): suptitle plus the column titles
BOTTOM_WIDE_IN = 0.52             # transposed bottom margin (in): the budget legend band


def direct_exceed_frame(samples, mu, line_names, dists, budgets, threshold):
    """Long per-(dist, budget, rep, line) direct estimate of P[|mu_l| > threshold].

    Direct draws are target-independent, so the estimate is the unbiased sum of the draws'
    importance weights on the exceeding configs.  `mu` is the law-independent |mu_l| table;
    only the sampled ids and their weights change from law to law.
    """
    rows = []
    for dist in dists:
        sl = direct_slice(samples, dist, budgets)
        for (budget, rep), g in sl.groupby(["budget", "rep"], sort=True):
            w, ids = g["weight"].to_numpy(), g["id"].to_numpy()
            prob = (w[:, None] * (mu[ids] > threshold)).sum(0)
            rows.append(pd.DataFrame({"dist": dist, "budget": budget, "rep": rep,
                                      "line": line_names, "prob": prob}))
    return pd.concat(rows, ignore_index=True)


def law_truths(c, dists, mu, threshold):
    """Per-line exact P[|mu_l| > threshold] for each law, and the per-line peak across laws."""
    truths = {d: exceedance_prob(mu, create_problem(c, d).w, threshold) for d in dists}
    peak = np.max(np.stack([truths[d] for d in dists]), axis=0)
    return truths, peak


def strip_panel(ax, sub, budgets, truth_p, bcolors, ylim, rng):
    """One (law, line) panel: replicate points jittered within each budget slot, plus truth.

    `bcolors` maps each budget to its point color, so a panel stacking every budget can tell
    them apart."""
    slot_of = {b: i for i, b in enumerate(budgets)}
    for b in budgets:
        pts = sub[sub["budget"] == b]
        x = slot_of[b] + rng.uniform(-JITTER, JITTER, len(pts))
        ax.scatter(x, pts["prob"].to_numpy(), s=POINT_SIZE ** 2, color=bcolors[b],
                   alpha=0.6, linewidth=0, zorder=3, rasterized=False)
    ax.axhline(truth_p, color="k", ls="--", lw=0.6, zorder=4)
    n = len(budgets)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xticks([])
    #ax.set_xticks(range(n))
    #ax.set_xticklabels([str(b) for b in budgets])
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=FONT["tick"], pad=1.5)
    ax.tick_params(axis="x", length=0)
    one_sigfig_yaxis(ax)


def line_ylims(frame, truths, order_names, name_index, dists):
    """Per-line y limits, one common height for every panel, whatever the grid orientation.

    Each line's panels share one range so its estimate reads on a single scale across the four
    laws; that range is centered on the line's own data, but ymax - ymin is a single shared value
    across the whole figure: the widest line's data range plus HEADROOM on each side.  A line that
    would fall below zero is shifted up to sit on zero, which keeps the shared height intact.
    """
    ranges = {}
    for name in order_names:
        vals = frame[frame["line"] == name]["prob"].to_numpy()
        vals = np.append(vals, [truths[d][name_index[name]] for d in dists])
        ranges[name] = (float(vals.min()), float(vals.max()))
    height = (1.0 + 2.0 * HEADROOM) * max(hi - lo for lo, hi in ranges.values()) + 1e-12
    ylims = {}
    for name, (lo, hi) in ranges.items():
        bottom = max(0.0, 0.5 * (lo + hi) - 0.5 * height)
        ylims[name] = (bottom, bottom + height)
    return ylims


def decorate_grid(fig, handles, title, supy_x, legend_title=None, legend_bottom=False):
    """Shared legend, suptitle, and shared y label for either grid orientation.

    `handles` are the figure legend entries (law colors for the default grid, budget colors for
    the transposed one); `legend_title` heads the legend, e.g. the budget key that replaces the
    x label; `legend_bottom` places the legend along the bottom edge rather than below the
    suptitle; `supy_x` is the figure-fraction x of the shared y label."""
    h = fig.get_figheight()
    loc, anchor = (("lower center", (0.5, LABEL_IN / h)) if legend_bottom
                   else ("upper center", (0.5, 1 - LEGEND_IN / h)))
    fig.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, title=legend_title,
               ncol=len(handles), frameon=True, fontsize=FONT["legend"],
               title_fontsize=FONT["axlabel"], handletextpad=0.3, columnspacing=0.8,
               borderpad=0.3, handlelength=1.4)
    fig.suptitle(title, y=1 - TITLE_IN / h, fontsize=FONT["suptitle"])
    fig.supylabel(r"$\hat{P}[\,|\mu_l| > c\,]$", x=supy_x, fontsize=FONT["axlabel"])


def figure_lines_as_cols(frame, dorder, order_names, name_index, truths, budgets, title):
    """Grid with laws down the rows and congested lines across the columns (the default)."""
    fig, axs = margin_grid(len(dorder), len(order_names))
    rng = np.random.default_rng(JITTER_SEED)
    ylims = line_ylims(frame, truths, order_names, name_index, dorder)
    for i, d in enumerate(dorder):
        sub_d = frame[frame["dist"] == d]
        bcolors = {b: DIST_COLOR[d] for b in budgets}
        for j, name in enumerate(order_names):
            ax = axs[i, j]
            strip_panel(ax, sub_d[sub_d["line"] == name], budgets,
                        float(truths[d][name_index[name]]), bcolors, ylims[name], rng)
            if i == 0:
                panel_title(ax, f"line {name}")
            if j == 0:
                ax.set_ylabel(DIST_LABEL[d], fontsize=FONT["panel_title"])
    handles = [Line2D([0], [0], marker="o", ls="", ms=2.5, color=DIST_COLOR[d],
                      label=DIST_LABEL[d]) for d in dorder]
    handles.append(Line2D([0], [0], color="k", ls="--", label="exact truth"))
    fig.supxlabel("sampling budget", y=LABEL_IN / fig.get_figheight(), fontsize=FONT["axlabel"])
    decorate_grid(fig, handles, title, supy_x=LABEL_IN / fig.get_figwidth() - 0.02)
    return fig


def figure_lines_as_rows(frame, dorder, order_names, name_index, truths, budgets, title):
    """Transposed grid: congested lines down the rows, laws across the columns.

    Rows share one y scale so a line's tick labels are drawn once (sharey='row'); the law names
    title the columns from outside the box; the points are colored by sampling budget, since the
    law is now the column and every panel stacks all budgets."""
    fig, axs = margin_grid(len(order_names), len(dorder), sharey="row", aspect=ROW_ASPECT, wspace=0.1, w=3)
    rng = np.random.default_rng(JITTER_SEED)
    ylims = line_ylims(frame, truths, order_names, name_index, dorder)
    bcolors = budget_colors(budgets)
    for i, name in enumerate(order_names):
        for j, d in enumerate(dorder):
            ax = axs[i, j]
            sub = frame[(frame["dist"] == d) & (frame["line"] == name)]
            strip_panel(ax, sub, budgets,
                        float(truths[d][name_index[name]]), bcolors, ylims[name], rng)
            if i == 0:
                ax.set_title(DIST_LABEL[d], fontsize=FONT["panel_title"], pad=3)
            if j == 0:
                ax.set_ylabel(f"{name}", fontsize=FONT["panel_title"])
    handles = [Line2D([0], [0], marker="o", ls="", ms=2.5, color=bcolors[b], label=str(b))
               for b in budgets]
    handles.append(Line2D([0], [0], color="k", ls="--", label="exact truth"))
    fig.subplots_adjust(left=LEFT_WIDE_IN / fig.get_figwidth(),
                        top=1 - TOP_WIDE_IN / fig.get_figheight(),
                        bottom=BOTTOM_WIDE_IN / fig.get_figheight())
    decorate_grid(fig, handles, title, supy_x=LABEL_IN / fig.get_figwidth(),
                  legend_title="sampling budget", legend_bottom=True)
    return fig


def run(case, dists=DIST_NAMES, budgets=None, threshold=100.0, transpose=False):
    """One direct-only strip figure over the nominal laws; returns it.

    The |mu_l| table and the sample frame are law-independent and built once; each law's exact
    truth is per law, and the lines follow the shared `select_lines` selection: descending
    E[|mu_l|] under the baseline (Bernoulli) law, no filter.  `transpose` swaps the grid axes
    (lines down the rows, laws across the columns).
    """
    c, samples, meta = direct_samples(case)
    case, case_dir = c.name, c.results_dir
    budgets = budgets or meta["budgets"]
    baseline = create_problem(c, dists[0])
    mu, line_names = baseline.mu_table()
    truths, _ = law_truths(c, dists, mu, threshold)
    idx, order_names = select_lines(mu, line_names, baseline.w, n=3)
    name_index = {name: i for name, i in zip(order_names, idx)}
    frame = direct_exceed_frame(samples, mu, line_names, dists, budgets, threshold)

    title = (f"{case.split('_')[0]}: direct  "
             rf"$P[\,|\mu_l| > {threshold:g}\,]$  |  {meta['n_rep']} reps")
    build = figure_lines_as_rows if transpose else figure_lines_as_cols
    fig = build(frame, dists, order_names, name_index, truths, budgets, title)
    suffix = "_linerows" if transpose else ""
    out = case_dir / f"line_exceed_strip_dist_thr{int(threshold)}{suffix}_{case_tag(case)}.pdf"
    fig.savefig(out)
    print(f"saved -> {out}")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="activsg2000_v12_lev3a3")
    parser.add_argument("--dists", default=",".join(DIST_NAMES))
    parser.add_argument("--budgets", default=None)
    parser.add_argument("--threshold", type=float, default=100.0)
    parser.add_argument("--transpose", action="store_true")
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else None
    run(args.case, args.dists.split(","), budgets, args.threshold, transpose=args.transpose)
    plt.show()


if __name__ == "__main__":
    main()
