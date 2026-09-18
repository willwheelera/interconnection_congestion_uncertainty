"""Per-line estimate of a |mu_l| statistic from only the few most probable configurations.

The most-probable-config counterpart of line_exceed_strip_dist.py.  That figure estimates a
per-line statistic by direct sampling and spreads the sampling budget along the x axis; here
the estimate is deterministic and built from the top k configurations of top_config_mu.py --
the k most probable configs under each law -- with k = 1..n_configs along the x axis.  Every
(law, line) pair gets its own panel; within a panel two curves cumulate the estimate as k
grows and are read against that law's exact nominal-weighted truth (the dashed line):

    unweighted    the plain average of the k configs' values (no weights)
    prob-weighted the configs' values weighted by their nominal probabilities, renormalized

Two statistics are available and each is drawn as its own figure: the mean E[|mu_l|] and the
threshold probability P[|mu_l| > c].  A line's panels share one y scale so its estimate can be
read across the laws.

The grid has two orientations, as in line_exceed_strip_dist.  The default puts the laws down
the rows and the congested lines across the columns; `--transpose` swaps them, which is easier
to read when the line ids are long (as they are for the Texas cases).

Usage:
    uv run python -m src.figures.mu_distributions.top_config_estimate --case activsg2000_v12_lev3a3
    uv run python -m src.figures.mu_distributions.top_config_estimate --case activsg2000_v12_lev3a3 \
        --quantity exceed --threshold 500 --transpose
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.cases.registry import get_case
from src.sampling.problem import case_tag, create_problem
from src.sampling.distributions import DIST_NAMES, DIST_LABEL, case_law_names
from src.figures.style import DIST_COLOR
from src.figures.mu_distributions._line_stats import exceedance_prob, mean_abs_mu, select_lines
from src.figures.mu_distributions._panel_layout import (FONT, LABEL_IN, LEGEND_IN, TITLE_IN,
                                           margin_grid, one_sigfig_yaxis, panel_title)

ROW_ASPECT = 1.0                 # transposed panels: width / height
LEFT_WIDE_IN = 0.66              # transposed left margin (in): room for the long line labels
TOP_WIDE_IN = 0.30              # transposed top margin (in): suptitle plus the column titles
BOTTOM_WIDE_IN = 0.58           # transposed bottom margin (in): the x label above the legend band
COLS_LEFT = 0.12                 # default grid left margin (frac): room for the law row labels

QUANTITIES = {
    "mean": dict(transform=lambda v, thr: v,
                 truth=lambda mu, w, thr: mean_abs_mu(mu, w),
                 ylabel=lambda thr: r"$E[\,|\mu_l|\,]$",
                 title=lambda thr: r"$E[\,|\mu_l|\,]$",
                 shared=False),   # per-line y range: the means span decades across lines
    "exceed": dict(transform=lambda v, thr: (v > thr).astype(float),
                   truth=lambda mu, w, thr: exceedance_prob(mu, w, thr),
                   ylabel=lambda thr: rf"$P[\,|\mu_l| > {thr:g}\,]$",
                   title=lambda thr: rf"$P[\,|\mu_l| > {thr:g}\,]$",
                   shared=True),  # one y range: probabilities are comparable across lines
}


def top_ranked(problem, n):
    """The n most probable config ids under the law (descending) and their nominal weights."""
    ids = np.argsort(-problem.w)[:n]
    return ids, problem.w[ids]


def topk_estimates(v_top, w_top):
    """Cumulative unweighted and prob-weighted estimates over the first 1..K ranked configs.

    v_top and w_top are one line's values and the configs' nominal weights, ranked most
    probable first.  The unweighted curve averages the first k values; the prob-weighted curve
    is their nominal-weight average, renormalized to the mass of the k configs actually used.
    """
    k = np.arange(1, len(v_top) + 1)
    return k, np.cumsum(v_top) / k, np.cumsum(w_top * v_top) / np.cumsum(w_top)


def precompute_estimates(mu, tops, order_idx, dists, transform, threshold):
    """est[d][l] = (k, unweighted, prob-weighted) for each law d and ordered line index l."""
    est = {}
    for d in dists:
        ids, w_top = tops[d]
        est[d] = {l: topk_estimates(transform(mu[ids, l], threshold), w_top) for l in order_idx}
    return est


def line_ranges(est, truths, order_idx, dists):
    """(lo, hi) y range per ordered line index over both estimators, every law, and the truths."""
    ranges = {}
    for l in order_idx:
        vals = [truths[d][l] for d in dists]
        for d in dists:
            _, unw, wtd = est[d][l]
            vals.extend(unw.tolist() + wtd.tolist())
        lo, hi = float(min(vals)), float(max(vals))
        if (hi-lo) < 0.1 * (hi+lo)/2:
            m = (hi+lo)/2
            lo, hi = 0.9*m, 1.1*m
        pad = 0.08 * (hi - lo) if hi > lo else max(1.0, 0.08 * abs(hi))
        ranges[l] = (max(0.0, lo - pad), hi + pad)
    return ranges


def share_ranges(ranges):
    """Collapse per-line (lo, hi) ranges to one common range shared by every line."""
    lo = min(r[0] for r in ranges.values())
    hi = max(r[1] for r in ranges.values())
    return {l: (lo, hi) for l in ranges}


def estimate_panel(ax, est_dl, truth, color, kmax, ylim):
    """One (law, line) panel: the two cumulative estimate curves and the exact truth line."""
    k, unw, wtd = est_dl
    ax.plot(k, unw, color=color, ls="-", marker="o", ms=1.5, lw=0.9, zorder=3)
    ax.plot(k, wtd, color=color, ls=":", marker="s", ms=1.5, lw=0.9, zorder=3)
    ax.axhline(truth, color="k", ls="--", lw=0.6, zorder=4)
    ax.set_xlim(0.5, kmax + 0.5)
    ax.set_xticks([5, 10])#range(1, kmax + 1))
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=FONT["tick"], pad=1.5)
    ax.tick_params(axis="x", length=0)
    one_sigfig_yaxis(ax, steps=(1, 2, 5, 10), sci_above=1000)


def estimate_legend():
    """Legend handles: the unweighted and prob-weighted estimators and the exact truth."""
    return [Line2D([0], [0], color="0.3", ls="-", marker="o", ms=3, label="unweighted"),
            Line2D([0], [0], color="0.3", ls=":", marker="s", ms=3, label="weighted"),
            Line2D([0], [0], color="k", ls="--", lw=0.8, label="exact")]


def decorate_grid(fig, title, ylabel, supy_x, legend_bottom=False):
    """Shared legend, suptitle, and shared y label for either grid orientation."""
    h = fig.get_figheight()
    loc, anchor = (("lower center", (0.5, LABEL_IN / h)) if legend_bottom
                   else ("upper center", (0.5, 1 - LEGEND_IN / h)))
    fig.legend(handles=estimate_legend(), loc=loc, bbox_to_anchor=anchor, ncol=3,
               frameon=True, fontsize=FONT["legend"], handletextpad=0.3, columnspacing=0.8,
               borderpad=0.3, handlelength=1.6)
    #fig.suptitle(title, y=1 - TITLE_IN / h, fontsize=FONT["suptitle"])
    fig.supylabel(ylabel, x=supy_x, fontsize=FONT["axlabel"])


def figure_lines_as_cols(est, dists, order_idx, order_names, truths, ranges, ylabel, title, kmax):
    """Grid with laws down the rows and congested lines across the columns (the default)."""
    fig, axs = margin_grid(len(dists), len(order_names), left=COLS_LEFT)
    for i, d in enumerate(dists):
        for j, l in enumerate(order_idx):
            estimate_panel(axs[i, j], est[d][l], float(truths[d][l]), DIST_COLOR[d], kmax,
                           ranges[l])
            if i == 0:
                panel_title(axs[i, j], f"line {order_names[j]}")
            if j == 0:
                axs[i, j].set_ylabel(DIST_LABEL[d], fontsize=FONT["panel_title"], labelpad=2)
    fig.supxlabel("No. top configs averaged",
                  y=LABEL_IN / fig.get_figheight(), fontsize=FONT["axlabel"])
    decorate_grid(fig, title, ylabel, supy_x=0.012)
    return fig


def figure_lines_as_rows(est, dists, order_idx, order_names, truths, ranges, ylabel, title, kmax):
    """Transposed grid: congested lines down the rows, laws across the columns."""
    fig, axs = margin_grid(len(order_idx), len(dists), sharey="row", aspect=ROW_ASPECT,
                           wspace=0.1, w=3)
    for i, l in enumerate(order_idx):
        for j, d in enumerate(dists):
            estimate_panel(axs[i, j], est[d][l], float(truths[d][l]), DIST_COLOR[d], kmax,
                           ranges[l])
            if i == 0:
                axs[i, j].set_title(DIST_LABEL[d], fontsize=FONT["panel_title"], pad=3)
            if j == 0:
                axs[i, j].set_ylabel(f"{order_names[i]}", fontsize=FONT["panel_title"])
    fig.subplots_adjust(left=LEFT_WIDE_IN / fig.get_figwidth(),
                        top=1 - TOP_WIDE_IN / fig.get_figheight(),
                        bottom=BOTTOM_WIDE_IN / fig.get_figheight())
    fig.supxlabel("No. top configs averaged",
                  y=(LABEL_IN + 0.26) / fig.get_figheight(), fontsize=FONT["axlabel"])
    decorate_grid(fig, title, ylabel, supy_x=LABEL_IN / fig.get_figwidth(), legend_bottom=True)
    return fig


def build_quantity_figure(case, quantity, mu, tops, problems, order_idx, order_names,
                          threshold, kmax, transpose):
    """One figure for a single statistic; write it under the case's results dir and return it."""
    spec = QUANTITIES[quantity]
    dists = list(problems)
    truths = {d: spec["truth"](mu, problems[d].w, threshold) for d in dists}
    est = precompute_estimates(mu, tops, order_idx, dists, spec["transform"], threshold)
    ranges = line_ranges(est, truths, order_idx, dists)
    if spec["shared"]:
        ranges = share_ranges(ranges)
    title = f"{case.name.split('_')[0]}: {spec['title'](threshold)} from the most probable configs"
    ylabel = spec["ylabel"](threshold)
    build = figure_lines_as_rows if transpose else figure_lines_as_cols
    fig = build(est, dists, order_idx, order_names, truths, ranges, ylabel, title, kmax)

    thr = f"_thr{int(threshold)}" if quantity == "exceed" else ""
    rows = "_linerows" if transpose else ""
    out = case.results_dir / f"top_config_estimate_{quantity}{thr}{rows}_{case_tag(case.name)}.pdf"
    fig.savefig(out)
    print(f"saved -> {out}")
    return fig


def run(case, dists=DIST_NAMES, quantities=("mean", "exceed"), threshold=100.0,
        n_configs=5, n_lines=5, transpose=False):
    """One top-config estimate figure per requested statistic; returns them in order.

    The |mu_l| table and the top-config ids are built once and shared by both statistics; the
    lines follow the shared `select_lines` selection (descending E[|mu_l|] under the baseline
    law, no filter), so a line sits in the same panel as elsewhere.
    """
    c = get_case(case)
    dists = case_law_names(c, dists)
    problems = {d: create_problem(c, d) for d in dists}
    mu, line_names = problems[dists[0]].mu_table()
    tops = {d: top_ranked(problems[d], n_configs) for d in dists}
    order_idx, order_names = select_lines(mu, line_names, problems[dists[0]].w, n=n_lines)
    return [build_quantity_figure(c, q, mu, tops, problems, order_idx, order_names,
                                  threshold, n_configs, transpose) for q in quantities]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="activsg2000_v12_lev3a3")
    parser.add_argument("--dists", default=",".join(DIST_NAMES))
    parser.add_argument("--quantity", default="both", choices=["mean", "exceed", "both"])
    parser.add_argument("--threshold", type=float, default=100.0,
                        help="exceedance level c for the threshold-probability statistic")
    parser.add_argument("--n-configs", type=int, default=10)
    parser.add_argument("--n-lines", type=int, default=5)
    parser.add_argument("--transpose", action="store_true")
    args = parser.parse_args()

    quantities = ("mean", "exceed") if args.quantity == "both" else (args.quantity,)
    run(args.case, args.dists.split(","), quantities, args.threshold,
        args.n_configs, args.n_lines, transpose=args.transpose)
    plt.show()


if __name__ == "__main__":
    main()
