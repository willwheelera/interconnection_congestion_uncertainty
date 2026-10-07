"""Per-line direct estimate of P[|mu_l| > c] with its standard error, across the nominal laws.

The points-with-error-bars twin of line_exceed_strip_dist.py.  That figure scatters every
replicate as its own jittered point; here a single direct run per budget carries one estimate
with the standard error that run reports for itself.  Direct sampling estimates each congested
line's tail probability P[|mu_l| > c] as the share of the run's draws that exceed c, and the
error bar is that indicator's sample standard deviation over sqrt(N) -- which is exactly the
Bernoulli standard error sqrt(p_hat (1 - p_hat) / N) of the proportion.  The bar shrinks as
1/sqrt(N) with the budget, so the panel shows the estimate converging onto the exact
nominal-weighted truth (the dashed line).

Direct draws are target-independent and their importance weights are uniform (1/N), so the
estimate is the plain share of the run's N draws with |mu_l| > c and the standard error is the
exceedance indicator's sample standard deviation over sqrt(N).  One run (the first replicate) is
used per budget, so the error bar stays tied to that budget's sample size.

Every (law, line) pair gets its own panel.  Probabilities span decades across lines, so each line
keeps its own y scale (its estimate reads across the four laws) but the lines do not share a scale
with each other.

The grid has two orientations.  The default puts the laws down the rows and the congested lines
across the columns, matching the other per-line grids.  `--transpose` swaps them -- lines down
the rows and laws across the columns -- which is easier to read when the line ids are long (as
they are for the Texas cases).

Line selection is the shared `select_lines`: the top lines by descending E[|mu_l|] under the
baseline (Bernoulli) law, no congestion filter, so a line sits in the same panel across figures.

Usage:
    uv run python -m src.figures.mu_distributions.line_exceed_error --case activsg2000_v12_lev3a3
    uv run python -m src.figures.mu_distributions.line_exceed_error --case activsg2000_v12_lev3a3 \
        --threshold 500 --budgets 640,2560
    uv run python -m src.figures.mu_distributions.line_exceed_error --case activsg2000_v12_lev3a3 \
        --transpose
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.sampling.problem import case_tag, create_problem
from src.sampling.distributions import DIST_NAMES, DIST_LABEL, case_law_names
from src.figures.style import DIST_COLOR
from src.figures.mu_distributions._line_stats import exceedance_prob, select_lines
from src.figures.mu_distributions._experiment_reads import direct_samples, direct_slice
from src.figures.mu_distributions._panel_layout import (FONT, LABEL_IN, LEGEND_IN, TITLE_IN,
                                           margin_grid, one_sigfig_yaxis, panel_title,
                                           square_grid_height)

EST_COLOR = "tab:blue"            # the direct estimate points and their bars
CAPSIZE = 2.0                     # error-bar cap width (points)
FLOOR_PAD = 0.08                  # y-range padding as a fraction of a line's own span
ROW_ASPECT = 1.0                  # transposed panels: width / height
LEFT_WIDE_IN = 0.66               # transposed left margin (in): room for the long "line" labels
TOP_WIDE_IN = 0.30               # transposed top margin (in): suptitle plus the column titles
BOTTOM_WIDE_IN = 0.28             # transposed bottom margin (in): budget x ticks, x label, key


def direct_exceed_frame(samples, mu, line_names, dists, budgets, threshold, rep=0):
    """Per-(dist, budget, line) estimate of P[|mu_l| > threshold] and its standard error.

    One replicate is one direct run of `budget` iid draws; direct weights are uniform, so the
    estimate is the plain share of the run's draws with |mu_l| > threshold and the standard error
    is that exceedance indicator's sample standard deviation over sqrt(budget) -- the Bernoulli
    standard error of the proportion.  A single run per budget keeps the error bar tied to that
    budget's sample size.  `mu` is the law-independent |mu_l| table; only the sampled ids change
    from law to law.
    """
    rows = []
    for dist in dists:
        sl = direct_slice(samples, dist, budgets)
        sl = sl[sl["rep"] == rep]
        for budget, g in sl.groupby("budget", sort=True):
            v = (mu[g["id"].to_numpy()] > threshold).astype(float)   # (n_draw, n_line)
            se = v.std(0, ddof=1) / np.sqrt(v.shape[0])
            rows.append(pd.DataFrame({"dist": dist, "budget": budget,
                                      "line": line_names, "est": v.mean(0), "se": se}))
    return pd.concat(rows, ignore_index=True)


def law_truths(c, dists, mu, threshold):
    """Per-line exact P[|mu_l| > threshold] for each law -- the dashed reference in each panel."""
    return {d: exceedance_prob(mu, create_problem(c, d).w, threshold) for d in dists}


def budget_ticklabels(budgets):
    """Labels for the budget x ticks: only the two endpoints, the rest blank (six points read
    without a legend)."""
    n = len(budgets)
    return [str(budgets[0])] + [""] * (n //2 - 1) + [str(budgets[n//2])] + [""] * (n //2 - 1)


def errorbar_panel(ax, sub, budgets, truth, ylim, color):
    """One (law, line) panel: per-budget estimate with a standard-error bar, plus exact truth.

    Budget is the x axis, one slot per budget from small to large; only the endpoint budgets are
    tick-labeled, so the estimate reads against the sample size directly instead of by color."""
    sigma = sub[sub["budget"] == 10240]["se"].iloc[0] * 10240**0.5
    for i, b in enumerate(budgets):
        p = sub[sub["budget"] == b]
        ax.errorbar(i, float(p["est"].iloc[0]), yerr=float(p["se"].iloc[0]), fmt="o",
                    ms=2, color=color, ecolor=color, elinewidth=1.0, capsize=CAPSIZE,
                    zorder=4)
        #ax.errorbar(i+0.3, float(p["est"].iloc[0]), yerr=sigma/b**0.5, fmt="", ecolor="magenta", elinewidth=0.5)
    ax.axhline(truth, color="k", ls="--", lw=0.6, zorder=5)
    ax.set_xlim(-0.5, len(budgets) - 0.5)
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels(budget_ticklabels(budgets))
    ax.set_yticks([0., .4, .8])
    ax.set_yticks([.2, .6], minor=True)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=FONT["tick"], pad=1.5)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", which="major", length=2.5)
    ax.tick_params(axis="y", which="minor", length=1.5)
    #one_sigfig_yaxis(ax)


def line_ylims(frame, truths, order_names, name_index, dists):
    """The full probability range [0, 1] for every line's panels."""
    return {name: (0.0, 1.0) for name in order_names}


def decorate_grid(fig, handles, threshold, supy_x, legend_title=None, legend_bottom=False):
    """Shared legend, suptitle, and shared y label for either grid orientation.

    `handles` are the figure legend entries; `legend_title` heads the legend; `legend_bottom`
    places the legend along the bottom edge rather than below the suptitle; `supy_x` is the
    figure-fraction x of the shared y label."""
    h = fig.get_figheight()
    loc, anchor = (("lower center", (0.5, LABEL_IN / h)) if legend_bottom
                   else ("upper center", (0.5, 1 - LEGEND_IN / h)))
    #fig.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, title=legend_title,
    #           ncol=len(handles), frameon=True, fontsize=FONT["legend"],
    #           title_fontsize=FONT["axlabel"], handletextpad=0.3, columnspacing=0.8,
    #           borderpad=0.3, handlelength=1.4)
    #fig.suptitle(title, y=1 - TITLE_IN / h, fontsize=FONT["suptitle"])
    fig.supylabel(rf"$\hat{{P}}[\,|\mu_l| > {threshold:g}\,]$", x=supy_x, fontsize=FONT["axlabel"])


def grid_legend():
    """The two legend entries shared by both orientations: the direct estimate and exact truth.

    Budget is read off the x axis now, so the legend only names the marker and the dashed line."""
    return [Line2D([0], [0], marker="o", ls="", ms=3, color=EST_COLOR, label="estimate"),
            Line2D([0], [0], color="k", ls="--", label="exact truth")]


def figure_lines_as_cols(frame, dorder, order_names, name_index, truths, budgets, threshold):
    """Grid with laws down the rows and congested lines across the columns (the default)."""
    fig, axs = margin_grid(len(dorder), len(order_names))
    ylims = line_ylims(frame, truths, order_names, name_index, dorder)
    for i, d in enumerate(dorder):
        sub_d = frame[frame["dist"] == d]
        for j, name in enumerate(order_names):
            ax = axs[i, j]
            errorbar_panel(ax, sub_d[sub_d["line"] == name], budgets,
                           float(truths[d][name_index[name]]), ylims[name], DIST_COLOR[d])
            if i == 0:
                panel_title(ax, f"{name}")
            if j == 0:
                ax.set_ylabel(DIST_LABEL[d], fontsize=FONT["panel_title"])
    fig.supxlabel("sampling budget", y=LABEL_IN / fig.get_figheight(), fontsize=FONT["axlabel"])
    decorate_grid(fig, grid_legend(), threshold, supy_x=LABEL_IN / fig.get_figwidth() - 0.02)
    return fig


def figure_lines_as_rows(frame, dorder, order_names, name_index, truths, budgets, threshold):
    """Transposed grid: congested lines down the rows, laws across the columns.

    Rows share one y scale so a line's tick labels are drawn once (sharey='row'); the law names
    title the columns from outside the box; budget is the x axis of every panel, endpoint-labeled,
    the same as the default grid."""
    fig, axs = margin_grid(len(order_names), len(dorder), sharey="row", aspect=ROW_ASPECT,
                           bottom=0.1, wspace=0.1, w=3.4)
    ylims = line_ylims(frame, truths, order_names, name_index, dorder)
    for i, name in enumerate(order_names):
        for j, d in enumerate(dorder):
            ax = axs[i, j]
            sub = frame[(frame["dist"] == d) & (frame["line"] == name)]
            errorbar_panel(ax, sub, budgets, float(truths[d][name_index[name]]), ylims[name], DIST_COLOR[d])
            if i == 0:
                ax.set_title(DIST_LABEL[d], fontsize=FONT["panel_title"], pad=3)
            if j == 0:
                ax.set_ylabel("ABC"[i], fontsize=FONT["panel_title"])
    h = square_grid_height(fig, len(order_names), len(dorder), TOP_WIDE_IN, BOTTOM_WIDE_IN,
                           LEFT_WIDE_IN)
    fig.set_figheight(h)
    fig.subplots_adjust(left=LEFT_WIDE_IN / fig.get_figwidth(),
                        top=1 - TOP_WIDE_IN / h, bottom=BOTTOM_WIDE_IN / h)
    fig.supxlabel("sampling budget", y=(LABEL_IN + 0.00) / h, fontsize=FONT["axlabel"])
    decorate_grid(fig, grid_legend(), threshold, supy_x=LABEL_IN / fig.get_figwidth(),
                  legend_bottom=True)
    return fig


def run(case, dists=DIST_NAMES, budgets=None, threshold=200.0, transpose=False):
    """One direct-only exceedance-probability figure with standard errors over the nominal laws.

    The |mu_l| table and the sample frame are law-independent and built once; each law's exact
    P[|mu_l| > c] is per law, and the lines follow the shared `select_lines` selection.
    `transpose` swaps the grid axes.
    """
    c, samples, meta = direct_samples(case)
    dists = case_law_names(c, dists)
    case, case_dir = c.name, c.results_dir
    budgets = budgets or meta["budgets"]
    baseline = create_problem(c, dists[0])
    mu, line_names = baseline.mu_table()
    truths = law_truths(c, dists, mu, threshold)
    idx, order_names = select_lines(mu, line_names, baseline.w, n=3)
    name_index = {name: i for name, i in zip(order_names, idx)}
    frame = direct_exceed_frame(samples, mu, line_names, dists, budgets, threshold)

    title = (f"{case.split('_')[0]}: direct  "
             rf"$P[\,|\mu_l| > {threshold:g}\,]$  |  budget = sample size")
    build = figure_lines_as_rows if transpose else figure_lines_as_cols
    fig = build(frame, dists, order_names, name_index, truths, budgets, threshold)
    suffix = "_linerows" if transpose else ""
    out = case_dir / f"line_exceed_error_thr{int(threshold)}{suffix}_{case_tag(case)}.pdf"
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
