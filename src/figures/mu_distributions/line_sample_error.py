"""Sampled per-line convergence of E[|mu_l|] and P[|mu_l| > c] against the sample budget.

The sampled counterpart of line_mean_error.py and line_exceed_error.py.  Those read an
enumerated case: a 2^ngen dual_cache, exact nominal weights, and an experiment_samples sweep,
with the exact nominal-weighted truth drawn as a dashed line.  A sample_only case (ngen ~ 195)
has none of those -- its congestion-dual distribution is only *estimated*, from the draws in
results/<case>/mcmc/<law>/dual_samples.parquet.  So this reads that wide draw table directly,
weights the draws uniformly (1 / N), and plots the estimate from the first N draws against N.
There is no exact truth at this ngen, so there is no reference line; the largest-budget point
is the full-sample estimate the smaller budgets converge toward.

One single-column (3.4 in) figure: sampled E[|mu_l|] on top (with the single most-probable
config overlaid) and sampled P[|mu_l| > c] below, both against the budget x axis.  Per-line
statistics and the line selection are the shared ones (`select_lines`, the top lines by
descending E[|mu_l|]), so a line sits in the same column as in every other per-line figure.

Usage:
    uv run python -m src.figures.mu_distributions.line_sample_error --case activsg2000_v12_levfull
    uv run python -m src.figures.mu_distributions.line_sample_error --case activsg2000_v12_levfull \
        --law bernoulli --threshold 500 --budgets 2000,8000,32000
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.cases import registry
from src.sampling.problem import case_tag
from src.figures.mu_distributions._line_stats import select_lines, line_label
from src.figures.mu_distributions._experiment_reads import read_dual_samples, read_modal_point
from src.figures.mu_distributions._panel_layout import (FONT, LABEL_IN, MARGIN_IN, margin_grid,
                                                        one_sigfig_yaxis, panel_title)

EST_COLOR = "tab:blue"            # the estimate points and their standard-error bars
MODAL_COLOR = "tab:red"           # the modal-config |mu| overlay line
CAPSIZE = 2.0                     # error-bar cap width (points)
FIG_W_COL = 3.4                   # single-column figure width, inches
LEFT_FRAC = 0.15                  # left band (fraction): room for the y-axis labels at this width


def default_budgets(nsamples, n=6):
    """A geometric ladder of n subsample sizes from a small start up to the full count."""
    #lo = max(32, nsamples // (1 << 2*(n - 1)))
    lo = 10
    #n = int(np.log2(nsamples)) // 2 # increase by factors of 4
    return lo << np.arange(n) * 2
    return np.geomspace(lo, nsamples, n).astype(int)


def estimate_frame(mu, line_names, budgets, reduce):
    """Per-(budget, line) estimate of a reduced draw value and its standard error.

    Uniform weights, so the estimate is the plain mean of the first N draws' reduced value and
    the standard error is their sample standard deviation over sqrt(N).  Nested prefixes take
    the first N rows of the draw table (iid draws, so a prefix is itself a valid sample).
    """
    rows = []
    start = 0
    for n in budgets:
        v = reduce(mu[start:start+n])
        start += n # take independent/nonoverlapping subsets
        se = v.std(0, ddof=1) / np.sqrt(n)
        rows.append(pd.DataFrame({"budget": n, "line": line_names, "est": v.mean(0), "se": se}))
    return pd.concat(rows, ignore_index=True)


def budget_ticklabels(budgets):
    """Only the endpoint budgets are tick-labeled; the rest blank (the x axis reads itself)."""
    n = len(budgets)
    #return [str(budgets[0])] + [""] * (n //2 - 1) + [str(budgets[n//2])] + [""] * (n //2 - 1)
    labels = [""] * n
    labels[0] = str(budgets[0])
    labels[n//2] = str(budgets[n//2])
    return labels


def errorbar_panel(ax, sub, budgets, ylim):
    """One line's panel: per-budget estimate with a standard-error bar, no reference line."""
    for i, b in enumerate(budgets):
        p = sub[sub["budget"] == b]
        ax.errorbar(i, float(p["est"].iloc[0]), yerr=float(p["se"].iloc[0]), fmt="o",
                    ms=2, color=EST_COLOR, ecolor=EST_COLOR, elinewidth=1.0, capsize=CAPSIZE,
                    zorder=4)
    ax.set_xlim(-1.5, len(budgets) - 0.5)
    #ax.set_xticks(range(len(budgets)))
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ymax = ax.get_ylim()[1]
        ax.set_ylim(bottom=0., top = ymax * 1.2)
    ax.tick_params(labelsize=FONT["tick"], pad=1.5)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", which="major", length=2.5)


def convergence_legend(ax, modal):
    """Panel-inset key: the sample estimate and, when present, the modal-config overlay."""
    handles = [Line2D([0], [0], marker="o", ls="", ms=3, color=EST_COLOR, label="MC samples")]
    if modal:
        handles.append(Line2D([0], [0], marker="o", ls="", ms=3, color=MODAL_COLOR,
                              label="modal config"))
    ax.legend(handles=handles, loc="lower center", frameon=False, fontsize=FONT["tick"],
              handlelength=0.8, handletextpad=0.3, borderpad=0.1, labelspacing=0.3)


def combined_figure(mean_frame, exceed_frame, order_names, budgets, threshold, modal=None):
    """Per-line convergence panels at single-column width, one column per line.

    Rows: sampled E[|mu_l|] on top, sampled P[|mu_l| > c] below, sharing the budget x axis
    (labeled once, below the exceedance row).  Each mean panel keeps its own y scale; the
    exceedance row is fixed to [0, 1].  Panels are titled with the shared `line_label`, so
    columns match the other per-line figures."""
    fig, axs = margin_grid(2, len(order_names), w=FIG_W_COL, sharex=False, left=LEFT_FRAC,
                           hspace=0.16, wspace=.30)
    for j, name in enumerate(order_names):
        top, mid = axs[0, j], axs[1, j]
        if modal and name in modal:
            top.scatter(-1, modal[name], color=MODAL_COLOR, zorder=5, s=4)
        errorbar_panel(top, mean_frame[mean_frame["line"] == name], budgets, None)
        one_sigfig_yaxis(top, steps=(1, 2, 5, 10), sci_above=1000)
        errorbar_panel(mid, exceed_frame[exceed_frame["line"] == name], budgets, (0.0, 1.0))
        panel_title(top, line_label(name))
    axs[0, 0].set_ylabel(r"$\hat{E}[\,|\mu_l|\,]$", fontsize=FONT["axlabel"])
    axs[1, 0].set_ylabel(rf"$\hat{{P}}[\,|\mu_l| > {threshold:g}\,]$", fontsize=FONT["axlabel"])

    for ax in axs[0]:
        ax.tick_params(labelbottom=False)             # the budget axis is labeled once, below
    for ax in axs[1]:                                  # modal dot sits left of the budget ticks at -1
        ax.set_xticks(range(-1, len(budgets)))
        ax.set_xticklabels([""] + budget_ticklabels(budgets))
    for ax in axs[1, 1:]:                              # exceedance is [0, 1] everywhere; label once
        ax.tick_params(labelleft=False)

    convergence_legend(axs[0, 2], modal)
    top_band = 0.04                                   # the figure legend moved into panel (0, 2)
    h = fig.get_figheight() - (MARGIN_IN["top"] - top_band)
    fig.set_figheight(h)
    fig.subplots_adjust(top=1 - top_band / h, bottom=MARGIN_IN["bottom"] / h)
    fig.supxlabel(r"Sampling budget $(1\!0\times 4^n)$", y=LABEL_IN / h, fontsize=FONT["axlabel"])
    return fig


def run(case, law="bernoulli", budgets=None, threshold=200.0, n_lines=3):
    """The combined sampled mean-and-exceedance convergence figure for one case/law; returns it.

    The two convergence rows read the same uniform-weight draw table and the same `select_lines`
    lines; only the reduced statistic (|mu_l| vs. the exceedance indicator) and the y axis differ.
    """
    c = registry.get(case)
    mu, line_names = read_dual_samples(c, law)
    w = np.full(len(mu), 1.0 / len(mu))
    _, order_names = select_lines(mu, line_names, w, n=n_lines)
    budgets = budgets or default_budgets(len(mu))
    print(budgets)
    modal = read_modal_point(c, law)                     # the single most-probable config, if solved

    mean_frame = estimate_frame(mu, line_names, budgets, lambda m: m)
    exceed_frame = estimate_frame(mu, line_names, budgets,
                                  lambda m: (m > threshold).astype(float))
    fig = combined_figure(mean_frame, exceed_frame, order_names, budgets, threshold, modal=modal)
    out = c.results_dir / "mcmc" / law / f"line_sample_{case_tag(c.name)}_{law}.pdf"
    fig.savefig(out)
    print(f"saved -> {out}")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="activsg2000_v12_levfull")
    parser.add_argument("--law", default="bernoulli")
    parser.add_argument("--budgets", default=None)
    parser.add_argument("--threshold", type=float, default=100.0)
    parser.add_argument("--lines", type=int, default=3)
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else None
    run(args.case, args.law, budgets, args.threshold, args.lines)
    plt.show()


if __name__ == "__main__":
    main()
