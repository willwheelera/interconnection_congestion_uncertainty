"""Per-line histogram of the sampled |mu_l| price distribution for a sample-only queue.

The sampled counterpart of direct_distribution_difference.py.  That reads an enumerated case and
fills the exact nominal-weighted mass of |mu_l| per bin, then overlays one step outline per
sampling budget.  A sample_only case (ngen ~ 195) has no exact table -- its congestion-dual
distribution is only estimated, from the draws in results/<case>/mcmc/<law>/dual_samples.parquet.
So the full draw table (weighted uniformly, 1 / N) stands in for the exact mass and is filled
gray: the best estimate of the price distribution at this ngen.  Each budget b then overlays the
mass of an independent, non-overlapping block of b draws as a step outline, so the outlines
converge onto the fill as b grows.

One column per line (most-congested first), the shared `select_lines` selection (top lines by
descending E[|mu_l|]), so a line sits in the same column as in every other per-line figure.  Each
panel is titled with its fixed `line_label` (A-E for the paper contingencies, the contingency name
otherwise).  Binning from zero includes the uncongested draws (|mu_l| ~ 0), so every panel's mass
sums to one and the panels are comparable in area.

Usage:
    uv run python -m src.figures.mu_distributions.line_sample_distribution --case activsg2000_v12_levfull
    uv run python -m src.figures.mu_distributions.line_sample_distribution --case activsg2000_v12_levfull \
        --law bernoulli --n-lines 5 --budgets 1000,16000 --bins 30
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from src.cases import registry
from src.sampling.problem import case_tag
from src.figures.style import budget_colors
from src.figures.mu_distributions._line_stats import select_lines, line_label
from src.figures.mu_distributions._experiment_reads import read_dual_samples
from src.figures.mu_distributions._panel_layout import (FONT, LABEL_IN, margin_grid,
                                                        one_sigfig_yaxis, panel_title)

EST_COLOR = "0.4"               # the filled full-sample histogram, gray under the budget outlines
FIG_W_COL = 3.4                 # single-column figure width, inches


def bin_mass(v, w, edges):
    """Weighted probability mass of |mu_l| per bin (values below the first edge dropped)."""
    return np.histogram(v, bins=edges, weights=w)[0]


def default_budgets(nsamples, n=3):
    """A short geometric ladder of prefix sizes below the full draw count, for the outlines."""
    #return [int(b) for b in np.geomspace(nsamples / 256, nsamples / 4, n)]
    return 10 * (1 << 4*np.arange(n))


def grid_legend(budgets, colors):
    """Legend handles: the filled full-sample mass, then one step outline per budget prefix."""
    handles = [Patch(facecolor=EST_COLOR, alpha=0.6, label="256k")]
    return handles + [Line2D([0], [0], color=colors[b], lw=1.4, label=str(b)) for b in budgets]


def line_column(ax, v, w, budgets, colors, label, floor, bins):
    """Fill one line's column: the full-sample mass filled, plus one step outline per budget.

    The full draw table's empirical mass of |mu_l| per bin is filled gray; each budget b overlays
    the mass of an independent, non-overlapping block of b draws (uniform weight 1 / b) as a step
    outline, so each block is its own sample and the outlines tighten onto the fill as b grows.
    Binning from floor = 0 puts the uncongested draws in the first bar, so every mass sums to one.
    An empty `label` leaves the panel untitled (the combined figure labels columns on another row)."""
    edges = np.linspace(floor, float(v.max()), bins + 1)
    ax.stairs(bin_mass(v, w, edges), edges, fill=True, color=EST_COLOR, alpha=0.6, lw=0)
    start = 0
    for b in budgets:
        block = v[start:start + b]
        start += b                                  # independent, non-overlapping blocks
        ax.stairs(bin_mass(block, np.full(b, 1.0 / b), edges), edges, color=colors[b], lw=0.7)
    if label:
        panel_title(ax, label)
    ax.tick_params(labelsize=5.5, length=2)
    ax.set_xlim(0.0, 1.02 * float(v.max()))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))


def draw_dist_row(axes, mu, idx, names, w, budgets, colors, floor, bins, labels=True):
    """Draw one full row of per-line |mu_l| distribution panels onto the given axes.

    Shared by the standalone figure and the combined line_sample_error figure (where the row is
    added as a third row), so the fill-plus-outline column, the x range, and the shared mass y
    axis are defined once.  `labels` titles each panel with its `line_label`; pass False when the
    columns are already labeled on another row.  `axes`, `idx`, `names` are aligned per column."""
    for ax, i, name in zip(axes, idx, names):
        line_column(ax, mu[:, i], w, budgets, colors, line_label(name) if labels else "",
                    floor, bins)
    axes[0].set_ylabel("prob. mass", fontsize=FONT["axlabel"])
    axes[0].set_ylim(0, 0.6)
    one_sigfig_yaxis(axes[0])


def plot_distribution(case, mu, line_names, w, budgets, n_lines, case_dir, law, floor=0.0, bins=30):
    """One figure: per-line full-sample histogram of |mu_l| with budget-prefix outlines.

    Lines are ranked by E[|mu_l|], most-congested first, via the shared `select_lines` selection
    (no congestion filter); each panel is titled with its fixed `line_label`.  All panels share one
    y axis (mass) so areas are comparable, and each column keeps its own |mu_l| range.  Binning
    from `floor` = 0 includes the uncongested draws, so every panel's mass sums to one."""
    idx, names = select_lines(mu, line_names, w, n=n_lines)
    colors = budget_colors(budgets)
    c = 3
    r = int(np.ceil(len(idx) / c))
    fig, axs = margin_grid(r, c, w=FIG_W_COL, sharex=False, sharey=True, wspace=0.15, left=.12)
    axs = axs.ravel()

    draw_dist_row(axs, mu, idx, names, w, budgets, colors, floor, bins)
    for ax in axs[len(idx):]:                         # hide any trailing blank panels
        ax.set_visible(False)
    h = fig.get_figheight()
    fig.legend(handles=grid_legend(budgets, colors), loc="center left",
               bbox_to_anchor=(0.50, .47), ncol=0*len(budgets) + 1, title=None,
               title_fontsize=FONT["legend"], frameon=True, fontsize=FONT["legend"],
               handlelength=1.2, borderpad=0.3, columnspacing=1.0)
    fig.supxlabel(r"Congestion price $|\mu_l|$ (\$/MWh)", y=LABEL_IN / h - 0.02, fontsize=FONT["axlabel"])
    out = case_dir / "mcmc" / law / f"line_sample_distribution_{case_tag(case)}_{law}.pdf"
    fig.savefig(out)
    print(f"saved -> {out}")
    return fig


def run(case, law="bernoulli", budgets=None, n_lines=5, floor=0.0, bins=30):
    """The sampled |mu_l| price-distribution histogram for one case/law; returns the figure."""
    c = registry.get(case)
    mu, line_names = read_dual_samples(c, law)
    w = np.full(len(mu), 1.0 / len(mu))
    budgets = budgets or default_budgets(len(mu))
    return plot_distribution(c.name, mu, line_names, w, budgets, n_lines, c.results_dir, law,
                             floor, bins)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="activsg2000_v12_levfull")
    parser.add_argument("--law", default="bernoulli")
    parser.add_argument("--n-lines", type=int, default=3,
                        help="number of most-congested lines to show")
    parser.add_argument("--budgets", default=None)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--floor", type=float, default=0.0,
                        help="lower bin edge; 0 includes the uncongested draws, a positive value drops them")
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else None
    run(args.case, args.law, budgets, args.n_lines, args.floor, args.bins)
    plt.show()


if __name__ == "__main__":
    main()
