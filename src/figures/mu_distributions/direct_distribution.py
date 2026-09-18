"""Per-line |mu_l| distribution: the exact mass filled, each sampling budget an outline.

A companion to the direct box plots (direct_box.py) and a sibling of the enumeration figure
mu_distribution_difference.py, sharing its filled-baseline plus step-outline idiom.  The box
plots summarize a single tail statistic across replicates; this figure shows the whole shape
of the congestion dual |mu_l| and how well direct sampling recovers it at each budget.

One figure per nominal law, one panel per line (most-congested first).  Each panel overlays:

  filled      the exact nominal-weighted mass of |mu_l| in each bin (full enumeration)
  outlines    one step-outline histogram per budget, from that budget's first replicate

Both sides estimate the same P[|mu_l| in bin] -- the exact from the config weights, each
budget from its draw counts over the whole budget -- so a small budget shows as a ragged
outline that tightens onto the fill as the budget grows.  Lines that never congest above the
floor are dropped so no empty panels are drawn.

Usage:
    uv run python -m src.figures.mu_distributions.direct_distribution --case case14_gen_per_bus
    uv run python -m src.figures.mu_distributions.direct_distribution --case case14_gen_per_bus \
        --budgets 640,2560 --n-lines 8 --bins 40
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.sampling.problem import case_tag, create_problem
from src.sampling.distributions import DIST_LABEL, DIST_NAMES
from src.figures.style import budget_colors
from src.figures.mu_distributions._line_stats import select_lines
from src.figures.mu_distributions._panel_layout import panel_grid, one_sigfig_yaxis
from src.figures.mu_distributions._experiment_reads import direct_samples, direct_slice

EXACT_COLOR = "0.4"             # the filled exact-mass histogram, gray under the budget outlines


def bin_mass(v, w, edges):
    """Weighted probability mass of |mu_l| per bin (values below the first edge dropped)."""
    return np.histogram(v, bins=edges, weights=w)[0]


def budget_first_rep_ids(samples, dist, budgets):
    """Config ids drawn by the first replicate of each budget's direct sample."""
    sl = direct_slice(samples, dist, budgets)
    ids = {}
    for b in budgets:
        g = sl[sl["budget"] == b]
        first = sorted(g["rep"].unique())[0]
        ids[b] = g[g["rep"] == first]["id"].to_numpy()
    return ids


def dist_hist_legend(budgets, colors):
    """Legend handles: the filled exact mass, then one step outline per sampling budget."""
    handles = [Patch(facecolor=EXACT_COLOR, alpha=0.5, label="exact")]
    return handles + [Line2D([0], [0], color=colors[b], lw=1.4, label=str(b)) for b in budgets]


def dist_hist_panel(ax, mu, i, w, budget_ids, colors, floor, bins):
    """One line's panel: exact mass filled, one step outline per budget's direct draws.

    The exact nominal-weighted mass of |mu_l| per bin is drawn filled; each budget's first
    replicate is overlaid as a step outline of its empirical mass (draw count over the whole
    budget, so both estimate the same P[|mu_l| in bin]).  A small budget shows as a ragged
    outline that tightens onto the fill as the budget grows; |mu_l| at or below `floor` is
    uncongested and falls left of the first bin edge."""
    v = mu[:, i]
    edges = np.linspace(floor, float(v.max()), bins + 1)
    ax.stairs(bin_mass(v, w, edges), edges, fill=True, color=EXACT_COLOR, alpha=0.5, lw=0)
    for b, ids in budget_ids.items():
        ax.stairs(bin_mass(mu[ids, i], np.full(len(ids), 1.0 / len(ids)), edges), edges,
                  color=colors[b], lw=0.7)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_ylim(bottom=0, top=0.6)
    one_sigfig_yaxis(ax)


def dist_hist_draw(mu, name_index, w, budget_ids, colors, floor, bins):
    """Build the panel_grid callback that histograms one line's |mu_l| distribution."""
    def draw(ax, name):
        dist_hist_panel(ax, mu, name_index[name], w, budget_ids, colors, floor, bins)
    return draw


def plot_distribution(case, samples, meta, mu, line_names, w, dist, budgets, n_lines, case_dir,
                      floor=1e-2, bins=30):
    """One figure: per-line histogram of |mu_l| -- exact mass filled, each budget an outline.

    Each panel overlays the exact nominal-weighted mass of |mu_l| (filled gray) with one step
    outline per sampling budget, drawn from that budget's first replicate, so the outlines
    tighten onto the fill as the budget grows.  Lines are ranked by E[|mu_l|], most-congested
    first, via the shared `select_lines` selection (no congestion filter)."""
    idx, shown = select_lines(mu, line_names, w, n=n_lines)
    name_index = {name: i for i, name in zip(idx, shown)}
    budget_ids = budget_first_rep_ids(samples, dist, budgets)
    colors = budget_colors(budgets)
    title = (f"{case}: direct  "
             rf"$|\mu_l|$ distribution  |  {DIST_LABEL[dist]}, {meta['n_rep']} reps")
    draw = dist_hist_draw(mu, name_index, w, budget_ids, colors, floor, bins)
    fig, _ = panel_grid(shown, draw, dist_hist_legend(budgets, colors), title,
                        r"$|\mu_l|$", "probability mass", sharex=False, sharey=True,
                        lft=0.07, wspace=0.03)
    out = case_dir / f"direct_distribution_hist_{dist}_{case_tag(case)}.pdf"
    fig.savefig(out)
    print(f"saved -> {out}")
    return fig


def run(case, dists, budgets=None, n_lines=20, floor=1e-2, bins=30):
    """One |mu_l| distribution histogram per nominal law; returns the figures in order."""
    c, samples, meta = direct_samples(case)
    case, case_dir = c.name, c.results_dir
    budgets = budgets or meta["budgets"]

    figs = []
    for dist in dists:
        problem = create_problem(c, dist)
        mu, line_names = problem.mu_table()
        figs.append(plot_distribution(case, samples, meta, mu, line_names, problem.w, dist,
                                      budgets, n_lines, case_dir, floor, bins))
    return figs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="activsg2000_v12_lev3a3")
    parser.add_argument("--dists", default=",".join(DIST_NAMES))
    parser.add_argument("--budgets", default=None)
    parser.add_argument("--n-lines", type=int, default=10,
                        help="number of most-congested lines to show per figure")
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--floor", type=float, default=1e-2,
                        help="drop lines that never congest above this |mu_l|")
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else None
    run(args.case, args.dists.split(","), budgets, args.n_lines, args.floor, args.bins)
    plt.show()


if __name__ == "__main__":
    main()
