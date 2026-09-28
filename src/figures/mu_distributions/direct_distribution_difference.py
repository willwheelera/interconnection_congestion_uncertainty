"""Per-line |mu_l| distribution with the direct-sampling error broken out below.

A two-row companion to direct_distribution.py, sharing its filled-exact-mass plus
budget-outline histogram but splitting the sampling error onto its own row, the way
mu_distribution_difference.py separates the coupling effect from the shared shape.  Here the
overlaid lines are sampling budgets, not different nominal laws.

  top row     the exact nominal-weighted mass of |mu_l| filled, plus one step outline per
              sampling budget over the same bins (that budget's first replicate)
  bottom row  each budget as a colored step outline of mass_budget(bin) - mass_exact(bin),
              on a zero-centered axis -- the sampling error, magnified onto its own scale

Both the fill and each outline estimate the same P[|mu_l| in bin] -- the fill from the config
weights, each budget from its draw counts over the whole budget -- so the difference is the
error a budget makes recovering the exact mass, and shrinks toward zero as the budget grows.
Each row shares one y axis across lines (sharey by row) and columns share an x axis, so a
small error on a busy line is not flattened by a per-panel rescale.

One figure per nominal law, one column per line (most-congested first).  Lines that never
congest above the floor are dropped so no empty columns are drawn.

Usage:
    uv run python -m src.figures.mu_distributions.direct_distribution_difference --case case14_gen_per_bus
    uv run python -m src.figures.mu_distributions.direct_distribution_difference --case case14_gen_per_bus \
        --budgets 640,2560 --n-lines 5 --bins 40
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from src.sampling.problem import case_tag, create_problem
from src.sampling.distributions import DIST_LABEL, DIST_NAMES
from src.figures.mu_distributions._line_stats import select_lines
from src.figures.mu_distributions._panel_layout import FIG_W, FONT, fraction_grid, one_sigfig_yaxis, panel_title
from src.figures.mu_distributions._experiment_reads import direct_samples, direct_slice

EXACT_COLOR = "0.4"             # the filled exact-mass histogram, gray under the budget outlines


def bin_mass(v, w, edges):
    """Weighted probability mass of |mu_l| per bin (values below the first edge dropped)."""
    return np.histogram(v, bins=edges, weights=w)[0]


def budget_colors(budgets):
    """One color per budget along a light-to-dark ramp, kept off the gray exact fill."""
    shades = colormaps["plasma"](np.linspace(0.1, 0.65, len(budgets)))
    return {b: shades[k] for k, b in enumerate(budgets)}


def budget_first_rep_ids(samples, dist, budgets):
    """Config ids drawn by the first replicate of each budget's direct sample."""
    sl = direct_slice(samples, dist, budgets)
    ids = {}
    for b in budgets:
        g = sl[sl["budget"] == b]
        first = sorted(g["rep"].unique())[0]
        ids[b] = g[g["rep"] == first]["id"].to_numpy()
    return ids


def grid_legend(budgets, colors):
    """Legend handles: the filled exact mass, then one step outline per sampling budget."""
    handles = [Patch(facecolor=EXACT_COLOR, alpha=0.5, label="exact")]
    return handles + [Line2D([0], [0], color=colors[b], lw=1.4, label=str(b)) for b in budgets]


def line_column(top, bot, mu, i, w, budget_ids, colors, name, floor, bins):
    """Fill one line's column: exact mass plus budget outlines on top, budget error below.

    The exact nominal-weighted mass of |mu_l| per bin is filled gray on top; each budget's
    first replicate is overlaid as a step outline of its empirical mass (draw count over the
    whole budget) on top, and as a step outline of that mass minus the exact mass on the
    zero-centered bottom axis.  Both estimate the same P[|mu_l| in bin], so the bottom curve
    is the budget's recovery error and tightens onto zero as the budget grows."""
    v = mu[:, i]
    edges = np.linspace(floor, float(v.max()), bins + 1)
    base = bin_mass(v, w, edges)
    top.stairs(base, edges, fill=True, color=EXACT_COLOR, alpha=0.5, lw=0)
    for b, ids in budget_ids.items():
        m = bin_mass(mu[ids, i], np.full(len(ids), 1.0 / len(ids)), edges)
        top.stairs(m, edges, color=colors[b], lw=0.7)
        #bot.stairs(m - base, edges, color=colors[b], lw=1.0)
    #bot.axhline(0.0, color="k", lw=0.5, zorder=1)
    panel_title(top, f"{name}")
    for ax in (top, ):#bot):
        ax.tick_params(labelsize=5.5, length=2)


def plot_distribution(case, samples, meta, mu, line_names, w, dist, budgets, n_lines, case_dir,
                      floor=1e-2, bins=30):
    """One two-row figure: exact-plus-budget histogram on top, budget error below.

    Lines are ranked by E[|mu_l|], most-congested first; lines that never congest above
    `floor` are dropped so no empty columns are drawn.  The top row shares its y axis (mass)
    across lines and the bottom row shares its own (error), symmetric about zero, so the same
    error reads at the same height on every line."""
    idx, names = select_lines(mu, line_names, w, n=n_lines)
    shown = list(zip(idx, names))
    budget_ids = budget_first_rep_ids(samples, dist, budgets)
    colors = budget_colors(budgets)
    fig, axs, figh = fraction_grid(1, len(shown), head=0.13, foot=0.2)

    for col, (i, name) in enumerate(shown):
        line_column(axs[0, col], axs[0, col], mu, i, w, budget_ids, colors, name, floor, bins)

    for j, (i, _) in enumerate(shown):
        axs[0, j].set_xlim(0., 1.02 * float(mu[:, i].max()))
        axs[0, j].xaxis.set_major_locator(MaxNLocator(nbins=4))
    axs[0, 0].set_ylabel("prob. mass", fontsize=FONT["axlabel"])
    #axs[1, 0].set_ylabel(r"$\Delta$ prob. mass", fontsize=FONT["axlabel"])

    axs[0, 0].set_ylim(0, 0.6)
    bound = 0.05 #max(abs(v) for v in axs[1, 0].get_ylim()) * 1.5
    #axs[1, 0].set_ylim(-bound, bound)
    one_sigfig_yaxis(axs[0, 0])
    #one_sigfig_yaxis(axs[1, 0])

    fig.legend(handles=grid_legend(budgets, colors), loc="center left",
               bbox_to_anchor=(1 - 0.8 / FIG_W, 0.5), ncol=1, title="budget",
               title_fontsize=FONT["legend"], frameon=True, fontsize=FONT["legend"],
               handlelength=1.6, borderpad=0.3)
    #fig.suptitle(f"{case}: direct  " rf"$|\mu_l|$ error  |  {DIST_LABEL[dist]}, {meta['n_rep']} reps",
    #             y=1 - 0.045 / figh, fontsize=FONT["suptitle"])
    fig.supxlabel(r"Congestion price $|\mu_l|$", y=0.0 / figh, fontsize=FONT["axlabel"])
    out = case_dir / f"direct_distribution_difference_{dist}_{case_tag(case)}.pdf"
    fig.savefig(out)
    print(f"saved -> {out}")
    return fig


def run(case, dists, budgets=None, n_lines=5, floor=1e-2, bins=30):
    """One |mu_l| distribution-plus-error figure per nominal law; returns the figures in order."""
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
    parser.add_argument("--n-lines", type=int, default=5,
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
