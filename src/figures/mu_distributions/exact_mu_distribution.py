"""Exact per-line distribution of the congestion dual |mu_l| under the nominal law.

Same per-line panel grid as direct_box.py (most-congested lines first, one figure per
nominal law), but each panel shows the full enumerated distribution of that line's |mu|
instead of a sampling estimate:

  bars    weighted probability mass per |mu| bin over the congested part |mu| > floor
  stem    the point mass at zero, P[|mu_l| <= floor] (the line is uncongested)
  line    the mean of |mu_l|, with the mean +- 1 sd interval as a shaded band

Mean and standard deviation are taken over the whole distribution, point mass included,
so a line that is uncongested 99% of the time has a small mean and a large sd.  The zero
mass is drawn as its own stem rather than as the first histogram bar so a point mass and a
binned mass are never read off the same mark.  Everything here is exact: the case is fully
enumerated, so every config's nominal weight is known.

Usage:
    uv run python -m src.figures.mu_distributions.exact_mu_distribution --case case30_gen_per_bus
    uv run python -m src.figures.mu_distributions.exact_mu_distribution --case case30_gen_per_bus \
        --dists bernoulli --n-lines 10 --bins 60
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from src.cases.registry import get_case
from src.sampling.problem import case_tag, create_problem
from src.sampling.distributions import DIST_LABEL
from src.figures.style import COLORS
from src.figures.mu_distributions._line_stats import select_lines
from src.figures.mu_distributions._panel_layout import panel_grid

ZERO_COLOR = "0.45"        # gray, so the point mass never reads as a histogram bar


def moments(v, w):
    """Weighted mean and standard deviation of one line's |mu| (zeros included)."""
    mean = float(w @ v)
    return mean, float(np.sqrt(w @ (v - mean) ** 2))


def congested_histogram(v, w, floor, bins):
    """Bin edges and weighted probability mass per bin over the congested part |mu| > floor."""
    keep = v > floor
    edges = np.linspace(floor, float(v.max()), bins + 1)
    mass, _ = np.histogram(v[keep], bins=edges, weights=w[keep])
    return edges, mass


def dist_panel(ax, v, w, floor, bins, ymin):
    """One line's panel: binned congested mass, the zero point mass, mean and +-1 sd."""
    edges, mass = congested_histogram(v, w, floor, bins)
    mean, sd = moments(v, w)
    ax.axvspan(max(mean - sd, 0.0), mean + sd, color="k", alpha=0.12, lw=0, zorder=0)
    ax.stairs(mass, edges, fill=True, color=COLORS["direct"], alpha=0.55, lw=0, zorder=2)
    p0 = float(w[v <= floor].sum())
    ax.vlines(0.0, ymin, p0, color=ZERO_COLOR, lw=2.0, zorder=3)
    ax.plot([0.0], [p0], marker="o", ms=3, color=ZERO_COLOR, zorder=3)
    ax.axvline(mean, color="k", lw=1.0, zorder=4)

    ax.set_xlim(-0.05 * edges[-1], 1.05 * edges[-1])
    ax.text(0.96, 0.80, f"mean {mean:.3g}\nsd {sd:.3g}\nP[0] {p0:.3g}",   # below the shared title
            transform=ax.transAxes, ha="right", va="top", fontsize=5, zorder=5,
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.0))


def dist_legend():
    """Legend handles: binned mass, the zero point mass, and the mean +- 1 sd marks."""
    return [Patch(facecolor=COLORS["direct"], alpha=0.55, label="mass per bin"),
            Line2D([0], [0], color=ZERO_COLOR, lw=2.0, marker="o", ms=3,
                   label=r"point mass at $|\mu_l| = 0$"),
            Line2D([0], [0], color="k", lw=1.0, label="mean"),
            Patch(facecolor="k", alpha=0.12, label=r"mean $\pm$ 1 sd")]


def plot_distribution(case, problem, mu, line_names, dist, floor, bins, n_lines, ymin, case_dir):
    """One figure: the exact |mu_l| distribution, one panel per shown line."""
    idx, names = select_lines(mu, line_names, problem.w, n=n_lines)
    shown = list(zip(idx, names))
    print(f"{dist}: showing {len(shown)} lines")
    column = {n: i for i, n in shown}

    def draw(ax, name):
        dist_panel(ax, mu[:, column[name]], problem.w, floor, bins, ymin)

    title = (f"{case}: exact distribution of the congestion dual  |  "
             f"{DIST_LABEL[dist]}, full enumeration")
    fig, axs = panel_grid([n for _, n in shown], draw, dist_legend(), title,
                          r"$|\mu_l|$", "probability", log_y=False, sharex=False, sharey=True,
                          label=lambda n: f"line {n}")
    axs[0, 0].set_ylim(ymin, 1.05)
    for ax in axs.ravel()[:len(shown)]:               # labeled 0/0.5/1, minor mark each tenth
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    out = case_dir / f"exact_mu_distribution_{dist}_{case_tag(case)}.pdf"
    fig.savefig(out)
    print(f"saved -> {out}")
    return fig


def run(case, dists, bins=20, n_lines=5, ymin=1e-7, mu_floor=1e-2):
    """One exact-distribution figure per nominal law; returns them in the given order."""
    c = get_case(case)
    case = c.name
    figs = []
    for dist in dists:
        problem = create_problem(c, dist)
        mu, line_names = problem.mu_table()
        figs.append(plot_distribution(case, problem, mu, line_names, dist, mu_floor,
                                      bins, n_lines, ymin, c.results_dir))
    return figs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="activsg2000_v12_lev3a3")
    parser.add_argument("--dists", default="bernoulli,mrf_cobuild,mrf_substitute")
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--n-lines", type=int, default=15)
    parser.add_argument("--ymin", type=float, default=1e-7)
    parser.add_argument("--mu-floor", type=float, default=1e-2,
                        help="treat |mu| below this as uncongested (near-binding dust)")
    args = parser.parse_args()

    run(args.case, args.dists.split(","), args.bins, args.n_lines, args.ymin, args.mu_floor)
    plt.show()


if __name__ == "__main__":
    dists = "bernoulli,mrf_cobuild,mrf_substitute"
    for i in range(4):
        case = f"activsg2000_v12_lev{i}a3"
        run(case, ["bernoulli","mrf_b"])#dists.split(","))
    plt.show()
