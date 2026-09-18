"""Per-line inverse-CDF (exceedance) grid: exact truth plus a few direct replicate curves.

One panel per line (congested lines first).  Each panel shows the exact nominal-weighted
survivor curve P[|mu_l| > mu_hat] (black) and a handful of individual direct (nominal Monte
Carlo) replicate survivor curves, so per-line replicate-to-replicate variability -- and the
tail floor at 1/budget where direct sampling runs out of draws -- is visible against truth.
Direct draws are target-independent, so only the nominal law (dist) and budget select the
sample.

Usage:
    uv run python -m src.figures.tail.direct_inverse_cdf --case case14_gen_per_bus
    uv run python -m src.figures.tail.direct_inverse_cdf --case case14_gen_per_bus \
        --dists bernoulli,mrf_a --budget 640 --n-reps 8
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from src.sampling.problem import TargetProblem, CASES, case_tag
from src.sampling.distributions import case_dist
from src.figures.tail.per_line_survivor import weighted_survivor, decimate_monotone, line_order
from src.figures.style import COLORS
from src.figures.tail.direct_common import load_direct_context

FIG_W = 6.0                     # single figure width, fits the text block of a two-column page


def survivor_grid(order_names, draw_panel, legend_handles, title, xlabel, ylabel):
    """6-inch-wide shared-axis per-line survivor grid; returns (fig, axs).

    order_names     ordered line ids, one panel each (congested first)
    draw_panel      callback(ax, name) that draws one line's panel
    legend_handles  figure legend entries
    """
    ncol = min(5, len(order_names))
    nrow = min(4, int(np.ceil(len(order_names) / ncol)))
    head, foot, lft, rgt = 0.58, 0.46, 0.60, 0.08
    panel_w = (FIG_W - lft - rgt) / ncol
    figh = head + nrow * 0.95 * panel_w + foot
    fig, axs = plt.subplots(nrow, ncol, figsize=(FIG_W, figh), squeeze=False,
                            sharex=True, sharey=True)
    axs[0, 0].set_yscale("log")
    axs[0, 0].xaxis.set_major_locator(MaxNLocator(nbins=3, prune="upper"))
    for name, ax in zip(order_names, axs.ravel()):
        draw_panel(ax, name)
        ax.tick_params(labelsize=6)
    for ax in axs.ravel()[len(order_names):]:
        ax.set_visible(False)

    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1 - 0.26 / figh), ncol=len(legend_handles),
               frameon=True, fontsize=6, handlelength=1.6, borderpad=0.3)
    fig.suptitle(title, y=1 - 0.08 / figh, fontsize=8)
    fig.supxlabel(xlabel, y=0.07 / figh, fontsize=8)
    fig.supylabel(ylabel, x=0.06 / FIG_W, fontsize=8)
    fig.subplots_adjust(left=lft / FIG_W, right=1 - rgt / FIG_W, bottom=foot / figh,
                        top=1 - head / figh, wspace=0.22, hspace=0.20)
    return fig, axs


def replicate_indices(n_rep, n_show):
    """Evenly spaced replicate indices so the overlaid curves span the full set."""
    return np.unique(np.linspace(0, n_rep - 1, n_show).round().astype(int))


def direct_rep_samples(samples, dist, budget, target="max_mu"):
    """Direct method's per-replicate (ids, weights), keyed by rep index."""
    sl = samples[(samples.method == "direct") & (samples.dist == dist)
                 & (samples.budget == budget) & (samples.target == target)
                 & (samples.snapshot == "multi")]
    return {r: (g["id"].to_numpy(), g["weight"].to_numpy())
            for r, g in sl.groupby("rep", sort=True)}


def panel(ax, line_idx, line_name, mu, truth_w, reps_samples, rep_ids):
    """Truth survivor (black) plus each chosen direct replicate's survivor curve (red)."""
    for r in rep_ids:
        ids, w = reps_samples[r]
        v, s = decimate_monotone(*weighted_survivor(mu[ids, line_idx], w))
        ax.plot(v, s, color=COLORS["direct"], lw=0.8, alpha=0.5, rasterized=True)
    v, s = decimate_monotone(*weighted_survivor(mu[:, line_idx], truth_w))
    ax.plot(v, s, color="k", lw=1.5, rasterized=True)
    ax.set_title(f"line {line_name}", y=0.82, fontsize=6)


def plot_replicates(case, problem, mu, line_names, dist, budget, reps_samples, rep_ids,
                    case_dir):
    """One figure: the exact per-line survivor curve plus a few direct replicate curves."""
    idx, names = line_order(mu, line_names, problem.w)
    idx_by_name = dict(zip(names, idx))

    def draw(ax, name):
        panel(ax, idx_by_name[name], name, mu, problem.w, reps_samples, rep_ids)
        ax.minorticks_off()

    handles = [Line2D([0], [0], color="k", lw=1.5, label="exact (full enumeration)"),
               Line2D([0], [0], color=COLORS["direct"], lw=0.8,
                      label=f"direct replicates ({len(rep_ids)})")]
    title = (f"{case}: per-line direct vs exact  |  {dist}, "
             f"budget {budget}, {len(rep_ids)} reps")
    fig, axs = survivor_grid(names, draw, handles, title, r"congestion threshold $\hat{\mu}$",
                             r"$P[\,|\mu_l| > \hat{\mu}\,]$")
    axs[0, 0].set_ylim(1e-4, 1.5)
    axs[0, 0].set_xlim(-mu.max() * 0.02, mu.max() * 1.02)
    axs[0, 0].set_yticks([10.0 ** -n for n in [1, 2, 3, 4]])

    out = case_dir / f"direct_per_line_inverse_cdf_{dist}_b{budget}_{case_tag(case)}.pdf"
    fig.savefig(out)
    print(f"saved -> {out}")
    return fig


def run(case, dists, budget=640, n_reps=8):
    """One direct-vs-exact figure per nominal law; returns them in the given order."""
    case_dir, samples, meta, tc, ngen, p = load_direct_context(case)
    figs = []
    for dist in dists:
        problem = TargetProblem(tc, dist=case_dist(CASES[case], dist))
        mu, line_names = problem.mu_table()
        reps_samples = direct_rep_samples(samples, dist, budget)
        rep_ids = [r for r in replicate_indices(meta["n_rep"], n_reps) if r in reps_samples]
        figs.append(plot_replicates(case, problem, mu, line_names, dist, budget,
                                    reps_samples, rep_ids, case_dir))
    return figs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="case14_gen_per_bus")
    parser.add_argument("--dists", default="bernoulli")
    parser.add_argument("--budget", type=int, default=640)
    parser.add_argument("--n-reps", type=int, default=8)
    args = parser.parse_args()

    run(args.case, args.dists.split(","), args.budget, args.n_reps)
    plt.show()


if __name__ == "__main__":
    main()
