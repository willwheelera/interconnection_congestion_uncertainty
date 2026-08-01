"""Grid of method-overlay tail curves for one case: rows = distribution, cols = target.

Each subplot overlays every sampler (distinct color) and ascent move (line style) as a
median + 10-90 percentile band over the replicates, against the exact enumerated truth
(black), at a fixed budget.  One figure per budget.  This is the multi-panel companion to
plot_inverse_cdf.py (which draws a single slice), so the same (dist, target) cell reads
identically here and there.

Usage:
    uv run python -m figures.tail.target_grid --case case14_gen_per_bus
    uv run python -m figures.tail.target_grid --case case14_gen_per_bus \
        --targets max_mu,sum_mu --budgets 640,2560
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from sampling.problem import REPO_ROOT, load_experiment, case_tag
from figures.style import band
from figures.tail.inverse_cdf import COLORS, MOVE_STYLE, load_slice, series_label

DISTS = ["bernoulli", "mrf_a", "mrf_b", "mrf_c"]
DIST_LABEL = {"bernoulli": "Bernoulli", "mrf_a": "MRF-A", "mrf_b": "MRF-B", "mrf_c": "MRF-C"}


def panel(ax, case_dir, dist, target, budget):
    """One (dist, target) slice: truth plus every sampler as median + 10-90 band."""
    sel = dict(dist=dist, target=target, snapshot="multi", budget=budget)
    b, truth, series, _ = load_slice(case_dir, sel)
    ax.plot(b, truth, "k-", lw=2.2, label="exact (full enumeration)")
    for (method, move), est in series.items():
        band(ax, b, est, COLORS.get(method, "gray"), series_label(method, move))
        ax.lines[-1].set_linestyle(MOVE_STYLE[move])
    ax.set_yscale("log")
    pos = truth[truth > 0]
    ax.set_ylim(max(1e-7, pos.min() * 0.5) if pos.size else 1e-7, 1.5)


def figure_for_budget(case_dir, targets, budget, meta):
    """A rows=distribution x cols=target grid of method-overlay panels for one budget."""
    fig, axes = plt.subplots(len(DISTS), len(targets), squeeze=False,
                             figsize=(5.4 * len(targets), 3.4 * len(DISTS)))
    for i, dist in enumerate(DISTS):
        for j, target in enumerate(targets):
            ax = axes[i][j]
            panel(ax, case_dir, dist, target, budget)
            if i == 0:
                ax.set_title(f"target: {target}")
            if i == len(DISTS) - 1:
                ax.set_xlabel(r"threshold $b$ on target")
            if j == 0:
                ax.set_ylabel(f"{DIST_LABEL[dist]}\n" + r"$P[\,\mathrm{target}\geq b\,]$")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(f"{meta['case']}: all methods, budget {budget}, {meta['n_rep']} reps")
    fig.tight_layout(rect=(0, 0.09, 1, 0.97))
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="case14_gen_per_bus")
    parser.add_argument("--targets", default="max_mu,sum_mu")
    parser.add_argument("--budgets", default=None)
    args = parser.parse_args()

    case_dir = REPO_ROOT / "results" / args.case
    meta = load_experiment(case_dir, "curves")[1]
    targets = args.targets.split(",")
    budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else meta["budgets"]

    for budget in budgets:
        fig = figure_for_budget(case_dir, targets, budget, meta)
        out = case_dir / f"target_grid_b{budget}_{case_tag(args.case)}.pdf"
        fig.savefig(out)
        print(f"saved -> {out}")
    plt.show()


if __name__ == "__main__":
    main()
