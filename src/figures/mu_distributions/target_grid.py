"""Grid of method-overlay tail curves for one case: rows = distribution, cols = target.

Each subplot overlays the selected samplers (distinct color) and ascent moves (line style)
as a median + 10-90 percentile band over the replicates, against the exact enumerated truth
(black), at a fixed budget.  One figure per budget.  This is the multi-panel companion to
inverse_cdf.py (which draws a single slice), so the same (dist, target) cell reads
identically here and there -- including the sampler selection, `direct` and ascent-IS at
beta 0.2 by default (see figures.style).

Usage:
    uv run python -m src.figures.mu_distributions.target_grid --case case14_gen_per_bus
    uv run python -m src.figures.mu_distributions.target_grid --case case14_gen_per_bus \
        --targets max_mu,sum_mu --budgets 640,2560
    uv run python -m src.figures.mu_distributions.target_grid --case activsg2000_original_v12 \
        --methods direct,ascent_is
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from src.cases.registry import get_case
from src.sampling.problem import load_experiment, case_tag
from src.sampling.distributions import DIST_LABEL
from src.figures.style import (MOVE_STYLE, band, method_color, series_label, series_tag,
                           selection_label, add_method_args, method_selection)
from src.figures.mu_distributions._experiment_reads import load_slice, select_series

DISTS = ["bernoulli", "mrf_a", "mrf_cobuild", "mrf_substitute"]


def panel(ax, case_dir, dist, target, budget, methods):
    """One (dist, target) slice: truth plus each selected sampler as median + 10-90 band."""
    sel = dict(dist=dist, target=target, snapshot="multi", budget=budget)
    b, truth, series, _ = load_slice(case_dir, sel)
    ax.plot(b, truth, "k-", lw=2.2, label="exact (full enumeration)")
    for (method, move), est in select_series(series, methods).items():
        band(ax, b, est, method_color(method), series_label(method, move))
        ax.lines[-1].set_linestyle(MOVE_STYLE[move])
    ax.set_yscale("log")
    pos = truth[truth > 0]
    ax.set_ylim(max(1e-7, pos.min() * 0.5) if pos.size else 1e-7, 1.5)


def figure_for_budget(case_dir, targets, budget, meta, methods):
    """A rows=distribution x cols=target grid of method-overlay panels for one budget."""
    fig, axes = plt.subplots(len(DISTS), len(targets), squeeze=False,
                             figsize=(5.4 * len(targets), 3.4 * len(DISTS)))
    for i, dist in enumerate(DISTS):
        for j, target in enumerate(targets):
            ax = axes[i][j]
            panel(ax, case_dir, dist, target, budget, methods)
            if i == 0:
                ax.set_title(f"target: {target}")
            if i == len(DISTS) - 1:
                ax.set_xlabel(r"threshold $b$ on target")
            if j == 0:
                ax.set_ylabel(f"{DIST_LABEL[dist]}\n" + r"$P[\,\mathrm{target}\geq b\,]$")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(f"{meta['case']}: {selection_label(methods)}, budget {budget}, "
                 f"{meta['n_rep']} reps")
    fig.tight_layout(rect=(0, 0.09, 1, 0.97))
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="case14_gen_per_bus")
    parser.add_argument("--targets", default="max_mu,sum_mu")
    parser.add_argument("--budgets", default=None)
    add_method_args(parser)
    args = parser.parse_args()
    methods = method_selection(args)

    c = get_case(args.case)
    case_dir = c.results_dir
    meta = load_experiment(case_dir, "curves")[1]
    targets = args.targets.split(",")
    budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else meta["budgets"]

    for budget in budgets:
        fig = figure_for_budget(case_dir, targets, budget, meta, methods)
        out = (case_dir / f"target_grid_b{budget}{series_tag(methods)}_"
                          f"{case_tag(c.name)}.pdf")
        fig.savefig(out)
        print(f"saved -> {out}")
    plt.show()


if __name__ == "__main__":
    main()
