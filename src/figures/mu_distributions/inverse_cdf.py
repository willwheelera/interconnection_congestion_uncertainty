"""Inverse-CDF (threshold-probability) plot for one slice of the experiment matrix.

Reads a case's experiment_curves.parquet (run_experiments.py) and, for one fixed
(dist, target, snapshot, budget) slice, overlays the selected samplers' estimated
exceedance curve P[target >= b] -- median with a 10-90 band over replicates -- against the
exact enumerated truth (black).  Ascent move strategies share the sampler's color and
differ by line style, so all/best/top5 read as one family; the ascent-IS defensive weights
are separate series in one green ramp.

By default this draws `direct` and ascent-IS at beta 0.2; every other sampler is in the
frame and comes back with --methods (see figures.style).

Usage:
    uv run python -m src.figures.mu_distributions.inverse_cdf --case case14_gen_per_bus
    uv run python -m src.figures.mu_distributions.inverse_cdf --case case14_gen_per_bus \
        --dist mrf_a --target sum_mu --budget 640
    uv run python -m src.figures.mu_distributions.inverse_cdf --case activsg2000_original_v12 \
        --methods direct,ascent_is
"""

import argparse

import matplotlib.pyplot as plt

from src.cases.registry import get_case
from src.sampling.problem import load_experiment, case_tag
from src.figures.style import (MOVE_STYLE, band, method_color, series_label,
                           series_tag, add_method_args, method_selection)
from src.figures.mu_distributions._experiment_reads import load_slice, select_series


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="case14_gen_per_bus")
    parser.add_argument("--dist", default="bernoulli")
    parser.add_argument("--target", default="max_mu")
    parser.add_argument("--snapshot", default="multi")
    parser.add_argument("--budget", type=int, default=None)
    add_method_args(parser)
    args = parser.parse_args()
    methods = method_selection(args)

    c = get_case(args.case)
    case_dir = c.results_dir
    budget = args.budget or load_experiment(case_dir, "curves")[1]["budgets"][0]
    sel = dict(dist=args.dist, target=args.target, snapshot=args.snapshot, budget=budget)
    b, truth, series, meta = load_slice(case_dir, sel)
    series = select_series(series, methods)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(b, truth, "k-", lw=2.5, label="exact (full enumeration)")
    for (method, move), est in series.items():
        band(ax, b, est, method_color(method), series_label(method, move))
        ax.lines[-1].set_linestyle(MOVE_STYLE[move])

    ax.set_yscale("log")
    ax.set_ylim(max(1e-7, truth[truth > 0].min() * 0.5), 1.5)
    ax.set_xlabel(rf"threshold $b$ on {args.target}")
    ax.set_ylabel(r"exceedance probability $P[\,\mathrm{target} \geq b\,]$")
    ax.set_title(f"{c.name}: {args.dist}, {args.target}/{args.snapshot}, "
                 f"budget {budget}, {meta['n_rep']} reps")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    out = (case_dir / f"inverse_cdf_{args.dist}_{args.target}_{args.snapshot}_b{budget}"
                      f"{series_tag(methods)}_{case_tag(c.name)}.pdf")
    fig.savefig(out)
    print(f"saved -> {out}")
    plt.show()


if __name__ == "__main__":
    main()
