"""Inverse-CDF (threshold-probability) plot for one slice of the experiment matrix.

Reads a case's experiment_curves.parquet (run_experiments.py) and, for one fixed
(dist, target, snapshot, budget) slice, overlays every sampler's estimated exceedance
curve P[target >= b] -- median with a 10-90 band over replicates -- against the exact
enumerated truth (black).  Ascent move strategies share the sampler's color and differ
by line style, so all/best/top5 read as one family.

Usage:
    uv run python -m tail_sampling.plot_inverse_cdf --case case14_gen_per_bus
    uv run python -m tail_sampling.plot_inverse_cdf --case case14_gen_per_bus \
        --dist mrf_b --target sum_mu --budget 640
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from tail_sampling.problem_setup import REPO_ROOT, load_experiment, case_tag
from tail_sampling.plot_style import COLORS, LABELS, band

COLORS = {**COLORS, "cross_entropy": "tab:brown", "truth": "k"}
MOVE_STYLE = {"na": "-", "all": "-", "best": ":", "top5": "--"}


def load_slice(case_dir, sel):
    """Return (thresholds, truth, {(method, move): (n_rep, n_thr)}, meta) for one slice."""
    df, meta = load_experiment(case_dir, "curves")
    for key, val in sel.items():
        df = df[df[key] == val]
    thresholds = np.sort(df["threshold"].unique())
    truth = (df[df.method == "truth"].set_index("threshold")["estimate"]
             .reindex(thresholds).to_numpy())
    series = {}
    for (m, mv), g in df[df.method != "truth"].groupby(["method", "ascent_move"]):
        series[(m, mv)] = (g.pivot(index="rep", columns="threshold", values="estimate")
                           .reindex(columns=thresholds).to_numpy())
    return thresholds, truth, series, meta


def series_label(method, move):
    """Sampler label, annotated with the ascent move when it is not the default."""
    return LABELS.get(method, method) + ("" if move == "na" else f" ({move})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="case14_gen_per_bus")
    parser.add_argument("--dist", default="bernoulli")
    parser.add_argument("--target", default="max_mu")
    parser.add_argument("--snapshot", default="multi")
    parser.add_argument("--budget", type=int, default=None)
    args = parser.parse_args()

    case_dir = REPO_ROOT / "results" / args.case
    budget = args.budget or load_experiment(case_dir, "curves")[1]["budgets"][0]
    sel = dict(dist=args.dist, target=args.target, snapshot=args.snapshot, budget=budget)
    b, truth, series, meta = load_slice(case_dir, sel)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(b, truth, "k-", lw=2.5, label="exact (full enumeration)")
    for (method, move), est in series.items():
        band(ax, b, est, COLORS.get(method, "gray"), series_label(method, move))
        ax.lines[-1].set_linestyle(MOVE_STYLE[move])

    ax.set_yscale("log")
    ax.set_ylim(max(1e-7, truth[truth > 0].min() * 0.5), 1.5)
    ax.set_xlabel(rf"threshold $b$ on {args.target}")
    ax.set_ylabel(r"exceedance probability $P[\,\mathrm{target} \geq b\,]$")
    ax.set_title(f"{args.case}: {args.dist}, {args.target}/{args.snapshot}, "
                 f"budget {budget}, {meta['n_rep']} reps")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    out = case_dir / f"inverse_cdf_{args.dist}_{args.target}_{args.snapshot}_b{budget}_{case_tag(args.case)}.pdf"
    fig.savefig(out)
    print(f"saved -> {out}")
    plt.show()


if __name__ == "__main__":
    main()
