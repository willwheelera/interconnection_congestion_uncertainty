"""Per-line box-and-whisker of the direct-sampling tail statistics.

Two companion plots to plot_line_exceed_strip.py, sharing its per-line panel layout
(congested lines first, one figure per nominal law, a direct box per sampling-budget cell,
exact nominal-weighted truth dashed):

  exceed      the tail probability P[|mu_l| > c]
  percentile  the q-quantile of |mu_l| (default 99th pct), its inverse

Both were separate scripts; this generates the same figures together.  Direct draws are
target-independent, so the x-axis groups sampling budgets.  For the percentile, numerical
near-binding dust is floored to 0 so a line that congests less than 1-q of the time reads as
a clean 0 rather than 1e-8 noise.

Usage:
    uv run python -m src.figures.tail.direct_box --case case14_gen_per_bus
    uv run python -m src.figures.tail.direct_box --case case14_gen_per_bus \
        --threshold 100 --quantile 0.99 --budgets 640,2560
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.sampling.problem import TargetProblem, CASES, case_tag
from src.sampling.distributions import case_dist, DIST_LABEL
from src.figures.tail.line_exceed_strip import truth_probabilities
from src.figures.tail.direct_common import (load_direct_context, direct_slice, panel_grid,
                                              box_legend, box_draw)


def direct_exceed_frame(samples, mu, line_names, dist, budgets, threshold, target="max_mu"):
    """Long per-(budget, rep, line) direct estimate of P[|mu_l| > threshold]."""
    sl = direct_slice(samples, dist, budgets, target)
    rows = []
    for (budget, rep), g in sl.groupby(["budget", "rep"], sort=True):
        w, ids = g["weight"].to_numpy(), g["id"].to_numpy()
        prob = (w[:, None] * (mu[ids] > threshold)).sum(0)
        rows.append(pd.DataFrame({"budget": str(budget), "rep": rep,
                                  "line": line_names, "prob": prob}))
    return pd.concat(rows, ignore_index=True)


def weighted_quantile(values, weights, q):
    """The q-quantile of values under weights (exact nominal-weighted truth)."""
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    return np.interp(q, cw / cw[-1], values[order])


def truth_quantiles(mu, w, q):
    """Exact nominal-weighted q-quantile of |mu_l| for every line."""
    return np.array([weighted_quantile(mu[:, l], w, q) for l in range(mu.shape[1])])


def direct_percentile_frame(samples, mu, line_names, dist, budgets, q, target="max_mu"):
    """Long per-(budget, rep, line) direct estimate of the q-quantile of |mu_l|.

    Direct draws carry uniform weight, so the weighted quantile is the plain empirical
    quantile of the drawn configs' |mu_l|."""
    sl = direct_slice(samples, dist, budgets, target)
    rows = []
    for (budget, rep), g in sl.groupby(["budget", "rep"], sort=True):
        val = np.quantile(mu[g["id"].to_numpy()], q, axis=0)
        rows.append(pd.DataFrame({"budget": str(budget), "rep": rep,
                                  "line": line_names, "value": val}))
    return pd.concat(rows, ignore_index=True)


def plot_exceed(case, samples, meta, mu, line_names, w, dist, budgets, xorder, threshold, case_dir):
    """One figure: per-line box of P[|mu_l| > threshold] for a single nominal law."""
    truth_p = truth_probabilities(mu, w, threshold)
    order = np.argsort(-truth_p)
    shown = [line_names[i] for i in order if truth_p[i] > 1e-4]
    truth_by_name = {line_names[i]: float(truth_p[i]) for i in range(len(line_names))}
    frame = direct_exceed_frame(samples, mu, line_names, dist, budgets, threshold)
    title = (f"{case}: direct  "
             rf"$P[\,|\mu_l| > {threshold:g}\,]$  |  "
             f"{DIST_LABEL[dist]}, {meta['n_rep']} reps")
    ylabel = rf"$\hat{{P}}[\,|\mu_l| > {threshold:g}\,]$"
    draw = box_draw(frame, xorder, truth_by_name, "prob")
    fig, _ = panel_grid(shown, draw, box_legend(), title, "sampling budget", ylabel)
    out = case_dir / f"direct_exceed_box_thr{int(threshold)}_{dist}_{case_tag(case)}.pdf"
    fig.savefig(out)
    print(f"saved -> {out}")
    return fig


def plot_percentile(case, samples, meta, mu, line_names, w, dist, budgets, xorder,
                    quantile, mu_floor, case_dir):
    """One figure: per-line box of the q-quantile of |mu_l| for a single nominal law."""
    mu = np.where(mu < mu_floor, 0.0, mu)                    # drop numerical near-binding dust
    truth_q = truth_quantiles(mu, w, quantile)
    order = np.argsort(-truth_q)
    shown = [line_names[i] for i in order if truth_q[i] > mu_floor]
    truth_by_name = {line_names[i]: float(truth_q[i]) for i in range(len(line_names))}
    frame = direct_percentile_frame(samples, mu, line_names, dist, budgets, quantile)
    title = (f"{case}: direct  {quantile * 100:g}th-percentile "
             rf"$|\mu_l|$  |  {DIST_LABEL[dist]}, {meta['n_rep']} reps")
    ylabel = rf"{quantile * 100:g}th-percentile $|\mu_l|$"
    draw = box_draw(frame, xorder, truth_by_name, "value")
    fig, _ = panel_grid(shown, draw, box_legend(), title, "sampling budget", ylabel)
    out = case_dir / f"direct_percentile_box_q{int(quantile * 100)}_{dist}_{case_tag(case)}.pdf"
    fig.savefig(out)
    print(f"saved -> {out}")
    return fig


def run(case, dists, budgets=None, threshold=100.0, quantile=0.99, mu_floor=1e-2):
    """The exceedance and percentile box figures for each nominal law; returns both, in order."""
    case_dir, samples, meta, tc, ngen, p = load_direct_context(case)
    budgets = budgets or meta["budgets"]
    xorder = [str(b) for b in budgets]

    figs = []
    for dist in dists:
        problem = TargetProblem(tc, dist=case_dist(CASES[case], dist))
        mu, line_names = problem.mu_table()
        figs.append(plot_exceed(case, samples, meta, mu, line_names, problem.w, dist,
                                budgets, xorder, threshold, case_dir))
        figs.append(plot_percentile(case, samples, meta, mu, line_names, problem.w, dist,
                                    budgets, xorder, quantile, mu_floor, case_dir))
    return figs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="activsg2000_original_v12")
    parser.add_argument("--dists", default="bernoulli,mrf_a,mrf_b,mrf_c")
    parser.add_argument("--budgets", default=None)
    parser.add_argument("--threshold", type=float, default=100.0)
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--mu-floor", type=float, default=1e-2,
                        help="zero |mu| below this (numerical near-binding dust, not congestion)")
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else None
    run(args.case, args.dists.split(","), budgets, args.threshold, args.quantile,
        args.mu_floor)
    plt.show()


if __name__ == "__main__":
    main()
