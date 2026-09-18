"""Per-line box-and-whisker of the direct-sampling tail statistics.

Two companion plots to plot_line_exceed_strip.py, sharing its per-line panel layout
(congested lines first, one figure per nominal law, a direct box per sampling-budget cell,
exact nominal-weighted truth dashed):

  exceed      the tail probability P[|mu_l| > c]
  percentile  the q-quantile of |mu_l| (default 99th pct), its inverse

Both were separate scripts; this generates the same figures together.  Direct draws are
target-independent, so the x-axis groups sampling budgets.  Each figure shows the n_lines
most-congested lines (--n-lines), ranked by the statistic it plots; lines that never congest
are dropped so no empty panels are drawn.  The full |mu_l| distribution behind these summaries
is its own figure, direct_distribution.py.

Usage:
    uv run python -m src.figures.mu_distributions.direct_box --case case14_gen_per_bus
    uv run python -m src.figures.mu_distributions.direct_box --case case14_gen_per_bus \
        --threshold 100 --quantile 0.99 --budgets 640,2560
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.sampling.problem import case_tag, create_problem
from src.sampling.distributions import DIST_LABEL, DIST_NAMES
from src.figures.style import COLORS
from src.figures.mu_distributions._line_stats import (exceedance_prob, quantile_abs_mu,
                                                      select_lines)
from src.figures.mu_distributions._panel_layout import panel_grid, one_sigfig_yaxis
from src.figures.mu_distributions._experiment_reads import direct_samples, direct_slice


def box_legend():
    """Legend handles shared by the two direct box plots."""
    return [Patch(facecolor=COLORS["direct"], alpha=0.55, label="direct replicates"),
            Line2D([0], [0], color="k", ls="--", label="exact truth")]


def box_panel(ax, sub, xorder, truth, ycol):
    """One line's panel: a direct box per budget cell plus the dashed exact truth."""
    sns.boxplot(data=sub, x="budget", y=ycol, order=xorder, color=COLORS["direct"],
                width=0.5, linewidth=0.8, fliersize=2, ax=ax,
                boxprops=dict(alpha=0.55), medianprops=dict(color="k"))
    ax.axhline(truth, color="k", ls="--", lw=1.0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", length=0)
    hi = max(float(truth), float(sub[ycol].max()))          # keep truth line and fliers in frame
    ax.set_ylim(0, hi * 1.2 if hi > 0 else None)
    one_sigfig_yaxis(ax)


def box_draw(frame, xorder, truth_by_name, ycol):
    """Build the panel_grid callback that boxes one line's cells from a long frame."""
    def draw(ax, name):
        box_panel(ax, frame[frame["line"] == name], xorder, truth_by_name[name], ycol)
    return draw


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


def plot_exceed(case, samples, meta, mu, line_names, w, dist, budgets, xorder, threshold,
                n_lines, case_dir):
    """One figure: per-line box of P[|mu_l| > threshold] for a single nominal law."""
    truth_p = exceedance_prob(mu, w, threshold)
    _, shown = select_lines(mu, line_names, w, n=n_lines)
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
                    quantile, n_lines, case_dir):
    """One figure: per-line box of the q-quantile of |mu_l| for a single nominal law."""
    truth_q = quantile_abs_mu(mu, w, quantile)
    _, shown = select_lines(mu, line_names, w, n=n_lines)
    truth_by_name = {line_names[i]: float(truth_q[i]) for i in range(len(line_names))}
    frame = direct_percentile_frame(samples, mu, line_names, dist, budgets, quantile)
    title = (f"{case}: direct  {quantile * 100:g}th-percentile "
             rf"$|\mu_l|$  |  {DIST_LABEL[dist]}, {meta['n_rep']} reps")
    ylabel = rf"{quantile * 100:g}th-percentile $|\mu_l|$"
    draw = box_draw(frame, xorder, truth_by_name, "value")
    fig, axs = panel_grid(shown, draw, box_legend(), title, "sampling budget", ylabel, sharey=True)
    axs[0, 0].set_ylim([0, frame["value"].max() * 1.1])
    out = case_dir / f"direct_percentile_box_q{int(quantile * 100)}_{dist}_{case_tag(case)}.pdf"
    fig.savefig(out)
    print(f"saved -> {out}")
    return fig


def run(case, dists, budgets=None, threshold=100.0, quantile=0.99, n_lines=20):
    """Exceedance and percentile box for each nominal law; returns both, in order."""
    c, samples, meta = direct_samples(case)
    case, case_dir = c.name, c.results_dir
    budgets = budgets or meta["budgets"]
    xorder = [str(b) for b in budgets]

    figs = []
    for dist in dists:
        problem = create_problem(c, dist)
        mu, line_names = problem.mu_table()
        figs.append(plot_exceed(case, samples, meta, mu, line_names, problem.w, dist,
                                budgets, xorder, threshold, n_lines, case_dir))
        #figs.append(plot_percentile(case, samples, meta, mu, line_names, problem.w, dist,
        #                            budgets, xorder, quantile, n_lines, case_dir))
    return figs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="activsg2000_v12_lev3a3")
    parser.add_argument("--dists", default=",".join(DIST_NAMES))
    parser.add_argument("--budgets", default=None)
    parser.add_argument("--threshold", type=float, default=200.0)
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--n-lines", type=int, default=10,
                        help="number of most-congested lines to show per figure")
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else None
    run(args.case, args.dists.split(","), budgets, args.threshold, args.quantile, args.n_lines)
    plt.show()


if __name__ == "__main__":
    main()
