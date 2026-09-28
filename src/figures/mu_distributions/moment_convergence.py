"""Convergence of the sampled mean, variance, and skewness of S with the solver budget.

Direct sampling estimates the whole nominal-weighted distribution of the target S = max_b
|mu_b|, not just a tail probability, so this figure tracks the first three moments of that
distribution as the budget grows.  For every (nominal law, budget, replicate) the weighted
sample mean, variance, and skewness of S are computed from that replicate's draws; a panel
per moment plots, versus budget, the replicate spread as error bars (mean +- 1 sd across
replicates) around the replicate mean, one colored series per law, against each law's exact
moment (the enumerated nominal-weighted value, a dashed line in the law's color).

S is the physics target and is shared across laws; a law enters only through which configs
its draws land on and their weights, so the series separate exactly where the laws weight the
tail of S differently.  Error bars shrinking toward the dashed line as budget grows is the
estimator converging; a bar that stays wide is a moment the budget does not yet pin down.

Usage:
    uv run python -m src.figures.mu_distributions.moment_convergence --case activsg2000_v12_lev3a3
    uv run python -m src.figures.mu_distributions.moment_convergence --case activsg2000_v12_lev3a3 \
        --dists bernoulli,mrf_a --target sum_mu
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.sampling.problem import case_tag, create_problem
from src.sampling.distributions import DIST_NAMES, DIST_LABEL
from src.figures.style import DIST_COLOR
from src.figures.mu_distributions._experiment_reads import direct_samples, direct_slice
from src.figures.mu_distributions._panel_layout import FIG_W, FONT

MOMENTS = ["mean", "variance", "skewness"]
DODGE = 0.035                     # multiplicative x offset per law, so error bars do not overlap


def weighted_moments(vals, weights):
    """Weighted mean, variance, and skewness of vals under weights (weights need not sum to 1)."""
    w = weights / weights.sum()
    mean = float(w @ vals)
    d = vals - mean
    var = float(w @ d ** 2)
    skew = float((w @ d ** 3) / var ** 1.5) if var > 0 else 0.0
    return mean, var, skew


def replicate_moments(samples, S, dists, budgets):
    """Long per-(dist, budget, rep) frame of the sampled mean, variance, skewness of S."""
    rows = []
    for dist in dists:
        sl = direct_slice(samples, dist, budgets)
        for (budget, rep), g in sl.groupby(["budget", "rep"], sort=True):
            m, v, s = weighted_moments(S[g["id"].to_numpy()], g["weight"].to_numpy())
            rows.append({"dist": dist, "budget": budget, "mean": m,
                         "variance": v, "skewness": s})
    return pd.DataFrame(rows)


def exact_moments(problems, S):
    """Each law's exact nominal-weighted mean, variance, skewness of S (full enumeration)."""
    return {d: dict(zip(MOMENTS, weighted_moments(S, pr.w))) for d, pr in problems.items()}


def moment_panel(ax, frame, exact, dists, budgets, moment):
    """One moment versus budget: per-law error bars (mean +- sd over reps) and exact dashes."""
    for k, d in enumerate(dists):
        g = frame[frame["dist"] == d].groupby("budget")[moment]
        mean, sd = g.mean().reindex(budgets), g.std().reindex(budgets)
        x = np.array(budgets, float) * (1 + DODGE * (k - (len(dists) - 1) / 2))
        ax.errorbar(x, mean.to_numpy(), yerr=sd.to_numpy(), color=DIST_COLOR[d], lw=1.0,
                    marker="o", ms=2.5, capsize=2, elinewidth=0.8, zorder=3)
        ax.axhline(exact[d][moment], color=DIST_COLOR[d], ls="--", lw=0.7, zorder=2)
    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_title(moment, fontsize=FONT["panel_title"])
    ax.tick_params(labelsize=FONT["tick"])
    ax.margins(x=0.12)


def build_figure(frame, exact, dists, budgets, title):
    """A one-row panel-per-moment figure with a shared law legend."""
    fig, axs = plt.subplots(1, len(MOMENTS), figsize=(FIG_W, 2.3))
    for ax, moment in zip(axs, MOMENTS):
        moment_panel(ax, frame, exact, dists, budgets, moment)
    handles = [Line2D([0], [0], color=DIST_COLOR[d], marker="o", ms=3, label=DIST_LABEL[d])
               for d in dists]
    handles.append(Line2D([0], [0], color="k", ls="--", label="exact"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.90),
               ncol=len(handles), frameon=True, fontsize=FONT["legend"], handletextpad=0.4,
               columnspacing=1.0, borderpad=0.3)
    fig.suptitle(title, y=0.995, fontsize=FONT["suptitle"])
    fig.supxlabel("sampling budget", y=0.02, fontsize=FONT["axlabel"])
    fig.tight_layout(rect=(0, 0.03, 1, 0.84))
    return fig


def run(case, dists=DIST_NAMES, budgets=None, target="max_mu"):
    """Sample-moment convergence figure for one case; returns it.

    S is read once from any law's problem (the target is law-independent); each law's exact
    moments and its replicate estimates are computed against that same S.
    """
    c, samples, meta = direct_samples(case)
    budgets = budgets or meta["budgets"]
    problems = {d: create_problem(c, d, target=target) for d in dists}
    S = next(iter(problems.values())).S
    frame = replicate_moments(samples, S, dists, budgets)
    exact = exact_moments(problems, S)

    title = f"{c.name.split('_')[0]}: moments of S={target}  |  {meta['n_rep']} reps"
    fig = build_figure(frame, exact, dists, budgets, title)
    out = c.results_dir / f"moment_convergence_{target}_{case_tag(c.name)}.pdf"
    fig.savefig(out)
    print(f"saved -> {out}")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="activsg2000_v12_lev3a3")
    parser.add_argument("--dists", default=",".join(DIST_NAMES))
    parser.add_argument("--budgets", default=None)
    parser.add_argument("--target", default="max_mu")
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else None
    run(args.case, args.dists.split(","), budgets, args.target)
    plt.show()


if __name__ == "__main__":
    main()
