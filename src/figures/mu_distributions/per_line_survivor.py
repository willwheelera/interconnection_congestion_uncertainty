"""Per-line congestion-dual survivor curves for one experiment cell (dist, target, budget).

Driven by the unified experiment matrix.  The guiding target S(x) is only the sampling
driver; what we care about are the per-line distributions.  For one fixed (dist, target,
snapshot, budget) slice this reads the raw weighted samples (experiment_samples.parquet
from run_experiments.py) and draws a grid with one subplot per line -- each selected
sampler's weighted survivor curve proportion(|mu_l| > mu_hat) (`direct` and ascent-IS at
beta 0.2 by default; --methods selects, see figures.style), pooled over replicates, against the
exact nominal-weighted truth (full enumeration).  A good sampler tracks the black truth
into the congested tail.  One figure per slice; main loops the requested dist x target x
budget slices, writing one PDF each.  The replicate-bootstrap helpers (rep_numerators,
bootstrap_band) are kept for an optional 95% error band but are not drawn by default --
the drawn curve is the pooled point estimate.

--kind qq draws the transposed view instead: a per-line QQ grid plotting each sampler's
weighted quantile of |mu_l| against the exact truth quantile at matched exceedance levels,
with a 45-degree identity line.  Points off the diagonal at large |mu| show a sampler that
recovers the wrong tail value (or never reaches it), complementing the survivor overlay.

Usage:
    uv run python -m src.figures.mu_distributions.per_line_survivor --case case14_gen_per_bus \
        --dists bernoulli --targets max_mu --budgets 640
    uv run python -m src.figures.mu_distributions.per_line_survivor --case case14_gen_per_bus  # every law
    uv run python -m src.figures.mu_distributions.per_line_survivor --case case14_gen_per_bus --kind qq
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from src.cases.registry import get_case
from src.sampling.problem import load_experiment, case_tag, create_problem
from src.sampling.distributions import DIST_NAMES, DIST_LABEL
from src.figures.style import (DEFAULT_METHODS, MOVE_STYLE, method_color, series_label,
                           select_methods, series_tag, add_method_args, method_selection)
from src.figures.mu_distributions._line_stats import (weighted_survivor, decimate_monotone, select_lines,
                                         weighted_quantiles)
from src.figures.mu_distributions._panel_layout import panel_title

BOOT_PCT = 95 #  bootstrap interval
_BOOT_PCT = (50 - BOOT_PCT / 2, 50 + BOOT_PCT / 2)
N_BOOT = 400
N_GRID = 250
QQ_PROBS = np.logspace(np.log10(0.9), -6, 80)          # exceedance levels for the QQ panels


def survivor_on_grid(values, weights, grid):
    """Self-normalized weighted survivor proportion(values > grid) on a fixed grid."""
    return (weights[:, None] * (values[:, None] > grid[None, :])).sum(0) / weights.sum()


def rep_numerators(mu_l, group, grid):
    """Per-replicate weighted exceedance numerator and total weight, on a fixed grid."""
    reps = np.sort(group["rep"].unique())
    num = np.zeros((reps.size, grid.size))
    tot = np.zeros(reps.size)
    for j, r in enumerate(reps):
        gr = group[group["rep"] == r]
        vals, w = mu_l[gr["id"].to_numpy()], gr["weight"].to_numpy()
        num[j] = (w[:, None] * (vals[:, None] > grid[None, :])).sum(0)
        tot[j] = w.sum()
    return num, tot


def bootstrap_band(num, tot, rng):
    """Pooled estimate and its bootstrap band from resampling replicates with replacement."""
    idx = rng.integers(0, num.shape[0], size=(N_BOOT, num.shape[0]))
    boot = num[idx].sum(1) / tot[idx].sum(1)[:, None]
    lo, hi = np.percentile(boot, _BOOT_PCT, axis=0)
    return num.sum(0) / tot.sum(), lo, hi


def panel(ax, line_idx, line_name, mu, truth_w, slice_samples):
    """Truth plus every sampler's pooled weighted survivor curve for one line."""
    v, s = decimate_monotone(*weighted_survivor(mu[:, line_idx], truth_w))
    ax.plot(v, s, color="k", lw=1.5, label="truth", rasterized=True)
    for (method, move), g in slice_samples.groupby(["method", "ascent_move"], sort=True):
        ids, w = g["id"].to_numpy(), g["weight"].to_numpy()
        v, s = decimate_monotone(*weighted_survivor(mu[ids, line_idx], w))
        ax.plot(v, s, color=method_color(method), lw=1.1,
                linestyle=MOVE_STYLE[move], label=series_label(method, move), rasterized=True)
    panel_title(ax, f"line {line_name}", y=0.80)      # curve fills the top; sit the label lower


def figure_for_slice(mu, line_names, truth_w, slice_samples, title, xmax=None):
    """A grid with one per-line survivor panel; shared log-y, congested lines first.

    xmax defaults to the data's own |mu| range so a case whose dominant line greatly
    exceeds the synthetic cases' scale (e.g. Texas ctg:68:18 ~2000) is not clipped."""
    xmax = float(mu.max()) * 1.02 if xmax is None else xmax
    idx, names = select_lines(mu, line_names, truth_w)
    ncol = 5
    nrow = int(np.ceil(len(idx) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 1.9, nrow * 1.7),
                            sharex=True, sharey=True, squeeze=False)
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_ylim(1e-6, 1.5)
    for line_idx, line_name, ax in zip(idx, names, axs.ravel()):
        panel(ax, line_idx, line_name, mu, truth_w, slice_samples)
        ax.minorticks_off()
    for ax in axs.ravel()[len(idx):]:
        ax.set_visible(False)
    axs[0, 0].set_xlim(-1, min(axs[0, 0].get_xlim()[1], xmax))
    axs[0, 0].set_yticks([10**(-n) for n in [1, 3, 5]])

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.suptitle(title, y=0.995)
    fig.supxlabel(r"congestion threshold $\hat{\mu}$")
    fig.supylabel(r"$P[\,|\mu_l| > \hat{\mu}\,]$")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=4, frameon=True, fontsize=6, columnspacing=1.0,
               handlelength=1.6, handletextpad=0.5)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.085, top=0.87,
                        wspace=0.18, hspace=0.22)
    return fig


def qq_panel(ax, line_idx, line_name, mu, truth_w, slice_samples, probs):
    """One line's QQ: sampler quantile vs exact truth quantile at matched exceedance levels."""
    qt = weighted_quantiles(mu[:, line_idx], truth_w, probs)
    hi = qt.max()
    for (method, move), g in slice_samples.groupby(["method", "ascent_move"], sort=True):
        ids, w = g["id"].to_numpy(), g["weight"].to_numpy()
        qs = weighted_quantiles(mu[ids, line_idx], w, probs)
        ax.plot(qt, qs, color=method_color(method), lw=1.0,
                linestyle=MOVE_STYLE[move], label=series_label(method, move), rasterized=True)
        hi = max(hi, qs.max())
    mumax = mu.max()
    ax.plot([0, mumax], [0, mumax], color="k", lw=0.5, ls="-", label="identity (= truth)")
    ax.set_aspect("equal", adjustable="box")
    panel_title(ax, f"line {line_name}", y=0.80)      # curve fills the top; sit the label lower


def qq_figure_for_slice(mu, line_names, truth_w, slice_samples, title, probs=QQ_PROBS):
    """A grid of per-line QQ panels; each panel has its own linear scale and 45-degree line."""
    idx, names = select_lines(mu, line_names, truth_w)
    ncol = 5
    nrow = int(np.ceil(len(idx) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 1.9, nrow * 1.9), squeeze=False, sharex=True, sharey=True)
    for line_idx, line_name, ax in zip(idx, names, axs.ravel()):
        qq_panel(ax, line_idx, line_name, mu, truth_w, slice_samples, probs)
        ax.minorticks_off()
        ax.tick_params(labelsize=6)
    for ax in axs.ravel()[len(idx):]:
        ax.set_visible(False)

    handles, labels = axs.ravel()[0].get_legend_handles_labels()
    fig.suptitle(title, y=0.995)
    fig.supxlabel(r"exact quantile of $|\mu_l|$ (truth)")
    fig.supylabel(r"sampler quantile of $|\mu_l|$")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=4, frameon=True, fontsize=6, columnspacing=1.0,
               handlelength=1.6, handletextpad=0.5)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.075, top=0.87,
                        wspace=0.30, hspace=0.30)
    return fig


def run(case, dists, targets=None, budgets=None, snapshot="multi", kind="survivor",
        methods=DEFAULT_METHODS):
    """One figure per (dist, target, budget) slice present in the experiment; returns them.

    A slice the experiment never ran is skipped rather than drawn empty."""
    c = get_case(case)
    case, case_dir = c.name, c.results_dir
    samples, meta = load_experiment(case_dir, "samples")
    samples = samples[samples["method"].isin(
        select_methods(samples["method"].unique(), methods))]
    problems = {d: create_problem(c, d) for d in dists}
    mu, line_names = next(iter(problems.values())).mu_table()
    budgets = budgets or meta["budgets"]

    make_fig = qq_figure_for_slice if kind == "qq" else figure_for_slice
    stem = "per_line_qq" if kind == "qq" else "per_line_survivor"
    grouped = samples.groupby(["dist", "target", "snapshot", "budget"], sort=False)
    figs = []
    for dist in dists:
        for target in targets:
            for budget in budgets:
                key = (dist, target, snapshot, budget)
                if key not in grouped.groups:
                    continue
                title = (f"{case}: per-line tail  |  {DIST_LABEL[dist]}, guide={target}, "
                         f"budget {budget}, {meta['n_rep']} reps")
                fig = make_fig(mu, line_names, problems[dist].w, grouped.get_group(key), title)
                out = (case_dir / f"{stem}_{dist}_{target}_{snapshot}_b{budget}"
                                  f"{series_tag(methods)}_{case_tag(case)}.pdf")
                fig.savefig(out)
                print(f"saved -> {out}")
                figs.append(fig)
    return figs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="activsg2000_v12_lev3a3")
    parser.add_argument("--dists", default=",".join(DIST_NAMES))
    parser.add_argument("--targets", default="sum_mu")
    parser.add_argument("--snapshot", default="multi")
    parser.add_argument("--budgets", default=None)
    parser.add_argument("--kind", default="survivor", choices=["survivor", "qq"])
    add_method_args(parser)
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else None
    run(args.case, args.dists.split(","), args.targets.split(","), budgets,
        args.snapshot, args.kind, method_selection(args))
    plt.show()


if __name__ == "__main__":
    main()
