"""Per-line congestion-dual survivor curves for one experiment cell (dist, target, budget).

Driven by the unified experiment matrix.  The guiding target S(x) is only the sampling
driver; what we care about are the per-line distributions.  For one fixed (dist, target,
snapshot, budget) slice this reads the raw weighted samples (experiment_samples.parquet
from run_experiments.py) and draws a grid with one subplot per line -- every sampler's
weighted survivor curve proportion(|mu_l| > mu_hat), pooled over replicates, against the
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
    uv run python -m tail_sampling.plot_per_line_survivor --case case14_gen_per_bus \
        --dist bernoulli --target max_mu --budget 640
    uv run python -m tail_sampling.plot_per_line_survivor --case case14_gen_per_bus  # all slices
    uv run python -m tail_sampling.plot_per_line_survivor --case case14_gen_per_bus --kind qq
"""

import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

from tail_sampling.problem_setup import TargetProblem, REPO_ROOT, load_experiment, CASES, case_tag
from tail_sampling.distributions import DIST_NAMES, case_p, make_dist
from tail_sampling.plot_inverse_cdf import COLORS, MOVE_STYLE, series_label

DIST_LABEL = {"bernoulli": "Bernoulli", "mrf_a": "MRF-A", "mrf_b": "MRF-B", "mrf_c": "MRF-C"}
BOOT_PCT = 95 #  bootstrap interval
_BOOT_PCT = (50 - BOOT_PCT / 2, 50 + BOOT_PCT / 2)
N_BOOT = 400
N_GRID = 250
QQ_PROBS = np.logspace(np.log10(0.9), -6, 80)          # exceedance levels for the QQ panels


def weighted_survivor(values, weights):
    """Return (sorted values, proportion of weight with value > each)."""
    order = np.argsort(values)
    v = values[order]
    cumw = np.zeros(weights.shape)
    cumw[1:] = np.cumsum(weights[order])[:-1]
    return v, 1.0 - cumw / cumw[-1]


def decimate_monotone(v, s, nbins=4000):
    """Thin a monotone survivor curve to <=~2*nbins points without visible change.

    s(v) is non-increasing, so per x-bin the first/last points are the y-extremes;
    keeping them reproduces the rendered polyline to within one bin in x (chosen
    finer than screen pixels), fabricating no information.  The dense head collapses
    while every sparse-tail point lands in its own bin and is kept verbatim."""
    if v.size <= 2 * nbins:
        return v, s
    span = v[-1] - v[0]
    if span <= 0:
        return v[[0, -1]], s[[0, -1]]
    bins = np.floor((v - v[0]) / span * nbins).astype(np.int64)
    change = bins[1:] != bins[:-1]
    keep = np.zeros(v.size, bool)
    keep[0] = keep[-1] = True
    keep[1:] |= change                                      # first point of each bin
    keep[:-1] |= change                                     # last point of each bin
    return v[keep], s[keep]


def line_order(mu, line_names, weights):
    """All lines, ordered by descending E[|mu|] = sum(mu*p) so the congested ones lead the grid."""
    idx = np.argsort(-(weights[:, None] * mu).sum(0))
    return idx, [line_names[i] for i in idx]


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
        ax.plot(v, s, color=COLORS.get(method, "gray"), lw=1.1,
                linestyle=MOVE_STYLE[move], label=series_label(method, move), rasterized=True)
    ax.set_title(f"line {line_name}", y=0.80, x=0.5, fontsize=8)


def figure_for_slice(mu, line_names, truth_w, slice_samples, title, xmax=None):
    """A grid with one per-line survivor panel; shared log-y, congested lines first.

    xmax defaults to the data's own |mu| range so a case whose dominant line greatly
    exceeds the synthetic cases' scale (e.g. Texas ctg:68:18 ~2000) is not clipped."""
    xmax = float(mu.max()) * 1.02 if xmax is None else xmax
    idx, names = line_order(mu, line_names, truth_w)
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


def weighted_quantiles(values, weights, probs):
    """Value q with weighted P[|mu_l| > q] = p, for each exceedance level p in probs."""
    v, s = weighted_survivor(values, weights)
    return np.interp(probs, s[::-1], v[::-1])


def qq_panel(ax, line_idx, line_name, mu, truth_w, slice_samples, probs):
    """One line's QQ: sampler quantile vs exact truth quantile at matched exceedance levels."""
    qt = weighted_quantiles(mu[:, line_idx], truth_w, probs)
    hi = qt.max()
    for (method, move), g in slice_samples.groupby(["method", "ascent_move"], sort=True):
        ids, w = g["id"].to_numpy(), g["weight"].to_numpy()
        qs = weighted_quantiles(mu[ids, line_idx], w, probs)
        ax.plot(qt, qs, color=COLORS.get(method, "gray"), lw=1.0,
                linestyle=MOVE_STYLE[move], label=series_label(method, move), rasterized=True)
        hi = max(hi, qs.max())
    mumax = mu.max()
    ax.plot([0, mumax], [0, mumax], color="k", lw=0.5, ls="-", label="identity (= truth)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"line {line_name}", y=0.80, x=0.5, fontsize=8)


def qq_figure_for_slice(mu, line_names, truth_w, slice_samples, title, probs=QQ_PROBS):
    """A grid of per-line QQ panels; each panel has its own linear scale and 45-degree line."""
    idx, names = line_order(mu, line_names, truth_w)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="case14_gen_per_bus")
    parser.add_argument("--dists", default=",".join(DIST_NAMES))
    parser.add_argument("--targets", default="max_mu,sum_mu")
    parser.add_argument("--snapshot", default="multi")
    parser.add_argument("--budgets", default=None)
    parser.add_argument("--kind", default="survivor", choices=["survivor", "qq"])
    args = parser.parse_args()

    case_dir = REPO_ROOT / "results" / args.case
    samples, meta = load_experiment(case_dir, "samples")
    tc = case_dir / "target_cache.parquet"
    ngen = int(json.loads(pq.read_schema(tc).metadata[b"run_meta"])["ngen"])
    p = case_p(CASES[args.case])

    problems = {d: TargetProblem(tc, dist=make_dist(d, ngen, p)) for d in args.dists.split(",")}
    mu, line_names = next(iter(problems.values())).mu_table()
    budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else meta["budgets"]

    make_fig = qq_figure_for_slice if args.kind == "qq" else figure_for_slice
    stem = "per_line_qq" if args.kind == "qq" else "per_line_survivor"
    grouped = samples.groupby(["dist", "target", "snapshot", "budget"], sort=False)
    for dist in args.dists.split(","):
        for target in args.targets.split(","):
            for budget in budgets:
                key = (dist, target, args.snapshot, budget)
                if key not in grouped.groups:
                    continue
                sl = grouped.get_group(key)
                title = (f"{args.case}: per-line tail  |  {DIST_LABEL[dist]}, guide={target}, "
                         f"budget {budget}, {meta['n_rep']} reps")
                fig = make_fig(mu, line_names, problems[dist].w, sl, title)
                out = case_dir / f"{stem}_{dist}_{target}_{args.snapshot}_b{budget}_{case_tag(args.case)}.pdf"
                fig.savefig(out)
                print(f"saved -> {out}")
                #plt.close(fig)
                break
            break
        break
    plt.show()


if __name__ == "__main__":
    main()
