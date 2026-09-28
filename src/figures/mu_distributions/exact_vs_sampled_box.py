"""Per-line box-and-whisker of the |mu_l| distribution: exact truth vs direct-sampling runs.

One axes, not a panel grid: the x-axis is line rank (the top --n-lines lines by descending
E[|mu_l|] under the nominal weights), and at each rank sit a group of boxes -- the exact
nominal-weighted distribution of |mu_l|, then one box per sampling budget.  Each box shows
the weighted mean (marker), the median, the 25-75 quantile box, and whiskers at the central
--coverage interval (95% by default).  This reads the whole per-line distribution at a glance
and shows how a direct-sampling run converges on its shape as the budget grows, where the
per-line survivor curves (per_line_survivor.py) show one line at a time.

The exact stats are weighted by the full enumerated nominal law.  Each budget's stats come
from a single direct run of that many draws (one replicate, --rep), so a budget box is the
empirical distribution a run of that size actually recovers -- not pooled across replicates,
which would wash the budget out.

Usage:
    uv run python -m src.figures.mu_distributions.exact_vs_sampled_box --case activsg2000_v12_lev3a3
    uv run python -m src.figures.mu_distributions.exact_vs_sampled_box --dist mrf_a \
        --budgets 640,2560,10240 --n-lines 12 --yscale symlog
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.sampling.problem import case_tag, create_problem
from src.sampling.distributions import DIST_LABEL, DIST_NAMES
from src.figures.style import budget_colors
from src.figures.mu_distributions._line_stats import weighted_quantile, select_lines
from src.figures.mu_distributions._experiment_reads import direct_samples, direct_slice

EXACT_COLOR = "0.6"          # gray fill for the exact-truth boxes


def weighted_box_stats(values, weights, coverage, label):
    """A matplotlib bxp stat dict for one line: weighted mean/median/quartiles/whiskers."""
    lo = (1.0 - coverage) / 2.0
    w = weights / weights.sum()
    return {"label": label,
            "mean": float(w @ values),
            "med": float(weighted_quantile(values, w, 0.5)),
            "q1": float(weighted_quantile(values, w, 0.25)),
            "q3": float(weighted_quantile(values, w, 0.75)),
            "whislo": float(weighted_quantile(values, w, lo)),
            "whishi": float(weighted_quantile(values, w, 1.0 - lo)),
            "fliers": []}


def exact_stats(mu, w, shown_idx, names, coverage):
    """Exact nominal-weighted box stats for each shown line, in rank order."""
    return [weighted_box_stats(mu[:, l], w, coverage, names[l]) for l in shown_idx]


def sampled_stats(mu, ids, weights, shown_idx, names, coverage):
    """Box stats of each shown line from one direct run's draws (ids, importance weights)."""
    drawn = mu[ids]
    return [weighted_box_stats(drawn[:, l], weights, coverage, names[l]) for l in shown_idx]


def one_run_draws(samples, dist, budget, rep, target, snapshot):
    """One replicate's direct draws at one budget: (ids, weights)."""
    sl = direct_slice(samples, dist, [budget], target, snapshot)
    sl = sl[sl["rep"] == rep]
    return sl["id"].to_numpy(), sl["weight"].to_numpy()


def draw_boxes(ax, stats, positions, width, color):
    """One bxp call: filled boxes with a mean marker, at the given x positions."""
    ax.bxp(stats, positions=positions, widths=width, showmeans=True, showfliers=False,
           patch_artist=True, meanline=False,
           boxprops=dict(facecolor=color, alpha=0.6, edgecolor="k", linewidth=0.6),
           medianprops=dict(color="k", linewidth=0.9),
           whiskerprops=dict(color="k", linewidth=0.6),
           capprops=dict(color="k", linewidth=0.6),
           meanprops=dict(marker="D", markerfacecolor="w", markeredgecolor="k",
                          markersize=2.5, markeredgewidth=0.6))


def group_offsets(k, slot=0.86):
    """Centered x-offsets and box width for k boxes sharing one line's slot."""
    w = slot / k
    return (np.arange(k) - (k - 1) / 2) * w, w * 0.9


def box_figure(case, groups, names, dist, coverage, yscale="linear"):
    """Draw the per-line box figure from groups=[(label, color, stats), ...] and return it."""
    n = len(names)
    x = np.arange(n)
    offsets, width = group_offsets(len(groups))
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    for (_, color, stats), off in zip(groups, offsets):
        draw_boxes(ax, stats, x + off, width, color)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
    ax.set_xlim(-0.7, n - 0.3)
    if yscale == "symlog":
        ax.set_yscale("symlog", linthresh=10)      # reveal the small-mu lines the top ones squash
    ax.set_ylim(bottom=0)
    ax.set_ylabel(r"$|\mu_l|$")
    ax.set_xlabel("line (ranked by " + r"$E[|\mu_l|]$)")
    ax.set_title(f"{case}: per-line $|\\mu_l|$ distribution  |  {DIST_LABEL[dist]}", fontsize=8)
    legend = [Patch(facecolor=color, alpha=0.6, edgecolor="k", label=label)
              for label, color, _ in groups]
    ax.legend(handles=legend, fontsize=6, frameon=False, ncol=2,
              title=f"box: 25-75, whiskers: {coverage * 100:g}%", title_fontsize=6)
    fig.tight_layout()
    return fig


def run(case, dist, budgets=None, rep=0, n_lines=15, coverage=0.95, yscale="linear",
        target="max_mu", snapshot="multi"):
    """Build and save the exact-vs-sampled per-line box figure over one case, law, budgets."""
    c, samples, meta = direct_samples(case)
    budgets = budgets or sorted(int(b) for b in samples.budget.unique())
    problem = create_problem(c, dist, target)
    mu, line_names = problem.mu_table()

    shown_idx, shown_names = select_lines(mu, line_names, problem.w, n=n_lines)
    colors = budget_colors(budgets)
    groups = [("exact truth", EXACT_COLOR,
               exact_stats(mu, problem.w, shown_idx, line_names, coverage))]
    for b in budgets:
        ids, weights = one_run_draws(samples, dist, b, rep, target, snapshot)
        groups.append((f"direct b{b}", colors[b],
                       sampled_stats(mu, ids, weights, shown_idx, line_names, coverage)))

    fig = box_figure(c.name, groups, shown_names, dist, coverage, yscale)
    btag = "-".join(str(b) for b in budgets)
    tag = f"_{yscale}" if yscale != "linear" else ""
    out = c.results_dir / f"exact_vs_sampled_box_{dist}_b{btag}{tag}_{case_tag(c.name)}.pdf"
    fig.savefig(out)
    print(f"saved -> {out}")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="activsg2000_v12_lev3a3")
    parser.add_argument("--dist", default="bernoulli", choices=DIST_NAMES)
    parser.add_argument("--budgets", default=None)
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--n-lines", type=int, default=15)
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--yscale", default="linear", choices=["linear", "symlog"])
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else None
    run(args.case, args.dist, budgets, args.rep, args.n_lines, args.coverage, args.yscale)
    plt.show()


if __name__ == "__main__":
    main()