"""Per-line jittered strip of the tail probability P[|mu_l| > c] across the matrix.

Facet-by-line companion to plot_line_exceed_violin.py (single line, violins): here every
line gets its own panel -- laid out like the plot_per_line_survivor grid, congested
lines first -- and each panel plots the raw per-replicate exceedance estimates as jittered
points.  Hue = sampler method; the categorical x axis encodes the (target, budget) cell, so
one panel compares every method's spread across importance functions and budgets at once,
against that line's exact nominal-weighted truth (dashed line).  One figure per config
distribution.

Usage:
    uv run python -m tail_sampling.plot_line_exceed_strip --case case14_gen_per_bus
    uv run python -m tail_sampling.plot_line_exceed_strip --case case14_gen_per_bus \
        --threshold 100 --targets max_mu,sum_mu --budgets 640,2560
"""

import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
import seaborn as sns
import pyarrow.parquet as pq

from tail_sampling.problem_setup import TargetProblem, REPO_ROOT, load_experiment, CASES, case_tag
from tail_sampling.distributions import make_dist, case_p
from tail_sampling.plot_inverse_cdf import COLORS, LABELS
from tail_sampling.plot_per_line_survivor import line_order

DIST_LABEL = {"bernoulli": "Bernoulli", "mrf_a": "MRF-A", "mrf_b": "MRF-B", "mrf_c": "MRF-C"}
METHOD_ORDER = ["direct", "subset_sim", "smc_tilt", "cross_entropy", "ascent_is", "ascent_exact"]

FIG_W = 6.0                       # total figure width, inches (two-column document limit)
PANEL_ASPECT = 2.5 / 2.2          # panel width / height
MARGIN_IN = {"left": 0.42, "right": 0.06, "bottom": 0.32, "top": 0.34}   # decoration bands
TITLE_IN, LEGEND_IN = 0.04, 0.19  # inches from the top edge down to the title / legend top
LABEL_IN = 0.03                   # inches from the bottom/left edge to the shared axis label
FONT = {"tick": 4.5, "panel_title": 5, "legend": 4.5, "suptitle": 7, "axlabel": 6}


def truth_probabilities(mu, truth_w, threshold):
    """Exact nominal-weighted P[|mu_l| > threshold] for every line (full enumeration)."""
    return (truth_w[:, None] * (mu > threshold)).sum(0)


def exceedance_frame(samples, mu, line_names, targets, budgets, threshold):
    """Long-form per-rep IS estimate of P[|mu_l| > threshold], one row per (cell, line).

    The estimate is the unbiased sum of IS weights on the exceeding configs (as in
    run_experiments.score_cell); ascent is restricted to the all-neighbors move.  The
    per-group weighted sums are reduced with bincount over integer codes to avoid
    materializing a multi-million-row wide table; xcat is the (target, budget) axis.
    """
    sl = samples[(samples["snapshot"] == "multi") & samples["target"].isin(targets)
                 & samples["budget"].isin(budgets) & samples["ascent_move"].isin(["na", "all"])]
    keys = ["dist", "target", "budget", "method", "ascent_move", "rep"]
    codes = sl.groupby(keys, sort=False, observed=True).ngroup().to_numpy()
    exc = (sl["weight"].to_numpy()[:, None] * (mu[sl["id"].to_numpy()] > threshold)).astype(np.float32)
    est = np.stack([np.bincount(codes, exc[:, l], minlength=codes.max() + 1)
                    for l in range(len(line_names))], axis=1)
    agg = sl[keys].iloc[np.unique(codes, return_index=True)[1]].reset_index(drop=True)
    agg[line_names] = est
    long = agg.melt(id_vars=keys, value_vars=line_names, var_name="line", value_name="prob")
    long["xcat"] = long["target"].astype(str) + "\n" + long["budget"].astype(str)
    return long


def fmt_one_sigfig(v, _pos):
    """One-sigfig tick label: scientific below 1e-3 (e.g. 1e-4), plain decimal at/above."""
    if v == 0:
        return "0"
    if abs(v) < 9.5e-3:
        mantissa, exponent = f"{v:.0e}".split("e")
        return f"{mantissa}e{int(exponent)}"
    return f"{v:g}"


def one_sigfig_yaxis(ax, nbins=4):
    """Restrict y ticks to one-significant-figure values (leaves the y limits unchanged)."""
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, steps=[1, 2, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_one_sigfig))


def xtick_labels(xorder):
    """Tick labels for the (target, budget) axis: budget on every tick, target once per group.

    At 6-inch width a panel is too narrow for the target name under all four ticks, so it is
    printed only on its group's first tick and overhangs into the blank slots beside it."""
    labels, seen = [], set()
    for cat in xorder:
        target, budget = cat.split("\n")
        labels.append(f"{target}\n{budget}" if target not in seen else f"\n{budget}")
        seen.add(target)
    return labels


def strip_panel(ax, sub, xorder, method_order, palette, truth_p, line_name):
    """One line's panel: jittered per-rep points, dodged by method, plus the truth line."""
    sns.stripplot(data=sub, x="xcat", y="prob", order=xorder, hue="method",
                  hue_order=method_order, palette=palette, dodge=True, jitter=0.3,
                  size=1.5, alpha=0.55, linewidth=0, ax=ax, legend=False)
    for coll in ax.collections:
        coll.set_rasterized(True)
    n = sub["xcat"].nunique()
    ax.axhline(truth_p, color="k", ls="--", lw=0.7)
    ax.axvline(x=n/2-0.5, color="gray", ls="-", lw=0.2)
    ax.set_xticks(range(len(xorder)))
    ax.set_xticklabels(xtick_labels(xorder))
    ax.set_title(f"line {line_name}", fontsize=FONT["panel_title"], y=0.84)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=FONT["tick"], pad=1.5)
    ax.tick_params(axis="x", length=0)
    ax.margins(y=0.08)
    one_sigfig_yaxis(ax)


def margin_grid(nrow, ncol):
    """Panel grid of fixed FIG_W width whose decoration margins are inches, not fractions.

    Panel width follows from the width left over after the margins; panel height follows
    from PANEL_ASPECT, so the figure grows downward with the row count while the bands
    reserved for the suptitle, legend, and shared axis labels keep the same absolute size."""
    w = FIG_W
    panel_w = (w - MARGIN_IN["left"] - MARGIN_IN["right"]) / ncol
    h = nrow * panel_w / PANEL_ASPECT + MARGIN_IN["top"] + MARGIN_IN["bottom"]
    fig, axs = plt.subplots(nrow, ncol, figsize=(w, h), squeeze=False, sharex=True)
    fig.subplots_adjust(left=MARGIN_IN["left"] / w, right=1 - MARGIN_IN["right"] / w,
                        bottom=MARGIN_IN["bottom"] / h, top=1 - MARGIN_IN["top"] / h,
                        wspace=0.24, hspace=0.12)
    return fig, axs


def figure_for_dist(frame, dist, order_names, truth_by_name, xorder, method_order, palette, title):
    """A grid of per-line strip panels (congested lines first) for one distribution."""
    df = frame[frame["dist"] == dist]
    ncol = 5
    n_nonzero = np.sum(np.array(list(truth_by_name.values())) > 1e-4)
    nrow = min(4, int(np.ceil(n_nonzero / ncol)))
    fig, axs = margin_grid(nrow, ncol)
    for name, ax in zip(order_names, axs.ravel()):
        strip_panel(ax, df[df["line"] == name], xorder, method_order, palette,
                    truth_by_name[name], name)
    for ax in axs.ravel()[len(order_names):]:
        ax.set_visible(False)

    handles = [Line2D([0], [0], marker="o", ls="", ms=2.5, color=palette[m], label=LABELS.get(m, m))
               for m in method_order]
    handles.append(Line2D([0], [0], color="k", ls="--", label="exact truth"))
    w, h = fig.get_figwidth(), fig.get_figheight()
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - LEGEND_IN / h),
               ncol=len(handles), frameon=True, fontsize=FONT["legend"], handletextpad=0.3,
               columnspacing=0.8, borderpad=0.3, handlelength=1.4)
    fig.suptitle(title, y=1 - TITLE_IN / h, fontsize=FONT["suptitle"])
    fig.supxlabel("importance target / sampling budget", y=LABEL_IN / h,
                  fontsize=FONT["axlabel"])
    fig.supylabel(r"$\hat{P}[\,|\mu_l| > c\,]$", x=LABEL_IN / w, fontsize=FONT["axlabel"])
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="case14_gen_per_bus")
    parser.add_argument("--dists", default="bernoulli,mrf_a,mrf_b,mrf_c")
    parser.add_argument("--targets", default="max_mu,sum_mu")
    parser.add_argument("--budgets", default=None)
    parser.add_argument("--threshold", type=float, default=100.0)
    args = parser.parse_args()

    case_dir = REPO_ROOT / "results" / args.case
    samples, meta = load_experiment(case_dir, "samples")
    tc = case_dir / "target_cache.parquet"
    ngen = int(json.loads(pq.read_schema(tc).metadata[b"run_meta"])["ngen"])
    p = case_p(CASES[args.case])
    problem = TargetProblem(tc, dist=make_dist("bernoulli", ngen, p))
    mu, line_names = problem.mu_table()

    targets = args.targets.split(",")
    budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else meta["budgets"]
    xorder = [f"{t}\n{b}" for t in targets for b in budgets]

    truth_ps = truth_probabilities(mu, problem.w, args.threshold)
    order_names = [line_names[i] for i in np.argsort(-truth_ps)]
    #_, order_names = line_order(mu, line_names, problem.w)
    truth_by_name = {line_names[i]: float(truth_ps[i]) for i in range(len(line_names))}

    frame = exceedance_frame(samples, mu, line_names, targets, budgets, args.threshold)
    method_order = [m for m in METHOD_ORDER if m in frame["method"].unique()]
    palette = {m: COLORS.get(m, "gray") for m in method_order}

    for dist in args.dists.split(","):
        title = (f"{args.case.split('_')[0]}: "
                 rf"$P[\,|\mu_l| > {args.threshold:g}\,]$  |  "
                 f"{DIST_LABEL[dist]}, {meta['n_rep']} reps")
        fig = figure_for_dist(frame, dist, order_names, truth_by_name, xorder,
                              method_order, palette, title)
        out = case_dir / f"line_exceed_strip_thr{int(args.threshold)}_{dist}_{case_tag(args.case)}.pdf"
        fig.savefig(out)
        print(f"saved -> {out}")
    plt.show()


if __name__ == "__main__":
    main()
