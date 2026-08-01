"""Sampler comparison on the single line that was used as the importance target.

The per-line strip figure (plot_line_exceed_strip.py) faceted over every line at once;
this one keeps only the `one_mu` line -- the one the target was built from -- and spends
the panels on the nominal law instead, so the four config distributions sit side by side
for that single line.  Each panel plots the raw per-replicate estimate of
P[|mu_l| > c] as jittered points, hue = sampler (direct included), against the exact
nominal-weighted truth (dashed).

The threshold c defaults to the mu level that line exceeds with probability --at, computed
exactly from the enumeration, so the same command works across cases whose mu scales
differ by an order of magnitude (case30 tops out near 340, Texas v12 near 2000).  It is
resolved separately under each nominal law -- so every panel sits at the same probability
on its own c, and c is reported in the panel title -- unless --threshold pins one value.

target_line_strip() takes every option as a plain python argument, so the figure can be
built from a session or another script without going through the command line:

    from tail_sampling.plot_target_line_strip import target_line_strip
    fig, specs, out = target_line_strip("case30_gen_per_bus", at=1e-3)
    fig, specs, _ = target_line_strip("activsg2000_original_v12", budgets=[2560], save=False)

main() only parses the equivalent command line and hands off to it:

    uv run python -m tail_sampling.plot_target_line_strip --case case14_gen_per_bus
    uv run python -m tail_sampling.plot_target_line_strip --case activsg2000_original_v12 \
        --at 1e-4 --targets one_mu,max_mu
"""

import argparse
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, SymmetricalLogLocator

from tail_sampling.problem_setup import (TargetProblem, REPO_ROOT, CASES, case_tag,
                                          experiment_dir, read_parquet_meta)
from tail_sampling.distributions import make_dist, case_p, DIST_NAMES
from tail_sampling.plot_inverse_cdf import COLORS, LABELS
from tail_sampling.plot_line_exceed_strip import DIST_LABEL, METHOD_ORDER, fmt_one_sigfig

KEYS = ["dist", "target", "budget", "method", "ascent_move", "rep"]
DEFAULT_CASE = "case14_gen_per_bus"
# single-column width: the 2x2 grid is meant to drop into a paper column or a slide at
# this size unscaled, so every font below is a point size at 3.4 in, not a relative unit
FIG_SIZE = (3.0, 3.4)


def load_samples(case_dir, targets):
    """Read only the requested targets out of the sample shards, plus the run metadata.

    Each shard carries its own budget in its metadata, so the combined budget list is the
    union across shards (as load_experiment does for the curves)."""
    d = experiment_dir(case_dir, "samples")
    frames, budgets, meta = [], [], None
    for shard in sorted(glob.glob(str(d / "b*.parquet"))):
        meta = read_parquet_meta(shard)[1]
        budgets.extend(meta["budgets"])
        frames.append(pq.read_table(shard, filters=[("target", "in", targets)]).to_pandas())
    return pd.concat(frames, ignore_index=True), {**meta, "budgets": sorted(set(budgets))}


def build_problem(case, case_dir):
    """TargetProblem on this case's one_mu target under its own nominal Bernoulli law."""
    tc = case_dir / "target_cache.parquet"
    ngen = int(read_parquet_meta(tc)[1]["ngen"])
    p = case_p(CASES[case])
    return TargetProblem(tc, target="one_mu", dist=make_dist("bernoulli", ngen, p))


def level_at(mu_l, w, prob):
    """The mu level this line exceeds with nominal probability prob (exact enumeration)."""
    order = np.argsort(-mu_l)
    surv = np.cumsum(w[order])
    ind = min(np.searchsorted(surv, prob), len(order) - 1)
    return int(float(mu_l[order][ind]))


def estimate_frame(samples, mu_l, thresholds, one_target):
    """Per-replicate IS estimate of P[|mu_l| > c], one row per matrix cell.

    The estimate is the sum of importance weights carried by the exceeding configs, the
    same reduction run_experiments.score_cell applies to the target itself; the ascent
    samplers are restricted to their all-neighbors move so each method appears once.
    `thresholds` maps each nominal law to its own c, applied row-wise.
    """
    sl = samples[(samples["snapshot"] == "multi") & samples["ascent_move"].isin(["na", "all"])]
    c = sl["dist"].map(thresholds).to_numpy()
    exc = sl["weight"].to_numpy() * (mu_l[sl["id"].to_numpy()] > c)
    out = sl[KEYS].assign(prob=exc).groupby(KEYS, observed=True, sort=False)["prob"].sum()
    out = out.reset_index()
    out["xcat"] = (out["budget"].astype(str) if one_target
                   else out["target"].astype(str) + "\n" + out["budget"].astype(str))
    return out


def thin_symlog_yaxis(ax, linthresh):
    """Few decade ticks, one sigfig each -- a 1.7 in panel has no room for the default set."""
    loc = SymmetricalLogLocator(base=10.0, linthresh=linthresh)
    loc.set_params(numticks=4)
    ax.yaxis.set_major_locator(loc)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_one_sigfig))


def dist_panel(ax, sub, xorder, method_order, palette, spec, dist):
    """One nominal law's panel: jittered per-rep points dodged by sampler, plus truth.

    The threshold is resolved per law, so it belongs in the panel title, not the figure's."""
    threshold, truth_p = spec
    sns.stripplot(data=sub, x="xcat", y="prob", order=xorder, hue="method",
                  hue_order=method_order, palette=palette, dodge=True, jitter=0.3,
                  size=1.8, alpha=0.6, linewidth=0, ax=ax, legend=False)
    for coll in ax.collections:
        coll.set_rasterized(True)
    ax.axhline(truth_p, color="k", ls="--", lw=0.9)
    ax.set_title(rf"{DIST_LABEL[dist]},  $c = {threshold:.4g}$", fontsize=6.5, pad=2.5)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=5.5, pad=1.5)
    ax.tick_params(axis="x", length=0)
    ax.margins(y=0.10)
    # symlog keeps the exact zeros (the usual direct-MC outcome deep in the tail) on the
    # axis while still resolving estimates that overshoot the truth by orders of magnitude
    ax.set_yscale("symlog", linthresh=truth_p / 5, linscale=0.4)
    thin_symlog_yaxis(ax, truth_p / 5)
    ax.set_ylim(bottom=0)#, top=8e-2)


def figure(frame, dists, specs, xorder, method_order, palette, title, xlabel):
    """2 x 2 grid of nominal-law panels for the single target line."""
    fig, axs = plt.subplots(2, 2, figsize=FIG_SIZE, squeeze=False, sharex=True, sharey=True)
    for dist, ax in zip(dists, axs.ravel()):
        dist_panel(ax, frame[frame["dist"] == dist], xorder, method_order, palette,
                   specs[dist], dist)
    for ax in axs.ravel()[len(dists):]:
        ax.set_visible(False)

    handles = [Line2D([0], [0], marker="o", ls="", ms=2.6, color=palette[m],
                      label=LABELS.get(m, m)) for m in method_order]
    handles.append(Line2D([0], [0], color="k", ls="--", lw=0.9, label="exact truth"))
    # legend under the grid in two rows: a single row of seven entries needs more than the
    # 3.4 in the whole figure is allowed, and the panels themselves have no spare room
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=4,
               frameon=False, fontsize=5.5, handletextpad=0.3, columnspacing=1.0,
               labelspacing=0.25)
    fig.suptitle(title, y=0.985, fontsize=7)
    fig.supxlabel(xlabel, fontsize=6.5, y=0.115)
    fig.supylabel(r"$\hat{P}[\,|\mu_l| > c\,]$", fontsize=6.5, x=0.015)
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.185, top=0.895,
                        wspace=0.10, hspace=0.30)
    return fig


def panel_specs(case, case_dir, dists, mu_l, at, fixed=None):
    """Per-nominal-law (threshold c, exact truth P[|mu_l| > c]) for the panels.

    Each law puts a different probability on the line, so c is distinct for each one: 
    every panel then sits at the same target probability `at` on its own threshold.  
    An explicit --threshold (fixed) pins c instead."""
    tc = case_dir / "target_cache.parquet"
    ngen = int(read_parquet_meta(tc)[1]["ngen"])
    p = case_p(CASES[case])
    out = {}
    for dist in dists:
        prob = TargetProblem(tc, target="one_mu", dist=make_dist(dist, ngen, p))
        c = fixed or level_at(mu_l, prob.w, at)
        out[dist] = (c, float(prob.w[mu_l > c].sum()))
    return out


def output_path(case_dir, case, threshold, at):
    """Destination pdf, keyed on the pinned c when there is one and on --at otherwise."""
    tag = f"c{threshold:.6g}" if threshold else f"at{at:g}"
    return case_dir / f"target_line_strip_{case_tag(case)}_{tag}.pdf"


def target_line_strip(case=DEFAULT_CASE, dists=DIST_NAMES, targets=("one_mu",),
                      budgets=None, threshold=None, at=1e-2, save=True):
    """Build (and by default save) the nominal-law strip figure for one case.

    The whole plot is reachable from python with these arguments -- no command line needed:

        fig, specs, out = target_line_strip("case30_gen_per_bus", at=1e-3)

    budgets defaults to every budget the case was run at; threshold pins c for all panels
    and otherwise each panel resolves its own c at probability `at`.  Returns the figure,
    the per-law {dist: (c, truth)} specs, and the output path (None when save is False).
    """
    case_dir = REPO_ROOT / "results" / case
    dists, targets = list(dists), list(targets)
    problem = build_problem(case, case_dir)
    mu_l = problem.targets["one_mu"]
    specs = panel_specs(case, case_dir, dists, mu_l, at, threshold)

    samples, meta = load_samples(case_dir, targets)
    budgets = [int(b) for b in budgets] if budgets else meta["budgets"]
    samples = samples[samples["budget"].isin(budgets)]
    one_target = len(targets) == 1
    frame = estimate_frame(samples, mu_l, {d: c for d, (c, _) in specs.items()}, one_target)

    xorder = ([str(b) for b in budgets] if one_target
              else [f"{t}\n{b}" for t in targets for b in budgets])
    method_order = [m for m in METHOD_ORDER if m in frame["method"].unique()]
    palette = {m: COLORS.get(m, "gray") for m in method_order}
    title = (f"{case_tag(case)}, line {problem.meta['one_mu_element']}, "
             f"{meta['n_rep']} reps per cell")
    xlabel = "sampling budget" if one_target else "importance target / sampling budget"

    fig = figure(frame, dists, specs, xorder, method_order, palette, title, xlabel)
    out = output_path(case_dir, case, threshold, at) if save else None
    if out:
        fig.savefig(out)
    return fig, specs, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--dists", default=",".join(DIST_NAMES))
    parser.add_argument("--targets", default="one_mu")
    parser.add_argument("--budgets", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--at", type=float, default=1e-2)
    args = parser.parse_args()

    _, specs, out = target_line_strip(
        case=args.case, dists=args.dists.split(","), targets=args.targets.split(","),
        budgets=args.budgets.split(",") if args.budgets else None,
        threshold=args.threshold, at=args.at)
    print(f"saved -> {out}  (" +
          ", ".join(f"{d} c={c:.6g} truth={t:.3e}" for d, (c, t) in specs.items()) + ")")
    plt.show()


if __name__ == "__main__":
    #main()
    for case in [
        "case14_gen_per_bus",
        "case30_gen_per_bus",
        "activsg2000_2025",
        "activsg2000_original_v12",
   ]: 
        _, specs, out = target_line_strip( case=case)
        specstrs = [f"{d} c={c:.6g} truth={t:.3e}" for d, (c, t) in specs.items()]
        print(f"saved -> {out}  ({'\n '.join(specstrs)})")
    plt.show()


