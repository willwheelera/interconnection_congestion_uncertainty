"""Per-line jittered strip of the tail probability P[|mu_l| > c] across the matrix.

Facet-by-line companion to plot_line_exceed_violin.py (single line, violins): here every
line gets its own panel -- laid out like the plot_per_line_survivor grid, congested
lines first -- and each panel plots the raw per-replicate exceedance estimates as jittered
points.  Hue = sampler method (each ascent-IS defensive weight beta is its own hue); the
categorical x axis encodes the (target, budget) cell, so one panel compares every selected
method's spread across importance functions and budgets at once, against that line's exact
nominal-weighted truth (dashed line).  One figure per config distribution.

Usage:
    uv run python -m src.figures.tail.line_exceed_strip --case case14_gen_per_bus
    uv run python -m src.figures.tail.line_exceed_strip --case case14_gen_per_bus \
        --threshold 100 --targets max_mu,sum_mu --budgets 640,2560
    uv run python -m src.figures.tail.line_exceed_strip --case activsg2000_original_v12 \
        --methods direct,ascent_is
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
import pyarrow.dataset as pads

from src.sampling.problem import (TargetProblem, REPO_ROOT, load_experiment_rows,
                              experiment_meta, CASES, case_tag)
from src.sampling.distributions import case_dist, DIST_LABEL
from src.figures.style import (DEFAULT_METHODS, method_color, method_label, select_methods,
                           series_tag, add_method_args, method_selection)

FIG_W = 6.0                       # total figure width, inches (two-column document limit)
PANEL_ASPECT = 2.5 / 2.2          # panel width / height
MARGIN_IN = {"left": 0.42, "right": 0.06, "bottom": 0.32, "top": 0.34}   # decoration bands
TITLE_IN, LEGEND_IN = 0.04, 0.19  # inches from the top edge down to the title / legend top
LABEL_IN = 0.03                   # inches from the bottom/left edge to the shared axis label
FONT = {"tick": 4.5, "panel_title": 5, "legend": 4.5, "suptitle": 7, "axlabel": 6}

POINT_SIZE = 1.5                  # marker diameter in points
DODGE_W, JITTER = 0.8, 0.3        # category span split between methods; jitter within a slot
JITTER_SEED = 0                   # fixed, so re-running moves no point


def truth_probabilities(mu, truth_w, threshold):
    """Exact nominal-weighted P[|mu_l| > threshold] for every line (full enumeration)."""
    return (truth_w[:, None] * (mu > threshold)).sum(0)


def load_cells(case_dir, targets, budgets):
    """The multi-snapshot sample rows this figure draws, filtered during the parquet scan.

    Only a fraction of a sweep's rows reach a panel, and the discarded ones dominate the
    read, so the cell selection is a pyarrow filter rather than a mask on a loaded frame;
    ascent is restricted to the all-neighbors move."""
    row_filter = ((pads.field("snapshot") == "multi") & pads.field("target").isin(targets)
                  & pads.field("budget").isin(budgets)
                  & pads.field("ascent_move").isin(["na", "all"]))
    return load_experiment_rows(case_dir, "samples", row_filter)


def exceedance_frame(sl, mu, line_names, threshold):
    """Long-form per-rep IS estimate of P[|mu_l| > threshold], one row per (cell, line).

    The estimate is the unbiased sum of IS weights on the exceeding configs (as in
    run_experiments.score_cell).  The threshold is applied once to the per-id mu table
    rather than to the gathered per-sample block, and the per-group weighted sums are
    reduced one line at a time with bincount over integer codes, so nothing the size of
    (samples x lines) is ever built; xcat is the (target, budget) axis.
    """
    keys = ["dist", "target", "budget", "method", "ascent_move", "rep"]
    codes = sl.groupby(keys, sort=False, observed=True).ngroup().to_numpy()
    hit = mu > threshold
    ids, weight = sl["id"].to_numpy(), sl["weight"].to_numpy()
    est = np.stack([np.bincount(codes, weight * hit[ids, l], minlength=codes.max() + 1)
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


def dodged_x(index, k, n_method, rng):
    """Jittered x positions for one method's points: category index, dodge slot, jitter.

    The n_method methods split the DODGE_W span around each category index into equal
    slots; a point sits in its method's slot, jittered by a fraction of the slot width.
    Drawing a method's whole column in one scatter is what makes this cheap -- a
    categorical strip plot issues one call per (category, method) pair instead."""
    slot = DODGE_W / n_method
    jitter = JITTER / n_method
    return index + slot * (k - (n_method - 1) / 2) + rng.uniform(-jitter, jitter, index.size)


def strip_panel(ax, sub, xorder, method_order, palette, truth_p, line_name):
    """One line's panel: jittered per-rep points, dodged by method, plus the truth line."""
    rng = np.random.default_rng(JITTER_SEED)
    slot_of = {cat: i for i, cat in enumerate(xorder)}
    for k, method in enumerate(method_order):
        pts = sub[sub["method"] == method]
        x = dodged_x(pts["xcat"].map(slot_of).to_numpy(), k, len(method_order), rng)
        ax.scatter(x, pts["prob"].to_numpy(), s=POINT_SIZE ** 2, color=palette[method],
                   alpha=0.55, linewidth=0, zorder=3, rasterized=True)
    n = len(xorder)
    ax.axhline(truth_p, color="k", ls="--", lw=0.7)
    ax.axvline(x=n/2-0.5, color="gray", ls="-", lw=0.2)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xticks(range(n))
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

    handles = [Line2D([0], [0], marker="o", ls="", ms=2.5, color=palette[m], label=method_label(m))
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


def run(case, dists, targets=None, budgets=None, threshold=100.0,
        methods=DEFAULT_METHODS):
    """One per-line strip figure per nominal law; returns them in the given order.

    The sample frame and the |mu| table are law-independent and are built once; the exact
    truth each panel is read against, and the line order it induces, are per law."""
    case_dir = REPO_ROOT / "results" / case
    meta = experiment_meta(case_dir, "samples")
    budgets = budgets or meta["budgets"]
    samples = load_cells(case_dir, targets, budgets)
    problems = {d: TargetProblem(case_dir / "target_cache.parquet",
                                 dist=case_dist(CASES[case], d)) for d in dists}
    mu, line_names = next(iter(problems.values())).mu_table()

    xorder = [f"{t}\n{b}" for t in targets for b in budgets]
    frame = exceedance_frame(samples, mu, line_names, threshold)
    method_order = select_methods(frame["method"].unique(), methods)
    palette = {m: method_color(m) for m in method_order}

    figs = []
    for dist in dists:
        truth_ps = truth_probabilities(mu, problems[dist].w, threshold)
        order_names = [line_names[i] for i in np.argsort(-truth_ps)]
        truth_by_name = {n: float(t) for n, t in zip(line_names, truth_ps)}
        title = (f"{case.split('_')[0]}: "
                 rf"$P[\,|\mu_l| > {threshold:g}\,]$  |  "
                 f"{DIST_LABEL[dist]}, {meta['n_rep']} reps")
        fig = figure_for_dist(frame, dist, order_names, truth_by_name, xorder,
                              method_order, palette, title)
        out = (case_dir / f"line_exceed_strip_thr{int(threshold)}_{dist}"
                          f"{series_tag(methods)}_{case_tag(case)}.pdf")
        fig.savefig(out)
        print(f"saved -> {out}")
        figs.append(fig)
    return figs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="activsg2000_original_v12")
    parser.add_argument("--dists", default="bernoulli")
    parser.add_argument("--targets", default="max_mu,sum_mu")
    parser.add_argument("--budgets", default=None)
    parser.add_argument("--threshold", type=float, default=100.0)
    add_method_args(parser)
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else None
    run(args.case, args.dists.split(","), args.targets.split(","), budgets, args.threshold,
        method_selection(args))
    plt.show()


if __name__ == "__main__":
    main()
