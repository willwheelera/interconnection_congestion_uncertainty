"""Per-line jittered strip of the tail probability P[|mu_l| > c] across the matrix.

Facet-by-line companion to plot_line_exceed_violin.py (single line, violins): here every
line gets its own panel -- laid out like the plot_per_line_survivor grid, congested
lines first -- and each panel plots the raw per-replicate exceedance estimates as jittered
points.  The categorical x axis is the sampling budget; hue is the sampler-and-target
series.  Direct sampling ignores the importance target, so it is one series; each
importance sampler splits into one series per target (e.g. ascent-IS max, ascent-IS sum).
One panel then compares every series' spread across budgets at once, against that line's
exact nominal-weighted truth (dashed line).  One figure per config distribution.

Usage:
    uv run python -m src.figures.mu_distributions.line_exceed_strip --case case14_gen_per_bus
    uv run python -m src.figures.mu_distributions.line_exceed_strip --case case14_gen_per_bus \
        --threshold 100 --targets max_mu,sum_mu --budgets 640,2560
    uv run python -m src.figures.mu_distributions.line_exceed_strip --case activsg2000_original_v12 \
        --methods direct,ascent_is
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pyarrow.dataset as pads

from src.cases.registry import get_case
from src.sampling.problem import load_experiment_rows, experiment_meta, case_tag, create_problem
from src.sampling.distributions import DIST_LABEL
from src.figures.style import (DEFAULT_METHODS, method_color, method_label, select_methods,
                           series_tag, add_method_args, method_selection)
from src.figures.mu_distributions._line_stats import exceedance_prob, select_lines
from src.figures.mu_distributions._panel_layout import (FONT, LABEL_IN, LEGEND_IN, TITLE_IN, margin_grid,
                                           one_sigfig_yaxis, panel_title)

POINT_SIZE = 1.5                  # marker diameter in points
DODGE_W, JITTER = 0.8, 0.3        # category span split between series; jitter within a slot
JITTER_SEED = 0                   # fixed, so re-running moves no point
TARGET_COLOR = {"max_mu": "tab:green", "sum_mu": "tab:purple", "one_mu": "tab:cyan"}


def build_series(method_order, targets):
    """(series keys in plotting order, {key: color}, {key: legend label}).

    Direct sampling is target-independent -- one series, direct's own color.  Every other
    (importance) sampler splits into one series per target, colored by target.  The default
    two-method selection over max_mu/sum_mu is the intended three-series case: direct,
    ascent-IS max, ascent-IS sum.
    """
    order, palette, labels = [], {}, {}
    for m in method_order:
        if m == "direct":
            order.append("direct")
            palette["direct"] = method_color("direct")
            labels["direct"] = method_label("direct")
        else:
            for t in targets:
                key = f"{m}|{t}"
                order.append(key)
                palette[key] = TARGET_COLOR[t]
                labels[key] = f"{method_label(m)} ({t.replace('_mu', '')})"
    return order, palette, labels


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
    long["xcat"] = long["budget"].astype(str)
    long["series"] = np.where(long["method"] == "direct", "direct",
                              long["method"] + "|" + long["target"].astype(str))
    return long


def dodged_x(index, k, n_series, rng):
    """Jittered x positions for one series' points: category index, dodge slot, jitter.

    The n_series series split the DODGE_W span around each category index into equal
    slots; a point sits in its series' slot, jittered by a fraction of the slot width.
    Drawing a series' whole column in one scatter is what makes this cheap -- a
    categorical strip plot issues one call per (category, series) pair instead."""
    slot = DODGE_W / n_series
    jitter = JITTER / n_series
    return index + slot * (k - (n_series - 1) / 2) + rng.uniform(-jitter, jitter, index.size)


def strip_panel(ax, sub, xorder, series_order, palette, truth_p, line_name):
    """One line's panel: jittered per-rep points, dodged by series, plus the truth line."""
    rng = np.random.default_rng(JITTER_SEED)
    slot_of = {cat: i for i, cat in enumerate(xorder)}
    for k, s in enumerate(series_order):
        pts = sub[sub["series"] == s]
        x = dodged_x(pts["xcat"].map(slot_of).to_numpy(), k, len(series_order), rng)
        ax.scatter(x, pts["prob"].to_numpy(), s=POINT_SIZE ** 2, color=palette[s],
                   alpha=0.55, linewidth=0, zorder=3, rasterized=True)
    n = len(xorder)
    ax.axhline(truth_p, color="k", ls="--", lw=0.7)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels(xorder)
    panel_title(ax, f"line {line_name}")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=FONT["tick"], pad=1.5)
    ax.tick_params(axis="x", length=0)
    ax.margins(y=0.08)
    one_sigfig_yaxis(ax)


def figure_for_dist(frame, dist, order_names, truth_by_name, xorder, series_order, palette,
                    series_labels, title):
    """A grid of per-line strip panels (congested lines first) for one distribution."""
    df = frame[frame["dist"] == dist]
    ncol = 5
    n_nonzero = np.sum(np.array(list(truth_by_name.values())) > 1e-4)
    nrow = min(2, int(np.ceil(n_nonzero / ncol)))
    fig, axs = margin_grid(nrow, ncol)
    for name, ax in zip(order_names, axs.ravel()):
        strip_panel(ax, df[df["line"] == name], xorder, series_order, palette,
                    truth_by_name[name], name)
    for ax in axs.ravel()[len(order_names):]:
        ax.set_visible(False)

    handles = [Line2D([0], [0], marker="o", ls="", ms=2.5, color=palette[s],
                      label=series_labels[s]) for s in series_order]
    handles.append(Line2D([0], [0], color="k", ls="--", label="exact truth"))
    w, h = fig.get_figwidth(), fig.get_figheight()
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - LEGEND_IN / h),
               ncol=len(handles), frameon=True, fontsize=FONT["legend"], handletextpad=0.3,
               columnspacing=0.8, borderpad=0.3, handlelength=1.4)
    fig.suptitle(title, y=1 - TITLE_IN / h, fontsize=FONT["suptitle"])
    fig.supxlabel("sampling budget", y=LABEL_IN / h, fontsize=FONT["axlabel"])
    fig.supylabel(r"$\hat{P}[\,|\mu_l| > c\,]$", x=LABEL_IN / w, fontsize=FONT["axlabel"])
    return fig


def run(case, dists, targets=None, budgets=None, threshold=100.0,
        methods=DEFAULT_METHODS):
    """One per-line strip figure per nominal law; returns them in the given order.

    The sample frame and the |mu| table are law-independent and are built once; the exact
    truth each panel is read against, and the line order it induces, are per law."""
    c = get_case(case)
    case, case_dir = c.name, c.results_dir
    meta = experiment_meta(case_dir, "samples")
    budgets = budgets or meta["budgets"]
    samples = load_cells(case_dir, targets, budgets)
    problems = {d: create_problem(c, d) for d in dists}
    mu, line_names = next(iter(problems.values())).mu_table()

    xorder = [str(b) for b in budgets]
    frame = exceedance_frame(samples, mu, line_names, threshold)
    method_order = select_methods(frame["method"].unique(), methods)
    # direct ignores the target, so its rows are duplicated across targets -- keep one copy
    frame = frame[~((frame["method"] == "direct") & (frame["target"] != targets[0]))]
    series_order, palette, series_labels = build_series(method_order, targets)

    figs = []
    for dist in dists:
        truth_ps = exceedance_prob(mu, problems[dist].w, threshold)
        _, order_names = select_lines(mu, line_names, problems[dist].w)
        truth_by_name = {n: float(t) for n, t in zip(line_names, truth_ps)}
        title = (f"{case.split('_')[0]}: "
                 rf"$P[\,|\mu_l| > {threshold:g}\,]$  |  "
                 f"{DIST_LABEL[dist]}, {meta['n_rep']} reps")
        fig = figure_for_dist(frame, dist, order_names, truth_by_name, xorder,
                              series_order, palette, series_labels, title)
        out = (case_dir / f"line_exceed_strip_thr{int(threshold)}_{dist}"
                          f"{series_tag(methods)}_{case_tag(case)}.pdf")
        fig.savefig(out)
        print(f"saved -> {out}")
        figs.append(fig)
    return figs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="activsg2000_v12_lev3a3")
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
    for i in range(3,4):
        case = f"activsg2000_v12_lev{i}a3"
        dists = ["bernoulli", "mrf_a", "mrf_cobuild", "mrf_substitute"]
        targets = ["sum_mu"]
        run(case, dists, targets, threshold=500)
    plt.show()
