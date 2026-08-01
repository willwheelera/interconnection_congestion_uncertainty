"""Descriptor-study figures: rank ordering and raw magnitude, one row per case.

Two grids over the same computed rows (`descriptors.study.compute_all`), from one code path
via `case_row(..., rank=True/False)`:

  rank    rank-rank scatter per descriptor annotated with Spearman and the linear R^2, plus a
          combined rank fit;
  values  raw descriptor magnitude vs the target with no fitted line -- the cloud itself is the
          evidence -- plus the combined raw linear fit as predicted-vs-actual.

Writes results/mu_static_descriptors_<target>_{rank,values}.png.

Usage:
    uv run python -m figures.descriptors                # mean, the default
    uv run python -m figures.descriptors quantile 0.95
    uv run python -m figures.descriptors exceed 200
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

from descriptors.study import (COLS, STUDY_CASES, compute_all, ols_r2, parse_target,
                               r2_no_top)
from paths import RESULTS

FIG_WIDTH, ROW_HEIGHT = 6.0, 1.7   # inches; 6 in wide fits a document text block
# type sizes to match: a 4-panel row is only ~1.5 in wide, so everything shrinks with it
STYLE = {"font.size": 4.5, "axes.labelsize": 5, "axes.titlesize": 5.5, "figure.titlesize": 7,
         "legend.fontsize": 4.5, "xtick.labelsize": 4, "ytick.labelsize": 4,
         "axes.linewidth": 0.5, "xtick.major.width": 0.4, "ytick.major.width": 0.4,
         "xtick.major.size": 2, "ytick.major.size": 2, "savefig.dpi": 300}


def target_axis(lab, rank):
    return f"rank of {lab}" if rank else lab


def panel(ax, x, y, xlabel, ylabel, text, ha="left"):
    """Scatter with axis labels and a corner annotation box."""
    ax.scatter(x, y, s=4, c="#3a6", alpha=0.85, zorder=2)
    lo, hi = float(np.min(y)), float(np.max(y))
    ax.set_ylim(lo - 0.06 * (hi - lo), hi + 0.34 * (hi - lo))   # headroom under the text box
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.text(0.03 if ha == "left" else 0.96, 0.97, text, transform=ax.transAxes, va="top", ha=ha,
            bbox=dict(boxstyle="round", pad=0.25, fc="w", ec="0.6", lw=0.4))


def mark(ax, x, y, binding, top, rank):
    """Rank basis marks the lines binding at the base solve; raw basis circles the leverage line."""
    if rank:
        ax.scatter(x[binding], y[binding], s=5, c="#d33", edgecolor="k", lw=0.3, zorder=3)
    else:
        ax.scatter(x[top], y[top], s=34, facecolors="none", edgecolors="k", lw=0.7, zorder=4)


def case_row(axes, d, y, binding, top, X, lab, rank=False):
    """One case: a panel per descriptor then the combined fit, in ranks (rank=True) or raw values."""
    tx = rankdata if rank else (lambda v: v)
    ty, ylabel = tx(y), target_axis(lab, rank)
    for ax, (name, xlabel, symlogx) in zip(axes, COLS):
        feat = d[name].values
        panel(ax, tx(feat), ty, f"rank of {name}" if rank else xlabel, ylabel,
              descriptor_text(feat, y, top, rank), ha="left" if rank else "right")
        mark(ax, tx(feat), ty, binding, top, rank)
        if symlogx and not rank:
            ax.set_xscale("symlog", linthresh=1.0)
            ax.xaxis.get_major_locator().set_params(numticks=4)   # decades do not fit at 1.5 in
    combined_fit(axes[3], X, y, top, lab, rank, tx)


def descriptor_text(feat, y, top, rank):
    """Annotation for one descriptor panel; both R^2 are on the raw values, not the ranks."""
    r2 = ols_r2(feat[:, None], y)[0]
    if rank:
        return f"Spearman = {spearmanr(feat, y).correlation:+.2f}\nlinear R2 = {r2:.2f}"
    return f"R2 = {r2:.2f}\nno circled line: {r2_no_top(feat[:, None], y, top):.2f}"


def combined_fit(ax, X, y, top, lab, rank, tx):
    """Fourth panel: slack+sigma+z fit in the row's basis, plotted as predicted vs actual."""
    Xt = np.column_stack([tx(col) for col in X.T])
    ty = tx(y)
    r2, pred = ols_r2(Xt, ty)
    text = (f"combined rank-R2 = {r2:.2f}\nslack alone = {ols_r2(Xt[:, :1], ty)[0]:.2f}" if rank
            else f"R2 = {r2:.2f}\nno circled line: {r2_no_top(Xt, ty, top):.2f}")
    xlabel = (f"rank of predicted {lab}" if rank else f"predicted {lab} (slack+sigma+z)")
    panel(ax, pred, ty, xlabel, target_axis(lab, rank), text)
    ax.set_title(f"combined fit ({'rank' if rank else 'raw'})")
    if not rank:
        mark(ax, pred, ty, None, top, rank)
        lims = [min(pred.min(), ty.min()), max(pred.max(), ty.max())]
        ax.plot(lims, lims, ls="--", c="#333", lw=0.5)


def legend_handle(lab, rank):
    """Marker key: binding lines (rank figure) or the circled leverage line (values figure)."""
    kwargs = dict(marker="o", ls="", mec="k",)
    if rank:
        kwargs["label"] = "binding at base solve"
        kwargs["c"] = "#d33"
    else:
        kwargs["label"] = f"highest-{lab} line (circled point, dominates the fit)"
        kwargs["mfc"] = "none"
    return plt.Line2D([], [], **kwargs)


def make_figure(data, lab, rank=False):
    """Grid of one row per case, in ranks (rank=True) or raw values; FIG_WIDTH inches wide."""
    subtitle = "rank ordering" if rank else "raw magnitude and combined linear fit"
    height = ROW_HEIGHT * len(STUDY_CASES) + 0.5
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(STUDY_CASES), 4, figsize=(FIG_WIDTH, height))
        for c, (case, regime) in enumerate(STUDY_CASES):
            case_row(axes[c], *data[case], lab, rank)
            axes[c, 0].set_ylabel(f"{case}\n({regime})\n{target_axis(lab, rank)}")
        fig.legend(handles=[legend_handle(lab, rank)], loc="lower center", frameon=False)
        fig.suptitle(f"Cheap static descriptors vs the sampled {lab}: {subtitle}")
        # strips reserved in inches, not fractions: they must not grow with the case count
        fig.tight_layout(rect=(0, 0.16 / height, 1, 1 - 0.24 / height), pad=0.6, w_pad=0.8)
    return fig


def main():
    target = parse_target(sys.argv[1:])
    data = compute_all(target)
    for tag, rank in [("rank", True), ("values", False)]:
        fig = make_figure(data, target.label, rank)
        out = RESULTS / f"mu_static_descriptors_{target.tag}_{tag}.png"
        fig.savefig(out, dpi=300)
        print("wrote", out)
    plt.show()


if __name__ == "__main__":
    main()
