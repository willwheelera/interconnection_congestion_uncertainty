"""Additivity figures: is the target a sum of per-candidate contributions?

Two figures over the same computed rows (`descriptors.additivity.compute_all`):

  fit     one panel per case, the true target against the fitted value of the linear model
          in the queue bits, over every enumerated configuration.  The identity line is
          the perfectly additive target; the 1% tail level is drawn on both axes, so the
          part of the cloud above the horizontal line and left of the vertical one is the
          tail the linear model cannot reach.
  orders  the exact variance share of each interaction order, stacked per case.  The
          bottom block is the additive share, which equals the fit's R^2.

Writes results/mu_additivity_<target>_{fit,orders}.png.

Usage:
    uv run python -m src.figures.additivity
    uv run python -m src.figures.additivity --target one_mu
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from src.descriptors.additivity import ADDITIVITY_CASES, MAX_ORDER, TAIL_Q, compute_all
from src.paths import RESULTS

FIG_WIDTH = 6.0                    # inches; fits a document text block
NCOL = 4
STYLE = {"font.size": 5, "axes.labelsize": 5.5, "axes.titlesize": 6, "figure.titlesize": 7.5,
         "legend.fontsize": 5, "xtick.labelsize": 4.5, "ytick.labelsize": 4.5,
         "axes.linewidth": 0.5, "xtick.major.width": 0.4, "ytick.major.width": 0.4,
         "xtick.major.size": 2, "ytick.major.size": 2, "savefig.dpi": 300}
ORDER_COLORS = ["#2c7fb8", "#7fcdbb", "#c7e9b4", "#fed976", "#f03b20"]


def fit_panel(ax, row, label, target):
    """One case: true target vs linear fit as a log-density hexbin, with the tail level."""
    y, pred = row["y"], row["pred"]
    lo, hi = float(min(y.min(), pred.min())), float(max(y.max(), pred.max()))
    pad = 0.04 * (hi - lo)
    ax.plot([lo, hi], [lo, hi], "--", color="0.45", lw=0.4, zorder=1)
    hb = ax.hexbin(pred, y, gridsize=45, cmap="Blues", mincnt=1, linewidths=0, zorder=2,
                   extent=(lo - pad, hi + pad, lo - pad, hi + pad))
    # explicit norm rather than bins="log": a case whose cloud collapses to two equally
    # filled cells has no count range for the automatic log scale to normalize against
    hb.set_norm(LogNorm(vmin=1, vmax=max(hb.get_array().max(), 2)))
    tail = float(np.quantile(y, 1 - TAIL_Q))
    ax.axhline(tail, color="#d33", lw=0.5, zorder=3)
    ax.axvline(tail, color="#d33", lw=0.5, ls=":", zorder=3)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.text(0.96, 0.05, f"{row['n_distinct']} distinct values\ntop candidate {row['top_share']:.2f}",
            transform=ax.transAxes, va="bottom", ha="right", zorder=4,
            bbox=dict(boxstyle="round", pad=0.2, fc="w", ec="0.6", lw=0.3))
    ax.set_title(f"{label}\n" + rf"$R^2$={row['r2']:.2f}   tail kept {row['tail_capture']:.2f}")
    ax.set_xlabel("linear fit ($/MWh)")
    ax.set_ylabel(f"{target} ($/MWh)")


def order_panel(ax, rows):
    """Variance share by interaction order, stacked, one bar per case."""
    labels = [lab for _, lab in ADDITIVITY_CASES]
    shares = np.array([rows[name]["shares"] for name, _ in ADDITIVITY_CASES])
    names = [f"order {k}" for k in range(1, MAX_ORDER + 1)] + [f"order {MAX_ORDER + 1}+"]
    bottom = np.zeros(len(labels))
    for k, (name, color) in enumerate(zip(names, ORDER_COLORS)):
        ax.bar(labels, shares[:, k], bottom=bottom, color=color, label=name,
               width=0.7, edgecolor="w", lw=0.3)
        bottom += shares[:, k]
    for x, name in enumerate(name for name, _ in ADDITIVITY_CASES):
        ax.text(x, 0.03, f"range {rows[name]['spread']:.0f} $/MWh", color="w", ha="center",
                va="bottom", rotation=90)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of target variance")
    ax.tick_params(axis="x", rotation=20)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False)


def fit_figure(rows, target):
    """Grid of true-vs-fitted panels, one per case."""
    nrow = int(np.ceil(len(ADDITIVITY_CASES) / NCOL))
    fig, axes = plt.subplots(nrow, NCOL, figsize=(FIG_WIDTH, 1.95 * nrow))
    for ax, (name, label) in zip(axes.ravel(), ADDITIVITY_CASES):
        fit_panel(ax, rows[name], label, target)
    for ax in axes.ravel()[len(ADDITIVITY_CASES):]:
        ax.axis("off")
    fig.suptitle(f"every configuration's {target} against a linear model in the queue bits"
                 f"\n(dashed: identity;  red: the top-{TAIL_Q:.0%} level of the target)")
    fig.tight_layout()
    return fig


def orders_figure(rows, target):
    """Stacked variance share by interaction order across the case matrix."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.4))
    order_panel(ax, rows)
    ax.set_title(f"{target}: additive share (order 1) against interactions", pad=18)
    fig.tight_layout()
    return fig


def build(target="max_mu"):
    """Both figures for one target, as (fit, orders)."""
    plt.rcParams.update(STYLE)
    rows = compute_all(target)
    return fit_figure(rows, target), orders_figure(rows, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="sum_mu", help="target_cache column to fit")
    args = parser.parse_args()

    fit, orders = build(args.target)
    fit.savefig(RESULTS / f"mu_additivity_{args.target}_fit.png")
    orders.savefig(RESULTS / f"mu_additivity_{args.target}_orders.png")
    plt.show()


if __name__ == "__main__":
    main()
