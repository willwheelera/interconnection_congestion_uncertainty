"""
Plot distributions of dual variables (lambda/LMP and mu) across all
UC solutions in results/gen8_four_bus.

Usage:
    uv run python src/plot_dual_distributions.py
"""

import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from utils.solution_io import load_dual_cache

from paths import REPO_ROOT
RESULTS_DIR = REPO_ROOT / "results"
OUT_DIR = RESULTS_DIR


def plot_lmp_distributions(ax: plt.Axes, lmp_all: pd.DataFrame) -> None:
    """Histogram of LMP values, one color per bus (per-bus bins)."""
    buses = lmp_all.columns
    cmap = plt.colormaps["tab10"].resampled(len(buses))
    for i, bus in enumerate(buses):
        vals = lmp_all[bus].dropna().values
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        lo, hi = vals.min(), vals.max()
        if hi <= lo:
            hi = lo + 1e-12
        bins = np.linspace(lo, hi, 80)
        bin_width = bins[1] - bins[0]
        ax.hist(vals, bins=bins, alpha=0.5, color=cmap(i), label=bus, edgecolor="none")
        if len(vals) > 1 and np.std(vals) > 1e-10:
            kde = gaussian_kde(vals)
            x_kde = np.linspace(lo, hi, 200)
            ax.plot(x_kde, kde(x_kde) * len(vals) * bin_width, color=cmap(i), linewidth=1.5)
    ax.set_xlabel("Σ λ ($/MWh)")
    ax.set_ylabel("Count (scenarios)")
    ax.set_title("LMP sum-over-hours distribution")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.8, top=len(lmp_all) * 1.5)
    ax.legend(fontsize=8, ncol=2, loc="center left", bbox_to_anchor=(1.02, 0.5),
              title="Bus", borderaxespad=0.)


def plot_mu_distributions(ax: plt.Axes, mu_all: pd.DataFrame, tol=1e-2) -> None:
    """Histogram of mu values, one color per branch."""
    branches = mu_all.columns
    cmap = plt.colormaps["tab20"].resampled(max(len(branches), 1))
    nonzero_counts = (mu_all.abs() > tol).sum() # empirically looks like 1e-2 is good cutoff
    all_vals = mu_all.values.ravel()
    all_vals = all_vals[np.isfinite(all_vals)]
    nz = all_vals[np.abs(all_vals) > tol]
    if len(nz) == 0:
        ax.text(0.5, 0.5, "All μ values are zero", transform=ax.transAxes,
                ha="center", va="center")
        return
    for i, br in enumerate(branches):
        vals = mu_all[br].dropna().values
        vals = vals[np.abs(vals) > 1e-6]
        if len(vals) == 0:
            continue
        lo, hi = vals.min(), vals.max()
        if hi <= lo:
            hi = lo + 1e-12
        bins = np.linspace(lo, hi, 80)
        bin_width = bins[1] - bins[0]
        ax.hist(vals, bins=bins, alpha=0.5, color=cmap(i), label=br, edgecolor="none")
        if len(vals) > 1 and np.std(vals) > 1e-10:
            kde = gaussian_kde(vals)
            x_kde = np.linspace(lo, hi, 200)
            ax.plot(x_kde, kde(x_kde) * len(vals) * bin_width, color=cmap(i), linewidth=1.5)
    ax.set_xlabel("Σ μ ($/MWh)")
    ax.set_ylabel("Count (scenarios)")
    ax.set_title(f"Congestion dual (sum) distribution (nonzero >{tol:.0e} only)")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.8, top=len(mu_all) * 1.5)
    ax.legend(fontsize=8, ncol=2, loc="center left", bbox_to_anchor=(1.02, 0.5),
              title="Branch", borderaxespad=0.)


def plot_congestion_frequency(ax: plt.Axes, mu_all: pd.DataFrame) -> None:
    """Bar chart: fraction of scenarios where each branch has nonzero total congestion."""
    branches = mu_all.columns
    cmap = plt.colormaps["tab20"].resampled(max(len(branches), 1))
    freq = (mu_all.abs() > 1e-6).mean()
    colors = [cmap(i) for i in range(len(branches))]
    ax.bar(freq.index, freq.values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Fraction of scenarios congested")
    ax.set_title("Congestion frequency by branch")
    ax.set_ylim(0, 1.0)
    for i, (br, v) in enumerate(freq.items()):
        ax.text(i, v + 1e-4, f"{v:.2%}", ha="center", va="bottom", fontsize=8)


def plot_lmp_vs_mu(lmp_all: pd.DataFrame, mu_all: pd.DataFrame, tol=1e-2) -> sns.JointGrid:
    """Joint scatter plot of total LMP vs total mu, colored by highest-μ branch.

    Scenarios whose max |μ| falls below *tol* are drawn in gray with no legend
    entry; the rest are colored by the branch carrying the largest |μ|.
    """
    lmp_cols = [c for c in lmp_all.columns if c != "total"]
    mu_cols = [c for c in mu_all.columns if c != "total"]

    lmp_total = np.linalg.norm(lmp_all[lmp_cols], axis=1)
    mu_total = np.linalg.norm(mu_all[mu_cols], axis=1)

    mu_abs = mu_all[mu_cols].abs()
    max_val = mu_abs.max(axis=1)
    max_branch = mu_abs.idxmax(axis=1).where(max_val >= tol, other=None)

    cmap = plt.colormaps["tab20"].resampled(max(len(mu_cols), 1))
    palette = {br: cmap(i) for i, br in enumerate(mu_cols)}

    df = pd.DataFrame({"lmp": lmp_total, "mu": mu_total, "hue": max_branch})

    g = sns.JointGrid(data=df, x="lmp", y="mu")
    bg = df[df["hue"].isna()]
    if not bg.empty:
        g.ax_joint.scatter(bg["lmp"], bg["mu"], s=12, alpha=0.3,
                           color="white", edgecolor="gray")
    for br in mu_cols:
        sub = df[df["hue"] == br]
        if sub.empty:
            continue
        g.ax_joint.scatter(sub["lmp"], sub["mu"], s=12, alpha=0.7,
                           color=palette[br], edgecolor="none", label=br)
    g.ax_joint.legend(fontsize=8, title="max-|μ| branch", loc="best")

    g.ax_marg_x.hist(df["lmp"].dropna(), bins=40, color="steelblue", alpha=0.7)
    g.ax_marg_y.hist(df["mu"].dropna(), bins=40, color="steelblue", alpha=0.7,
                     orientation="horizontal")

    g.set_axis_labels("Σ λ across buses ($/MWh)", "Σ μ across branches ($/MWh)")
    g.figure.suptitle("Total LMP vs total congestion dual per scenario", y=1.02)
    _annotate_marginal_counts(g.ax_marg_x, orient="v")
    _annotate_marginal_counts(g.ax_marg_y, orient="h")
    return g


def _annotate_marginal_counts(ax: plt.Axes, orient: str) -> None:
    """Label each histogram bar in a marginal axis with its sample count.

    orient='v': vertical bars (top marginal); label above each bar.
    orient='h': horizontal bars (right marginal); label to the right.
    """
    for patch in ax.patches:
        count = patch.get_height() if orient == "v" else patch.get_width()
        if count <= 0:
            continue
        if orient == "v":
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_height()
            ax.text(x, y, f"{int(count)}", ha="center", va="bottom", fontsize=6)
        else:
            x = patch.get_width()
            y = patch.get_y() + patch.get_height() / 2
            ax.text(x, y, f"{int(count)}", ha="left", va="center", fontsize=6)


def print_highest(df, n=10):
    df = df.copy()
    df["total"] = df.sum(axis=1).abs()
    df = df.sort_values("total", ascending=False)
    print(df[:n].round(0).astype(int))


def all_distributions(lmp_all, mu_all, result_path):
    n_scenarios = len(lmp_all)
    fig, axes = plt.subplots(3, 1, figsize=(8, 9))

    plot_lmp_distributions(axes[0], lmp_all)
    plot_mu_distributions(axes[1], mu_all)
    plot_congestion_frequency(axes[2], mu_all)

    fig.suptitle(f"Dual variable distributions ({result_path.name}, {n_scenarios} scenarios)", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def main(result_path):
    df = load_dual_cache(result_path / "dual_cache.parquet")
    lmp_all, mu_all = df["lmp"], df["mu"]
    print_highest(mu_all)
    print_highest(lmp_all)

    fig = all_distributions(lmp_all, mu_all, result_path)
    out_path = result_path / f"dual_distributions_{result_path.name}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved to {out_path}")

    g = plot_lmp_vs_mu(lmp_all, mu_all)
    scatter_path = result_path / f"lmp_vs_mu_{result_path.name}.pdf"
    g.savefig(scatter_path, bbox_inches="tight")
    print(f"Saved to {scatter_path}")

    plt.show()


if __name__ == "__main__":
    import sys
    #subdir = sys.argv[1] if len(sys.argv) > 1 else "test_scuc_masked"
    subdir = "case14_gen_per_bus"

    #build_dual_cache(RESULTS_DIR / subdir)
    main(RESULTS_DIR / subdir)
