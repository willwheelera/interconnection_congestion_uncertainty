"""
Plot distributions of dual variables (lambda/LMP and mu) across all
UC solutions in results/gen8_four_bus.

Usage:
    uv run python code/plot_dual_distributions.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from plot_optimal_values import load_lmp_mu

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "gen8_four_bus"
OUT_DIR = RESULTS_DIR


def load_all_duals() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load LMP and mu from every solution JSON, summed over hours.

    Returns:
        lmp_all — DataFrame with columns = bus names, one row per scenario
        mu_all  — DataFrame with columns = branch names, one row per scenario
    """
    lmp_rows = []
    mu_rows = []
    for path in sorted(RESULTS_DIR.glob("uc_gen8_four_bus_avail*.json")):
        lmp_df, mu_df = load_lmp_mu(path)
        lmp_rows.append(lmp_df.sum())
        mu_rows.append(mu_df.sum())
    lmp_all = pd.DataFrame(lmp_rows).reset_index(drop=True)
    mu_all = pd.DataFrame(mu_rows).reset_index(drop=True)
    return lmp_all, mu_all


def plot_lmp_distributions(ax: plt.Axes, lmp_all: pd.DataFrame) -> None:
    """Histogram of LMP values, one color per bus."""
    buses = lmp_all.columns
    cmap = plt.colormaps["tab10"].resampled(len(buses))
    all_vals = lmp_all.values.ravel()
    all_vals = all_vals[np.isfinite(all_vals)]
    bins = np.linspace(all_vals.min(), all_vals.max(), 40)
    for i, bus in enumerate(buses):
        vals = lmp_all[bus].dropna().values
        ax.hist(vals, bins=bins, alpha=0.5, color=cmap(i), label=bus, edgecolor="none")
        if len(vals) > 1 and np.std(vals) > 1e-10:
            kde = gaussian_kde(vals)
            x_kde = np.linspace(bins[0], bins[-1], 200)
            bin_width = bins[1] - bins[0]
            ax.plot(x_kde, kde(x_kde) * len(vals) * bin_width, color=cmap(i), linewidth=1.5)
    ax.set_xlabel("Σ λ ($/MWh)")
    ax.set_ylabel("Count (scenarios)")
    ax.set_title("LMP sum-over-hours distribution")
    ax.legend(fontsize=8)


def plot_mu_distributions(ax: plt.Axes, mu_all: pd.DataFrame) -> None:
    """Histogram of mu values, one color per branch."""
    branches = mu_all.columns
    cmap = plt.colormaps["Set1"].resampled(max(len(branches), 1))
    nonzero_counts = (mu_all.abs() > 1e-6).sum()
    all_vals = mu_all.values.ravel()
    all_vals = all_vals[np.isfinite(all_vals)]
    nz = all_vals[np.abs(all_vals) > 1e-6]
    if len(nz) == 0:
        ax.text(0.5, 0.5, "All μ values are zero", transform=ax.transAxes,
                ha="center", va="center")
        return
    bins = np.linspace(nz.min(), nz.max(), 40)
    for i, br in enumerate(branches):
        vals = mu_all[br].dropna().values
        vals = vals[np.abs(vals) > 1e-6]
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=bins, alpha=0.5, color=cmap(i), label=br, edgecolor="none")
        if len(vals) > 1 and np.std(vals) > 1e-10:
            kde = gaussian_kde(vals)
            x_kde = np.linspace(bins[0], bins[-1], 200)
            bin_width = bins[1] - bins[0]
            ax.plot(x_kde, kde(x_kde) * len(vals) * bin_width, color=cmap(i), linewidth=1.5)
    ax.set_xlabel("Σ μ ($/MWh)")
    ax.set_ylabel("Count (scenarios)")
    ax.set_title("Congestion dual sum-over-hours distribution (nonzero only)")
    ax.legend(fontsize=8)


def plot_congestion_frequency(ax: plt.Axes, mu_all: pd.DataFrame) -> None:
    """Bar chart: fraction of scenarios where each branch has nonzero total congestion."""
    branches = mu_all.columns
    cmap = plt.colormaps["Set1"].resampled(max(len(branches), 1))
    freq = (mu_all.abs() > 1e-6).mean()
    colors = [cmap(i) for i in range(len(branches))]
    ax.bar(freq.index, freq.values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Fraction of scenarios congested")
    ax.set_title("Congestion frequency by branch")
    ax.set_ylim(0, min(1.0, freq.max() * 1.3) if freq.max() > 0 else 1.0)
    for i, (br, v) in enumerate(freq.items()):
        ax.text(i, v + 1e-4, f"{v:.2%}", ha="center", va="bottom", fontsize=8)


def main():
    lmp_all, mu_all = load_all_duals()
    n_scenarios = len(list(RESULTS_DIR.glob("uc_gen8_four_bus_avail*.json")))
    print(f"Loaded {n_scenarios} scenarios (duals summed over hours)")

    fig, axes = plt.subplots(3, 1, figsize=(8, 9))

    plot_lmp_distributions(axes[0], lmp_all)
    plot_mu_distributions(axes[1], mu_all)
    plot_congestion_frequency(axes[2], mu_all)

    fig.suptitle(f"Dual variable distributions ({n_scenarios} scenarios)", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = OUT_DIR / "dual_distributions.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
