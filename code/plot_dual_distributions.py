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

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUT_DIR = RESULTS_DIR


def _hourly(field, n_hours):
    """Expand a scalar-or-time-series field into a length-n_hours list."""
    return field["values"] if isinstance(field, dict) else [float(field)] * n_hours


def load_all_duals(result_path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load LMP and mu from every solution JSON, summed over hours.

    Each file is read once; generation total, load total, LMP, and μ are
    all extracted from the same parsed dict.

    Returns:
        lmp_all — DataFrame with columns = bus names, one row per scenario
        mu_all  — DataFrame with columns = branch names, one row per scenario
    """
    lmp_rows = {}
    mu_rows = {}
    skipped = 0
    for path in sorted(result_path.glob("*.json")):
        avail_id = int(path.stem.split("avail")[1])
        if sum(map(int, np.binary_repr(avail_id))) < 3:
            continue
        raw = json.loads(path.read_text())
        n_hours = len(raw["system"]["time_keys"])

        total_gen = np.zeros(n_hours)
        for g in raw["elements"]["generator"].values():
            total_gen += np.array(_hourly(g.get("pg", 0.0), n_hours))

        total_load = np.zeros(n_hours)
        for load in raw["elements"]["load"].values():
            total_load += np.array(_hourly(load["p_load"], n_hours))

        if not np.allclose(total_gen, total_load, rtol=1e-6, atol=1e-6):
            skipped += 1
            print(f"Skipping {avail_id}")
            continue

        lmp_sum = {}
        for name, b in raw["elements"]["bus"].items():
            lmp = b.get("lmp")
            if lmp is None:
                continue
            lmp_sum[name] = float(np.sum(_hourly(lmp, n_hours)))

        mu_sum = {}
        for name, br in raw["elements"]["branch"].items():
            mu_u = br.get("mu_upper")
            mu_l = br.get("mu_lower")
            if mu_u is None or mu_l is None:
                continue
            upper = np.array(_hourly(mu_u, n_hours))
            lower = np.array(_hourly(mu_l, n_hours))
            mu_sum[name] = float((upper + lower).sum()) # sum abs instead of positives/negatives

        lmp_rows[avail_id] = pd.Series(lmp_sum)
        mu_rows[avail_id] = pd.Series(mu_sum)
    if skipped:
        print(f"Skipped {skipped} scenarios with load/generation mismatch")
    lmp_all = pd.DataFrame(lmp_rows).T
    lmp_all.index.name = "avail_id"
    mu_all = pd.DataFrame(mu_rows).T
    mu_all.index.name = "avail_id"
    return lmp_all, mu_all


def plot_lmp_distributions(ax: plt.Axes, lmp_all: pd.DataFrame) -> None:
    """Histogram of LMP values, one color per bus."""
    buses = lmp_all.columns
    cmap = plt.colormaps["tab10"].resampled(len(buses))
    all_vals = lmp_all.values.ravel()
    all_vals = all_vals[np.isfinite(all_vals)]
    bins = np.linspace(all_vals.min(), all_vals.max(), 80)
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
    #ax.legend(fontsize=8)


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
    bins = np.linspace(nz.min(), nz.max(), 80)
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


def main(result_path):
    lmp_all, mu_all = load_all_duals(result_path)
    n_scenarios = len(lmp_all)
    print(f"Loaded {n_scenarios} scenarios (duals summed over hours)")

    fig, axes = plt.subplots(3, 1, figsize=(8, 9))

    plot_lmp_distributions(axes[0], lmp_all)
    plot_mu_distributions(axes[1], mu_all)
    plot_congestion_frequency(axes[2], mu_all)

    # print out the highest dual cases
    mu_all["total"] = mu_all.sum(axis=1).abs()
    lmp_all["total"] = lmp_all.sum(axis=1).abs()
    tmp_mu = mu_all.sort_values("total", ascending=False)
    tmp_lmp = lmp_all.sort_values("total", ascending=False)
    print(tmp_lmp[:5].round(0))
    print(tmp_mu[:5].round(0))

    fig.suptitle(f"Dual variable distributions ({n_scenarios} scenarios)", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = OUT_DIR / "dual_distributions.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main(RESULTS_DIR / "test_scuc_masked")
