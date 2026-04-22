"""
Plot distributions of dual variables (lambda/LMP and mu) across all
UC solutions in results/gen8_four_bus.

Usage:
    uv run python code/plot_dual_distributions.py
"""

import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
    all extracted from the same parsed dict. Works for both multi-period UC
    results (fields are dicts of hourly values) and single-period OPF
    results (fields are scalars) — in the OPF case "sum over hours" is
    just the single value.

    Returns:
        lmp_all — DataFrame with columns = bus names, one row per scenario
        mu_all  — DataFrame with columns = branch names, one row per scenario
    """
    lmp_rows = {}
    mu_rows = {}
    skipped = 0
    for path in sorted(result_path.glob("*.json.gz")):
        stem = path.stem.replace(".json", "")
        if "avail" in stem:
            avail_id = int(stem.split("avail")[1])
        else:
            avail_id = stem
        with gzip.open(path, "rt") as f:
            raw = json.load(f)
        time_keys = raw["system"].get("time_keys")
        n_hours = len(time_keys) if time_keys else 1
        if isinstance(avail_id, int) and time_keys and sum(map(int, np.binary_repr(avail_id))) < 3:
            continue

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


def build_dual_cache(result_path: Path) -> Path:
    """Aggregate per-scenario dual sums into a single Parquet file.

    Reads every uc_*.json.gz in *result_path*, computes hourly sums of LMP
    and mu, and writes two column groups into one Parquet file:
      - lmp/<bus_name>  for each bus
      - mu/<branch_name> for each branch
    Rows are indexed by avail_id.  Scenarios that fail the gen/load balance
    check are excluded.

    Returns the path to the written Parquet file.
    """
    lmp_all, mu_all = load_all_duals(result_path)
    combined = pd.concat(
        {"lmp": lmp_all, "mu": mu_all}, axis=1,
    )
    out = result_path / "dual_cache.parquet"
    combined.to_parquet(out)
    print(f"Wrote {out} ({len(combined)} scenarios)")
    return out


def load_dual_cache(result_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached dual sums from Parquet, rebuilding if missing."""
    cache = result_path / "dual_cache.parquet"
    if not cache.exists():
        build_dual_cache(result_path)
    df = pd.read_parquet(cache)
    lmp_all = df["lmp"]
    mu_all = df["mu"]
    lmp_all.index.name = "avail_id"
    mu_all.index.name = "avail_id"
    return lmp_all, mu_all


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
    lmp_total = lmp_all[lmp_cols].sum(axis=1)
    mu_total = mu_all[mu_cols].sum(axis=1)

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
                           color="lightgray", edgecolor="none")
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


def main(result_path):
    lmp_all, mu_all = load_dual_cache(result_path)
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
    print(tmp_lmp[:5].round(0).astype(int))
    print(tmp_mu[:5].round(0).astype(int))

    fig.suptitle(f"Dual variable distributions ({result_path.name}, {n_scenarios} scenarios)", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = result_path / f"dual_distributions_{result_path.name}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved to {out_path}")

    g = plot_lmp_vs_mu(lmp_all, mu_all)
    scatter_path = result_path / f"lmp_vs_mu_{result_path.name}.pdf"
    g.savefig(scatter_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    plt.close(g.figure)
    print(f"Saved to {scatter_path}")


if __name__ == "__main__":
    import sys
    #subdir = sys.argv[1] if len(sys.argv) > 1 else "test_scuc_masked"
    subdir = "case14_gen_per_bus"

    #build_dual_cache(RESULTS_DIR / subdir)
    main(RESULTS_DIR / subdir)
