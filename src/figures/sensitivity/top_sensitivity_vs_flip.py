"""Test whether sensitivity predicts mu changes under generator-availability flips.

The cached sensitivity matrix S(x) holds, for one base configuration x,

    S(x)_{l, j}  =  d mu_l / d load_j   evaluated at the OPF solution of x.

Subtracting load at bus j is the same as adding injection there, so flipping
generator k on->off (at bus b = bus(k)) is approximately an *increase* in
load at bus b by the amount that generator was producing.  The first-order
prediction is therefore

    mu_l(y) - mu_l(x)  ~  sign_flip * |delta P_k| * S(x)_{l, b(k)},

where sign_flip = +1 if x_k=1 -> y_k=0 (gen drops out, load effectively
rises), and sign_flip = -1 in the other direction.  We have no |delta P_k|
on hand, so this script just looks at whether the *signed* delta mu lines
up with the *signed* sensitivity entry, lumping the |delta P| factor into
the scatter.

Picks one base configuration, gathers its 14 one-flip neighbors, and plots
(S, delta mu) for every (branch l, flipped generator k) pair.  Each point
is one branch's dual change for one neighbor; color encodes flip direction.

Usage:
    uv run python src/plot_sensitivity_vs_flip.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.availability import id_to_avail, avail_to_id, get_all_neighbors
from src.utils.solution_io import load_gen_bus_map
from src.figures.sensitivity_vs_flip import report_metrics, collect_points_all

from src.paths import REPO_ROOT
RESULTS_DIR = REPO_ROOT / "results" / "case14_gen_per_bus"
CASE_PATH = REPO_ROOT / "data" / "case14_gen_per_bus.json"
NGEN = 14
BASE_AVAIL_ID = 127


def plot_top_sensitivity(pts: pd.DataFrame, out_path: Path, eps: float = 1e-6) -> None:
    """Scatter of dmu vs S_signed split by highest-S and lowest-S point per config.

    Each configuration contributes one point per row: the (branch, generator)
    pair with the largest (most positive) S_signed in the top row, and the
    pair with the smallest (most negative) S_signed in the bottom row.  Both
    rows therefore have the same number of points (one per config).  This
    isolates the accuracy of the prediction when we trust only the
    sensitivity's most extreme signed entry.
    """
    high_pts = extreme_per_config(pts, "S_signed", "max")
    low_pts = extreme_per_config(pts, "S_signed", "min")
    max_gen = max(high_pts["p_gen"].max(), low_pts["p_gen"].max())

    fig, axs = plt.subplots(2, 2, figsize=(9, 8))
    for row, (sub, label) in enumerate([(high_pts, "highest $S$"), (low_pts, "lowest $S$")]):
        ax = axs[row, 0]
        c = ax.scatter(sub["S_signed"], sub["dmu"], c=sub["p_gen"],
                       s=18, alpha=0.6, linewidths=0, vmin=0.0, vmax=max_gen)
        corr = np.corrcoef(sub["S_signed"], sub["dmu"])[0, 1]
        print("correlation", corr)
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel(r"extreme $S$:  flip_sign $\cdot\; \partial \mu_l / \partial \mathrm{load}_{b(k)}$")
        ax.set_ylabel(r"$\mu_l(y_k) - \mu_l(x)$  (same $l,k$)")
        ax.set_title(f"{label}: correlation {corr:.2f}")
        plt.colorbar(c, ax=ax, label="p_gen at b(k)")

        ax = axs[row, 1]
        bins = np.linspace(sub["dmu"].min(), sub["dmu"].max(), 30) if len(sub) else 30
        ax.hist(sub["dmu"][sub["S_signed"] > eps], bins=bins, alpha=0.6, label="S > 0")
        ax.hist(sub["dmu"][sub["S_signed"] < -eps], bins=bins, alpha=0.6, label="S < 0")
        ax.hist(sub["dmu"][sub["S_signed"].abs() <= eps], bins=bins, alpha=0.6, label="S = 0")
        ax.set_xlabel(r"$\Delta\mu$ at extreme-$S$ point")
        ax.set_title(f"{label} $\\Delta\\mu$ histogram")
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Wrote {out_path}")
    report_metrics(high_pts["S_signed"].values, high_pts["dmu"].values, "highest-S")
    report_metrics(low_pts["S_signed"].values, low_pts["dmu"].values, "lowest-S")


def extreme_per_config(pts: pd.DataFrame, col: str, mode: str) -> pd.DataFrame:
    """For each config_id, keep the row at the raw extremum of pts[col].

    mode="max" returns the row with the largest (most positive) value;
    mode="min" returns the row with the smallest (most negative) value.
    """
    grp = pts.groupby("config_id")[col]
    idx = grp.idxmax() if mode == "max" else grp.idxmin()
    return pts.loc[idx]


def plot_top_bus_heatmap(pts: pd.DataFrame, gen_buses: list[str],
                         out_path: Path) -> None:
    """2x2 grid of (bus-of-extreme-S, bus-of-extreme-dmu) heatmaps.

    Rows: switch-on flips (flip_sign < 0) vs switch-off flips (flip_sign > 0).
    Columns: largest (raw max) vs smallest (raw min) S_signed and dmu, per
    config.  Each cell shows bus_of_extreme_S on the y-axis and
    bus_of_extreme_dmu on the x-axis; mass on the diagonal means
    sensitivity's extreme bus matches dmu's extreme bus.
    """
    buses = sorted(set(gen_buses), key=int)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    rows = [("switch-on", pts[pts["flip_sign"] < 0]),
            ("switch-off", pts[pts["flip_sign"] > 0])]
    cols = [("largest", "max"), ("smallest", "min")]

    for i, (row_label, sub) in enumerate(rows):
        for j, (col_label, mode) in enumerate(cols):
            bus_S = extreme_per_config(sub, "S_signed", mode).set_index("config_id")["bus"]
            bus_D = extreme_per_config(sub, "dmu", mode).set_index("config_id")["bus"]
            H = pd.crosstab(bus_D, bus_S).reindex(index=buses, columns=buses, fill_value=0)
            H = H.astype(float)
            H[H == 0] = 0.7
            ax = axs[i, j]
            im = ax.imshow(H.values, origin="lower", cmap="viridis", aspect="equal", norm="log", vmin=0.7, vmax=3.5e3)
            ax.set_xticks(range(len(buses)))
            ax.set_yticks(range(len(buses)))
            ax.set_xticklabels(buses, fontsize=7)
            ax.set_yticklabels(buses, fontsize=7)
            ax.set_xlabel(f"bus of {col_label} $S$")
            ax.set_ylabel(f"bus of {col_label} $\\Delta\\mu$")
            total = int(H.values.sum())
            diag = float(np.trace(H.values)) / max(total, 1)
            ax.set_title(f"{row_label}, {col_label}  ($n$={total}, diag={diag:.3f})")
            plt.colorbar(im, ax=ax, label="configs")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Wrote {out_path}")


def main():
    import time
    t0 = time.perf_counter()
    df = pd.read_parquet(RESULTS_DIR / "dual_cache.parquet")
    sens = pd.read_parquet(RESULTS_DIR / "sensitivity_cache.parquet")
    assert np.linalg.norm(df.index - sens.index) == 0
    gen_buses = load_gen_bus_map(CASE_PATH)

    print(f"loaded data {time.perf_counter() - t0:.3f}")
    pts_all = collect_points_all(df, sens, gen_buses)
    print(f"collected points {time.perf_counter() - t0:.3f}")

    plot_top_sensitivity(pts_all, RESULTS_DIR / "sensitivity_vs_flip_top.pdf")
    plot_top_bus_heatmap(pts_all, gen_buses,
                         RESULTS_DIR / "sensitivity_vs_flip_bus_heatmap.pdf")

    plt.show()


if __name__ == "__main__":
    main()
