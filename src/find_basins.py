"""Basins of attraction of the scalar target S over the 2^20 config hypercube.

Greedy steepest-neighbor hill climb: each config maps to its best single-bit-flip
neighbor (or itself). Following that map to a fixed point assigns every config to a
local maximum of S. Reports how many basins exist, their sizes, and their peaks.

Runs on the full ACTIVSg2000 sweep results/activsg2000/S_cache.parquet (queue of 20
gens, one scalar S per config).

Usage:
    uv run python src/find_basins.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
CASE_DIR = REPO_ROOT / "results" / "activsg2000"


def load_target(path):
    """Load S_cache as arrays indexed by config id (requires a full contiguous sweep)."""
    df = pd.read_parquet(path).sort_index()
    n = int(round(np.log2(len(df))))
    assert len(df) == 1 << n and df.index[0] == 0 and df.index[-1] == len(df) - 1
    return df["S"].to_numpy(), df["n_on"].to_numpy(), n


def steepest_neighbor(S, n):
    """Map each config to its highest-S single-bit-flip neighbor, or itself if a local max."""
    ids = np.arange(len(S))
    ascent = ids.copy()
    for i in range(n):
        neighbor = ids ^ (1 << i)
        bigger = S[neighbor] > S[ascent]
        ascent[bigger] = neighbor[bigger]
    return ascent


def basins_of(ascent):
    """Compose the ascent map with itself (pointer doubling) until it reaches fixed
    points, giving each config's local-max basin. Converges in at most log2 passes."""
    for _ in range(int(np.log2(len(ascent))) + 1):
        nxt = ascent[ascent]
        if np.array_equal(nxt, ascent):
            break
        ascent = nxt
    return ascent


def basin_stats(S, n_on, basin):
    """One row per basin: size, peak S, mean S over its members, and peak n_on."""
    peaks, counts = np.unique(basin, return_counts=True)
    sums = np.bincount(basin, weights=S)[peaks]
    stats = pd.DataFrame(index=peaks)
    stats.index.name = "basin"
    stats["log_count"] = np.log(counts)
    stats["peak_S"] = S[peaks]
    stats["avg_S"] = sums / counts
    stats["n_on"] = n_on[peaks]
    return stats.sort_values("peak_S", ascending=False)


def plot_basins(stats, out):
    """Pairplot of basin size, peak, and mean, saved to out."""
    g = sns.pairplot(stats[["log_count", "peak_S", "avg_S", "n_on"]], height=2)
    g.figure.suptitle(f"{len(stats)} basins over 2^20 configs", y=1.02)
    plt.savefig(out, bbox_inches="tight")
    plt.show()


def main():
    S, n_on, n = load_target(CASE_DIR / "S_cache.parquet")
    ascent = steepest_neighbor(S, n)
    basin = basins_of(ascent)
    stats = basin_stats(S, n_on, basin)
    print(f"{len(stats)} basins over {len(S)} configs")
    print(f"global max S {stats.peak_S.iloc[0]:.1f} at config {stats.index[0]} "
          f"(n_on={stats.n_on.iloc[0]}, basin size {int(np.exp(stats.log_count.iloc[0]))})")
    print(stats.head(10).to_string())
    plot_basins(stats, CASE_DIR / "basin_stats.pdf")


if __name__ == "__main__":
    main()
