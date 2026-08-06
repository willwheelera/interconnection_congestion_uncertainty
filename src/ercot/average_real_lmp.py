"""Time-averaged real ERCOT LMPs per settlement point --- a model-independent baseline.

Reads the cached SCED LMP-by-node/zone/hub tables (np6-788, 5-minute, columns
SCEDTimestamp / SettlementPoint / LMP) and reduces them over time to one average LMP
per settlement point, plus the spread statistics. Two quantities matter for comparison
with the reconstructed model LMPs (src/texas/reconstruct_lmp.py):

  - level: the mean LMP per point (includes the energy component, ~bulk price),
  - spatial spread: the range/std of the per-point time-means across points, which
    isolates the congestion-driven spatial variation (the model reconstructs only the
    congestion component, so this is the like-for-like comparison).

COVERAGE CAVEAT: the cached LMP files are the high-congestion sample days, not a uniform
year, so these averages are biased high relative to an annual mean. Point new day-files
at the same directory to widen coverage.

Usage:
    uv run python src/grid_data/average_real_lmp.py
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.ercot.data_paths import ERCOT_DATA
OUT = ERCOT_DATA / "real_lmp_time_average.parquet"


def load_lmp() -> pd.DataFrame:
    """Concatenate every cached node/zone/hub LMP day-file."""
    files = sorted(glob.glob(str(ERCOT_DATA / "lmp_node_zone_hub_2025-*.parquet")))
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df["day"] = df["SCEDTimestamp"].str.slice(0, 10)
    return df, len(files)


def per_point_average(df: pd.DataFrame) -> pd.DataFrame:
    """Time-mean LMP and dispersion per settlement point, over all cached intervals."""
    g = df.groupby("SettlementPoint")["LMP"]
    return pd.DataFrame({
        "mean_lmp": g.mean(), "std_lmp": g.std(), "max_lmp": g.max(),
        "min_lmp": g.min(), "n_intervals": g.size(),
    }).sort_values("mean_lmp", ascending=False)


def report(df: pd.DataFrame, per_point: pd.DataFrame, n_days: int) -> None:
    """Print the LMP level, its spatial spread, and the raw-interval magnitude tail."""
    m = per_point["mean_lmp"]
    print(f"real ERCOT SCED LMP over {n_days} cached high-congestion days, "
          f"{df['SettlementPoint'].nunique()} settlement points, {len(df):,} intervals\n")
    print("time-mean LMP across settlement points ($/MWh):")
    print(f"  system mean {m.mean():7.2f}   spatial spread max-min {m.max() - m.min():7.2f}"
          f"   std {m.std():6.2f}")
    for q in (0.5, 0.9, 0.99, 1.0):
        print(f"  point-mean q{q:<5} {m.quantile(q):8.2f}")
    print("\nraw 5-min LMP magnitude tail (all points, all intervals):")
    for q in (0.5, 0.99, 0.999, 1.0):
        print(f"  q{q:<6} {df['LMP'].quantile(q):9.2f}")
    print("\ntop settlement points by time-mean LMP:")
    print(per_point.head(8).to_string())


def plot_top(df):
    g = df.groupby("SettlementPoint")["LMP"]
    meandf = g.apply("mean").sort_values(ascending=False)
    names = meandf.index[:10]
    for name in names:
        tmp = df[df["SettlementPoint"] == name]
        tmp = tmp.sort_values("LMP", ignore_index=True)
        plt.plot(tmp["LMP"], label=name)
    plt.show()
        

def main():
    df, n_days = load_lmp()
    plot_top(df)
    #per_point = per_point_average(df)
    #per_point.to_parquet(OUT)
    #report(df, per_point, n_days)
    #print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
