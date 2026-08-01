"""Consolidate the per-config mu cache into the scalar-target S cache.

The full sweep streams every config's load-averaged mu vector to `mu_cache/` in long
format (id, n_on, constraint, mu). The scalar sampling target is S(x) = max over
constraints of that load-averaged mu, so this reduces each config to its single peak:
S, the driving constraint, and n_on. All duals for a given config live in one part file
(parts are chunked by config), so the peak is exact within each part with no cross-part
merge. The result feeds reconstruct_lmp and score_samplers.

Usage:
    HARNESS_CASE=2025 uv run python -m texas.consolidate_s_cache
"""

import glob
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO / "src" / "texas"))

from sample_harness import RESULTS_NAME          # results dir of the HARNESS_CASE sweep

RESULTS = REPO / "results" / RESULTS_NAME
MU_DIR = RESULTS / "mu_cache"
OUT = RESULTS / "S_cache.parquet"


def peak_rows(part):
    """Reduce one mu part to one row per config: the max-mu (driving) constraint."""
    df = pd.read_parquet(part)
    top = df.loc[df.groupby("id")["mu"].idxmax()]
    return top.rename(columns={"constraint": "driver", "mu": "driver_mu"})


def consolidate():
    """Peak every part, concatenate, and shape into the S-cache schema keyed by id."""
    tops = [peak_rows(p) for p in sorted(glob.glob(str(MU_DIR / "part_*.parquet")))]
    df = pd.concat(tops, ignore_index=True)
    df["S"] = df["driver_mu"]
    return df.set_index("id").sort_index()[["n_on", "S", "driver", "driver_mu"]]


def report(df):
    """Print the S distribution and the constraints that drive it."""
    print(f"consolidated {len(df)} configs")
    for q in (0.0, 0.5, 0.9, 0.99, 0.999, 1.0):
        print(f"  q{q:<6} S {df.S.quantile(q):8.1f} $/MWh")
    print(f"  configs at cap (S>=3499): {(df.S >= 3499).sum()}")
    g = df.groupby("driver").agg(n=("S", "size"), mean_S=("S", "mean")).sort_values(
        "n", ascending=False)
    print(f"\ntop driving constraints (count | mean S):\n{g.head(8).to_string()}")


def main():
    df = consolidate()
    df.to_parquet(OUT)
    report(df)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
