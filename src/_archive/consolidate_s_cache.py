"""Consolidate the per-config mu cache into the scalar-target S cache.

The full sweep streams every config's load-averaged mu vector to `mu_cache/` in long
format (id, n_on, constraint, mu). The scalar sampling target is S(x) = max over
constraints of that load-averaged mu, so this reduces each config to its single peak:
S, the driving constraint, and n_on. All duals for a given config live in one part file
(parts are chunked by config), so the peak is exact within each part with no cross-part
merge. The result feeds reconstruct_lmp.

Usage:
    uv run python -m src.diagnostics.consolidate_s_cache --case activsg2000_2025
"""

import argparse
import glob

import pandas as pd

from cases import registry


def peak_rows(part):
    """Reduce one mu part to one row per config: the max-mu (driving) constraint."""
    df = pd.read_parquet(part)
    top = df.loc[df.groupby("id")["mu"].idxmax()]
    return top.rename(columns={"constraint": "driver", "mu": "driver_mu"})


def consolidate(case):
    """Peak every part, concatenate, and shape into the S-cache schema keyed by id."""
    tops = [peak_rows(p)
            for p in sorted(glob.glob(str(case.mu_cache / "part_*.parquet")))]
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
    parser = argparse.ArgumentParser()
    registry.add_case_arg(parser)
    case = registry.get(parser.parse_args().case)
    out = case.results_dir / "S_cache.parquet"
    df = consolidate(case)
    df.to_parquet(out)
    report(df)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
