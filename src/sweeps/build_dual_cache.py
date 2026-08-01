"""Build a compact wide dual_cache.parquet for a long-format case from its mu_cache.

The per-line tail plots read a dense per-line table via Problem.mu_table (a
dual_cache.parquet whose "mu" columns are the lines).  The security-constrained sweeps
instead store a sparse long mu_cache (id, n_on, constraint, mu), one row per binding
dual.  This script
selects a small set of constraints -- the top N ranked by their global max |mu| across all
configs, or (with --threshold) every constraint whose global max exceeds a floor -- and
scatters them into a dense id x constraint table, matching the wide format mu_table expects.
Congestion concentrates on a handful of lines, so 25 columns capture the tail.

The mu source is the case's registry `mu_cache`, the same one build_target_cache reduces,
so the per-line truth matches the aggregates the existing target_cache and
experiment_samples were scored against.

Usage:
    uv run python -m sweeps.build_dual_cache --case all
    uv run python -m sweeps.build_dual_cache --case activsg2000_2025 --top 25
    uv run python -m sweeps.build_dual_cache --case activsg2000_original --threshold 10
"""

import argparse
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from cases import registry


def constraint_global_max(parts):
    """One streaming pass over the long mu_cache: max |mu| over all configs per constraint."""
    cmax = defaultdict(float)
    for part in parts:
        d = pd.read_parquet(part, columns=["constraint", "mu"])
        for con, v in d.groupby("constraint")["mu"].max().items():
            cmax[con] = max(cmax[con], float(v))
    cmax.pop("none", None)                               # the no-binding sentinel row
    return cmax


def select_constraints(cmax, top, threshold):
    """Constraints kept, ordered by descending global max |mu| (top N, or > threshold)."""
    ranked = sorted(cmax, key=cmax.get, reverse=True)
    if threshold is not None:
        return [c for c in ranked if cmax[c] > threshold]
    return ranked[:top]


def fill_table(parts, selected, nconf):
    """Scatter the selected constraints' mu into a dense (nconf x nselected) float32 table."""
    col = {c: i for i, c in enumerate(selected)}
    keep = set(selected)
    table = np.zeros((nconf, len(selected)), dtype=np.float32)
    for part in parts:
        d = pd.read_parquet(part, columns=["id", "constraint", "mu"])
        d = d[d["constraint"].isin(keep)]
        table[d["id"].to_numpy(), d["constraint"].map(col).to_numpy()] = d["mu"].to_numpy()
    return table


def wide_dual_cache(table, selected):
    """Nonzero-config rows as a wide frame with ("mu", constraint) MultiIndex columns."""
    nonzero = table.any(axis=1)
    cols = pd.MultiIndex.from_product([["mu"], selected])
    df = pd.DataFrame(table[nonzero], index=np.nonzero(nonzero)[0], columns=cols)
    df.index.name = "id"
    return df


def build(case, top, threshold):
    """Build and write results/<case>/dual_cache.parquet for one long-format case."""
    parts = sorted(glob.glob(str(Path(case.mu_cache) / "part_*.parquet")))
    cmax = constraint_global_max(parts)
    selected = select_constraints(cmax, top, threshold)
    nconf = 1 << int(pd.read_parquet(parts[-1], columns=["id"])["id"].max()).bit_length()
    table = fill_table(parts, selected, nconf)
    df = wide_dual_cache(table, selected)
    dest = case.dual_cache
    df.to_parquet(dest)
    size_mb = dest.stat().st_size / 1e6
    print(f"{case.name:<24} lines={len(selected)}  rows={len(df)}  {size_mb:.1f} MB  -> {dest}")
    print("  " + ", ".join(f"{c}={cmax[c]:.0f}" for c in selected))


def main():
    parser = argparse.ArgumentParser()
    registry.add_case_arg(parser, default="all", allow_all=True)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    for case in registry.resolve(args.case):
        if case.ingest != "long":
            continue
        build(case, args.top, args.threshold)


if __name__ == "__main__":
    main()
