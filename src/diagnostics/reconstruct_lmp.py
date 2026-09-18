"""Reconstruct congestion LMPs from the cached duals via PTDF --- no new solves.

In DC power flow the LMP at a bus is  lambda - sum_l mu_l * PTDF[l, bus]  (energy
component lambda, a uniform shift, plus the congestion component set by the line
duals mu and the PTDF matrix). PTDF is pure topology (independent of load and of
which queue generators are on), so the load-averaged congestion LMP is exactly
- sum_l (avg mu_l) * PTDF[l, .] using the load-averaged mu already cached in
the case's `mu_cache/`. This reads those duals, projects them through PTDF
(contingency duals through the post-outage PTDF row, which folds in the LODF), and
writes per-config congestion-LMP spread stats, then relates them to S = max_l mu_l.

Only the congestion component is reconstructed: the energy component lambda is a
uniform per-bus offset (it cancels out of any spatial spread) and is not in the mu
cache. All spread statistics (range, std, max abs) are invariant to it.

Usage:
    uv run python -m src.diagnostics.reconstruct_lmp --case activsg2000_2025
"""

import argparse
import glob
import json

import numpy as np
import pandas as pd
from egret.data.model_data import ModelData

from src.cases import registry
from src.model.scdcopf import build_audit_ptdf

CHUNK = 20000


def build_ptdf(case):
    """VirtualPTDFMatrix on the full topology + a fixed bus ordering (ref bus last)."""
    md = ModelData(json.loads(case.case_file.read_text()))
    ptdf = build_audit_ptdf(md, case.case_file)
    return ptdf, list(ptdf.buses_keys)


def constraint_rows(ptdf, constraints, bus_order):
    """Dense PTDF row (length n_bus, reference bus = 0) for each binding constraint id."""
    idx = {b: j for j, b in enumerate(bus_order)}
    P = np.zeros((len(constraints), len(bus_order)))
    for i, name in enumerate(constraints):
        parts = name.split(":")
        if parts[0] == "base":
            it = ptdf.get_branch_ptdf_iterator(parts[1])
        else:
            it = ptdf.get_contingency_branch_ptdf_iterator(parts[1], parts[2])
        for bus, val in it:
            P[i, idx[bus]] = val
    return P


def load_mu_matrix(case, constraints):
    """Dense (nconf, n_constraint) load-averaged mu; column j is constraints[j]."""
    col = {c: j for j, c in enumerate(constraints)}
    nconf = 1 << case.ngen
    M = np.zeros((nconf, len(constraints)), dtype=np.float32)
    for part in sorted(glob.glob(str(case.mu_cache / "part_*.parquet"))):
        df = pd.read_parquet(part)
        keep = df["constraint"].isin(col)
        M[df.loc[keep, "id"].values,
          df.loc[keep, "constraint"].map(col).values] = df.loc[keep, "mu"].values
    return M


def lmp_spread_stats(M, P):
    """Per-config congestion-LMP spread: range (max-min), std, max abs, over buses."""
    n = M.shape[0]
    rng = np.empty(n); std = np.empty(n); mab = np.empty(n)
    for s in range(0, n, CHUNK):
        C = M[s:s + CHUNK] @ P                       # (chunk, n_bus) congestion LMP (sign-free stats)
        rng[s:s + CHUNK] = C.max(1) - C.min(1)
        std[s:s + CHUNK] = C.std(1)
        mab[s:s + CHUNK] = np.abs(C).max(1)
    return rng, std, mab


def unique_constraints(case):
    """All constraint ids that appear anywhere in the mu cache."""
    seen = set()
    for part in sorted(glob.glob(str(case.mu_cache / "part_*.parquet"))):
        seen.update(pd.read_parquet(part, columns=["constraint"])["constraint"].unique())
    seen.discard("none")
    return sorted(seen)


def report(df):
    """Print LMP-spread distribution and its relationship to S = max mu."""
    print(f"\ncongestion-LMP spread across {len(df)} configs ($/MWh, spatial range max-min):")
    for q in (0.5, 0.9, 0.99, 0.999, 1.0):
        print(f"  q{q:<6} range {df.lmp_range.quantile(q):8.2f}   maxabs {df.lmp_maxabs.quantile(q):8.2f}")
    print(f"\ncorr(S, lmp_range)   = {df.S.corr(df.lmp_range):.3f}")
    print(f"corr(S, lmp_maxabs)  = {df.S.corr(df.lmp_maxabs):.3f}")
    top_s = set(df.nlargest(1000, "S").index)
    top_l = set(df.nlargest(1000, "lmp_range").index)
    print(f"\ntop-1000 by S vs top-1000 by lmp_range: overlap {len(top_s & top_l)}/1000")
    print("\nlmp_range by S-driver line (mean range | count) for the main drivers:")
    g = df.groupby("driver").agg(n=("S", "size"), mean_range=("lmp_range", "mean"),
                                 mean_S=("S", "mean")).sort_values("n", ascending=False)
    print(g.head(8).to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, help="a registered case name")
    case = registry.get(parser.parse_args().case)
    out = case.results_dir / "lmp_cache.parquet"

    constraints = unique_constraints(case)
    print(f"{len(constraints)} binding constraints across the cache")
    ptdf, bus_order = build_ptdf(case)
    P = constraint_rows(ptdf, constraints, bus_order)
    M = load_mu_matrix(case, constraints)
    rng, std, mab = lmp_spread_stats(M, P)

    # The LMP columns are all this file can produce; max_mu / driver come from the
    # target_cache, and are joined here only for the report, not written back.
    lmp = pd.DataFrame({"lmp_range": rng, "lmp_std": std, "lmp_maxabs": mab},
                       index=np.arange(len(rng)))
    lmp.index.name = "id"
    lmp.to_parquet(out)
    tgt = pd.read_parquet(case.target_cache, columns=["max_mu", "n_on", "driver"])
    report(lmp.join(tgt).rename(columns={"max_mu": "S"}))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
