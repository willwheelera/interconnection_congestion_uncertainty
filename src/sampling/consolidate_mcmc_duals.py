"""Scatter the first N rounds of an MCMC sweep into a wide per-line dual-sample table.

The per-line tail figures read a dense per-line |mu| table -- a dual_cache.parquet whose
"mu" columns are the lines, built for the enumerated cases by sweeps.build_dual_cache.  A
sample_only case has no 2^ngen enumeration; its duals arrive from sampling.mcmc_sweep as a
sparse long table per round (round, chain, n_on, constraint, mu).  This selects the top
constraints by global max |mu| and scatters them into the same wide ("mu", constraint)
frame, one row per drawn configuration, so the shared `_line_stats` per-line statistics
apply unchanged to the sampled draws (with uniform weights 1 / nsamples).

Only the first `--rounds` round files are read, by name, not by globbing: a cluster run
leaves far more rounds than an estimate needs.  Unlike the enumerated dual_cache -- which
drops all-zero configs and recovers the config-space size from the maximum id -- every
sampled row is kept: the estimate weights the draws uniformly, so the row count must be
the sample count.  Rows are keyed by round*chains + chain, a sampled stand-in for the
int64 config id (which overflows past ngen 63).

Usage:
    uv run python -m src.sampling.consolidate_mcmc_duals --case activsg2000_v12_levfull \
        --law mrf_cobuild --rounds 40 --top 25
"""

import argparse

import numpy as np
import pandas as pd

from src.cases import registry
from src.sampling.consolidate_mcmc import round_parts
from src.sampling.problem import write_parquet_meta
from src.sweeps.build_dual_cache import constraint_global_max, select_constraints


def fill_table(parts, selected, chains):
    """Scatter the selected constraints' mu into a dense (nsamples x nselected) float32 table.

    Each drawn configuration is one row at round*chains + chain; a missing (round, chain)
    slot stays zero, so the row count is the full sample count the uniform weight needs.
    """
    col = {c: i for i, c in enumerate(selected)}
    d = pd.concat([pd.read_parquet(p, columns=["round", "chain", "constraint", "mu"])
                   for p in parts], ignore_index=True)
    d = d[d["constraint"].isin(set(selected))]
    table = np.zeros((chains * len(parts), len(selected)), dtype=np.float32)
    rows = (d["round"] * chains + d["chain"]).to_numpy()
    table[rows, d["constraint"].map(col).to_numpy()] = d["mu"].to_numpy()
    return table


def wide_dual_cache(table, selected):
    """Every sampled row as a wide frame with ("mu", constraint) MultiIndex columns."""
    df = pd.DataFrame(table, columns=pd.MultiIndex.from_product([["mu"], selected]))
    df.index.name = "id"
    return df


def build(case, law, rounds, top, threshold):
    """Build and write results/<case>/mcmc/<law>/dual_cache.parquet from the first N rounds."""
    parts = round_parts(case.results_dir / "mcmc" / law / "duals", rounds)
    cmax = constraint_global_max(parts)
    selected = select_constraints(cmax, top, threshold)
    chains = int(pd.read_parquet(parts[0], columns=["chain"])["chain"].max()) + 1
    df = wide_dual_cache(fill_table(parts, selected, chains), selected)
    meta = dict(case=case.name, law=law, rounds=rounds, chains=chains,
                nsamples=len(df), lines=list(map(str, selected)))
    dest = case.results_dir / "mcmc" / law / "dual_samples.parquet"
    write_parquet_meta(df, meta, dest)
    print(f"{case.name}/{law}  lines={len(selected)}  samples={len(df)}  -> {dest}")
    print("  " + ", ".join(f"{c}={cmax[c]:.0f}" for c in selected))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, help="a registered sample_only case name")
    parser.add_argument("--law", default="mrf_cobuild", help="a sampled nominal law name")
    parser.add_argument("--rounds", type=int, required=True,
                        help="number of round files to load (each round == chains configs)")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    build(registry.get(args.case), args.law, args.rounds, args.top, args.threshold)


if __name__ == "__main__":
    main()
