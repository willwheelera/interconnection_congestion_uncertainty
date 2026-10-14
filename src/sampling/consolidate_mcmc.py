"""Reduce the first N rounds of an MCMC sweep to one per-sample mu table.

`sampling.mcmc_sweep` writes per-round part files under results/<case>/mcmc/<law>/
(duals/, snapmax/, configs/), each round holding `chains` solved configurations.  A
sample_only case has no 2^ngen reference, so its congestion-dual distribution is
*estimated* from these draws.  This reduces the sampled duals to one row per drawn
configuration -- n_on, max_mu, sum_mu, the driving constraint, and the worst
single-snapshot peak -- the sampled analogue of sweeps.build_target_cache's per-id frame.

Only the first `--rounds` round files are read, by name (part_00000000 .. part_{N-1:08d}),
not by globbing the directory: a cluster run leaves far more rounds on disk than an
estimate needs.  N counts the round files (each round == `chains` configurations).

Usage:
    uv run python -m src.sampling.consolidate_mcmc --case activsg2000_v12_levfull \
        --law mrf_cobuild --rounds 40
"""

import argparse

import pandas as pd

from src.cases import registry
from src.sampling.problem import write_parquet_meta

COLUMNS = ["round", "chain", "n_on", "max_mu", "sum_mu", "max_mu_single", "driver"]


def round_parts(part_dir, rounds):
    """The first `rounds` round part files, named by round index (no globbing)."""
    return [part_dir / f"part_{i:08d}.parquet" for i in range(rounds)]


def reduce_duals(duals_dir, rounds):
    """Per-sample max/sum mu, n_on, and driving constraint over the first `rounds` rounds.

    Keyed by (round, chain): the packed-bit config id is not loaded, so each drawn
    configuration is identified by the round and chain that produced it.
    """
    d = pd.concat([pd.read_parquet(p) for p in round_parts(duals_dir, rounds)],
                  ignore_index=True)
    key = ["round", "chain"]
    driver = d.loc[d.groupby(key)["mu"].idxmax()].set_index(key)["constraint"]
    g = d.groupby(key)
    out = pd.DataFrame({"n_on": g["n_on"].first(), "max_mu": g["mu"].max(),
                        "sum_mu": g["mu"].sum(), "driver": driver})
    return out.reset_index()


def single_snapshot_peak(snapmax_dir, rounds):
    """Per-sample peak mu over snapshots, from the first `rounds` snapmax round files.

    The duals mu is snapshot-averaged; this is the worst single hour's peak, the tail
    quantity that averaging hides.
    """
    s = pd.concat([pd.read_parquet(p) for p in round_parts(snapmax_dir, rounds)],
                  ignore_index=True)
    return s.groupby(["round", "chain"])["max_mu"].max().rename("max_mu_single")


def consolidate(case, law, rounds):
    """Reduce the first `rounds` rounds of one case/law sweep to its per-sample cache."""
    mcmc_dir = case.results_dir / "mcmc" / law
    out = reduce_duals(mcmc_dir / "duals", rounds)
    peak = single_snapshot_peak(mcmc_dir / "snapmax", rounds)
    out = out.merge(peak, on=["round", "chain"], how="left")[COLUMNS]
    driver = out.loc[out["max_mu"].idxmax(), "driver"]
    meta = dict(case=case.name, law=law, rounds=rounds,
                chains=int(out["chain"].max()) + 1, nsamples=len(out), driver=driver)
    dest = mcmc_dir / "target_samples.parquet"
    write_parquet_meta(out, meta, dest)
    print(f"{case.name}/{law}  rounds={rounds}  samples={len(out)}  "
          f"driver={driver}  -> {dest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, help="a registered sample_only case name")
    parser.add_argument("--law", default="mrf_cobuild", help="a sampled nominal law name")
    parser.add_argument("--rounds", type=int, required=True,
                        help="number of round files to load (each round == chains configs)")
    args = parser.parse_args()
    consolidate(registry.get(args.case), args.law, args.rounds)


if __name__ == "__main__":
    main()
