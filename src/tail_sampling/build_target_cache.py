"""Reduce every testbed to one common per-config target cache.

The rare-event runner needs each case in an identical format: one row per
generator configuration id, a fixed set of candidate *target* (importance)
functions as columns, and n_on.  Two enumerations feed this:

  - wide (the gen-per-bus cases): the generic dual_cache.parquet already holds
    per-line |mu| and per-bus LMP; max/sum over the mu group, and one chosen line / bus.
  - long (the security-constrained sweeps -- Texas and the IEEE queue cases): the
    2^ngen sweep is streamed to mu_cache/ in long format (id, n_on, constraint, mu);
    reduce per id.  The LMP spread comes from lmp_cache and the single-snapshot peak
    from snapmax_cache, both of which only the Texas sweeps carry.

Every case emits the same columns so the runner can swap cases by name alone:
    max_mu, sum_mu, one_mu, one_lmp, n_on, max_mu_single
where one_lmp is NaN when a case has no cached LMP and max_mu_single is NaN when a
case has no per-snapshot enumeration.  The chosen line/bus/snapshot are recorded in
the file metadata.

Usage:
    uv run python -m tail_sampling.build_target_cache --case case14_gen_per_bus
    uv run python -m tail_sampling.build_target_cache --case all
"""

import argparse
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from tail_sampling.problem_setup import REPO_ROOT, write_parquet_meta

RESULTS = REPO_ROOT / "results"
COLUMNS = ["max_mu", "sum_mu", "one_mu", "one_lmp", "n_on", "max_mu_single"]

# Each case: how to ingest it and where the common cache is written.  "wide" reads
# a dual_cache; "long" reads a long mu_cache (+ optional lmp_cache / snapmax_cache).
# "one_mu" names the element behind the single-element target; None picks the globally
# most-congested one.  case14 line 6 is set explicitly: it is the interesting estimation
# regime (P[mu > 100] ~ 5%), where the most-congested line is far rarer.  The IEEE queue
# cases leave one_mu unset until their sweeps exist and the element can be chosen from the
# enumerated tail.
CASES = {
    "case14_gen_per_bus": dict(kind="wide", dual=RESULTS / "case14_gen_per_bus" / "dual_cache.parquet",
                               one_mu="6"),
    "case30_gen_per_bus": dict(kind="wide", dual=RESULTS / "case30_gen_per_bus" / "dual_cache.parquet",
                               one_mu="27"),
    "activsg2000_original": dict(
        kind="long",
        mu_dir=RESULTS / "activsg2000" / "original_case_source" / "mu_cache",
        lmp=RESULTS / "activsg2000" / "original_case_source" / "lmp_cache_old.parquet",
        snapmax=None,
        one_mu="base:9",
    ),
    "activsg2000_2025": dict(
        kind="long",
        mu_dir=RESULTS / "activsg2000_2025" / "mu_cache",
        lmp=RESULTS / "activsg2000_2025" / "lmp_cache.parquet",
        snapmax=RESULTS / "activsg2000_2025" / "snapmax_cache",
        one_mu="ctg:464:461",
    ),
    "activsg2000_original_v12": dict(
        kind="long",
        mu_dir=RESULTS / "activsg2000_original_v12" / "mu_cache",
        lmp=None,
        snapmax=RESULTS / "activsg2000_original_v12" / "snapmax_cache",
        one_mu="ctg:68:18",
    ),
    "case14_queue": dict(kind="long", mu_dir=RESULTS / "case14_queue" / "mu_cache",
                         lmp=None, snapmax=None, one_mu="ctg:10:14"),
    "case30_queue": dict(kind="long", mu_dir=RESULTS / "case30_queue" / "mu_cache",
                         lmp=None, snapmax=None, one_mu="ctg:10:41"),
    "case118_queue": dict(kind="long", mu_dir=RESULTS / "case118_queue" / "mu_cache",
                          lmp=None, snapmax=None, one_mu=None),
}


def popcount(ids, ngen):
    """Number of on-generators per config id (MSB-first encoding, ngen bits)."""
    return ((ids[:, None] & (1 << np.arange(ngen))) != 0).sum(axis=1)


def build_wide(dual_path, one_mu=None):
    """Common targets from a wide dual_cache: max/sum |mu|, one line, one LMP bus."""
    df = pd.read_parquet(dual_path)
    df.index = df.index.astype(int)
    mu, lmp = df["mu"].abs(), df["lmp"]
    one_mu_col = one_mu if one_mu else mu.max(axis=0).idxmax()   # else most-congested line
    one_lmp_col = lmp.var(axis=0).idxmax()               # highest-variance bus LMP
    ngen = int(df.index.max()).bit_length()
    out = pd.DataFrame(index=df.index)
    out["max_mu"], out["sum_mu"] = mu.max(axis=1), mu.sum(axis=1)
    out["one_mu"], out["one_lmp"] = mu[one_mu_col], lmp[one_lmp_col]
    out["n_on"] = popcount(df.index.values, ngen)
    out["max_mu_single"] = np.nan                        # no per-snapshot enumeration
    meta = dict(kind="wide", one_mu_element=str(one_mu_col), one_lmp_element=str(one_lmp_col))
    return out[COLUMNS], meta


def reduce_mu_cache(mu_dir):
    """One pass over the long mu_cache: per-id max/sum/n_on and per-constraint global max."""
    parts = sorted(glob.glob(str(Path(mu_dir) / "part_*.parquet")))
    per_id, cmax = [], defaultdict(float)
    for part in parts:
        d = pd.read_parquet(part)
        g = d.groupby("id")
        per_id.append(pd.DataFrame({"max_mu": g["mu"].max(), "sum_mu": g["mu"].sum(),
                                    "n_on": g["n_on"].first()}))
        for c, v in d.groupby("constraint")["mu"].max().items():
            cmax[c] = max(cmax[c], v)
    reduced = pd.concat(per_id).sort_index()
    driver = max(cmax, key=cmax.get)                     # globally most-congested constraint
    return reduced, parts, driver


def extract_constraint(parts, constraint, index):
    """The driving constraint's mu per config id (0 where it does not bind)."""
    rows = [pd.read_parquet(p).query("constraint == @constraint").set_index("id")["mu"]
            for p in parts]
    return pd.concat(rows).reindex(index).fillna(0.0)


def single_snapshot_peak(snapmax_dir):
    """Peak mu per config for one chosen snapshot, from snapmax_cache; (series, snapshot)."""
    parts = sorted(glob.glob(str(Path(snapmax_dir) / "*.parquet")))
    snap = sorted(pd.read_parquet(parts[0])["snapshot"].unique())[0]
    rows = [pd.read_parquet(p).query("snapshot == @snap").set_index("id")["max_mu"] for p in parts]
    return pd.concat(rows).sort_index(), snap


def build_long(mu_dir, lmp, snapmax, one_mu=None):
    """Common targets from a long enumeration (+ optional LMP / snapshot caches)."""
    reduced, parts, driver = reduce_mu_cache(mu_dir)
    driver = one_mu or driver
    out = pd.DataFrame(index=reduced.index)
    out["max_mu"], out["sum_mu"] = reduced["max_mu"], reduced["sum_mu"]
    out["one_mu"] = extract_constraint(parts, driver, reduced.index)
    out["one_lmp"] = (pd.read_parquet(lmp)["lmp_maxabs"].reindex(reduced.index)
                      if lmp else np.nan)
    out["n_on"] = reduced["n_on"]
    snap = None
    if snapmax:
        peak, snap = single_snapshot_peak(snapmax)
        out["max_mu_single"] = peak.reindex(reduced.index)
    else:
        out["max_mu_single"] = np.nan
    meta = dict(kind="long", one_mu_element=str(driver),
                one_lmp_element="lmp_maxabs" if lmp else None, snapshot=snap)
    return out[COLUMNS], meta


def build_case(name):
    """Build and write results/<name>/target_cache.parquet for one registered case."""
    spec = CASES[name]
    if spec["kind"] == "wide":
        out, meta = build_wide(spec["dual"], spec["one_mu"])
    else:
        out, meta = build_long(spec["mu_dir"], spec["lmp"], spec["snapmax"], spec["one_mu"])
    meta.update(case=name, ngen=int(out.index.max()).bit_length(), nrows=len(out))
    dest = RESULTS / name / "target_cache.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_meta(out, meta, dest)
    print(f"{name:<22} rows={len(out)}  one_mu={meta['one_mu_element']}  "
          f"one_lmp={meta['one_lmp_element']}  -> {dest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="all", choices=[*CASES, "all"])
    args = parser.parse_args()
    for name in (CASES if args.case == "all" else [args.case]):
        build_case(name)


if __name__ == "__main__":
    main()
