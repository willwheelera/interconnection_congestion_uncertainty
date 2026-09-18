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
    max_mu, sum_mu, one_mu, one_lmp, n_on, max_mu_single, driver
where one_lmp is NaN when a case has no cached LMP and max_mu_single is NaN when a
case has no per-snapshot enumeration.  The chosen line/bus/snapshot are recorded in
the file metadata.

`driver` names the element attaining max_mu.  It is here rather than in a separate
scalar cache because the Texas cases used to carry the same per-configuration reduction
three times over -- an `S_cache` (n_on, S, driver, driver_mu, where S and driver_mu were
the same column), a partial `config_cache` with that schema, and an `lmp_cache` that
repeated all four and added the LMP spread.  All of it is one groupby over mu_cache, so
it lives here now and `lmp_cache` keeps only the LMP columns it alone can produce.

Which ingest a case takes, where its raw caches are, and which element is its `one_mu`
all come off its registry record -- this file holds no case table of its own.

Usage:
    uv run python -m src.sweeps.build_target_cache --case case14_gen_per_bus
    uv run python -m src.sweeps.build_target_cache --case all
"""

import argparse
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.cases import registry
from src.sampling.problem import write_parquet_meta

COLUMNS = ["max_mu", "sum_mu", "one_mu", "one_lmp", "n_on", "max_mu_single", "driver"]


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
    out["driver"] = mu.idxmax(axis=1)                    # the element attaining max_mu
    meta = dict(kind="wide", one_mu_element=str(one_mu_col), one_lmp_element=str(one_lmp_col))
    return out[COLUMNS], meta


def reduce_mu_cache(mu_dir):
    """One pass over the long mu_cache: per-id max/sum/n_on/driver and per-constraint max.

    `driver` is the constraint attaining that configuration's max mu.  All of a
    configuration's duals live in one part file (parts are chunked by configuration), so
    the per-id argmax is exact within a part and needs no cross-part merge.
    """
    parts = sorted(glob.glob(str(Path(mu_dir) / "part_*.parquet")))
    per_id, cmax = [], defaultdict(float)
    for part in parts:
        d = pd.read_parquet(part)
        top = d.loc[d.groupby("id")["mu"].idxmax()].set_index("id")["constraint"]
        g = d.groupby("id")
        per_id.append(pd.DataFrame({"max_mu": g["mu"].max(), "sum_mu": g["mu"].sum(),
                                    "n_on": g["n_on"].first(), "driver": top}))
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
    """Peak mu per config for one chosen snapshot, from snapmax_cache; (series, snapshot).

    Returns (None, None) for a single-load-level case, whose snapmax rows carry a null
    snapshot: there the single-hour peak is the averaged peak, so `max_mu_single` would
    just repeat `max_mu`.  Also (None, None) for a sweep written before the snapmax
    schema was unified, whose parts have no snapshot or max_mu column at all.
    """
    parts = sorted(glob.glob(str(Path(snapmax_dir) / "*.parquet")))
    if not parts:
        return None, None
    head = pd.read_parquet(parts[0])
    if "snapshot" not in head or "max_mu" not in head or head["snapshot"].isna().all():
        return None, None
    snap = sorted(head["snapshot"].dropna().unique())[0]
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
    out["n_on"], out["driver"] = reduced["n_on"], reduced["driver"]
    peak, snap = single_snapshot_peak(snapmax) if snapmax else (None, None)
    out["max_mu_single"] = peak.reindex(reduced.index) if peak is not None else np.nan
    meta = dict(kind="long", one_mu_element=str(driver),
                one_lmp_element="lmp_maxabs" if lmp else None, snapshot=snap)
    return out[COLUMNS], meta


def build_case(case):
    """Build and write results/<case>/target_cache.parquet for one registered case."""
    if case.ingest == "wide":
        out, meta = build_wide(case.dual_cache, case.one_mu)
    else:
        out, meta = build_long(case.mu_cache, case.lmp_cache, case.snapmax_cache, case.one_mu)
    meta.update(case=case.name, ngen=int(out.index.max()).bit_length(), nrows=len(out))
    dest = case.target_cache
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_meta(out, meta, dest)
    print(f"{case.name:<24} rows={len(out)}  one_mu={meta['one_mu_element']}  "
          f"one_lmp={meta['one_lmp_element']}  -> {dest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="all", help="a case name, a name substring, or 'all'")
    args = parser.parse_args()
    for case in registry.get_cases(args.case):
        build_case(case)


if __name__ == "__main__":
    main()
