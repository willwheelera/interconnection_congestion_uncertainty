"""Solve a set of queue configurations and stream their duals to a resumable cache.

With `--mode full` this is the ground truth every estimate in the project is scored
against: all 2^ngen configurations solved once each, so the per-line tail probability
`P[mu_l > c]` is known exactly rather than estimated, and every sampler downstream runs
as a table lookup.  `--mode random` and `--mode leq_k` sweep subsets instead, for a
direct-Monte-Carlo check or the low-cardinality shell.

**The full sweep uses the full contingency set, not the screened one.**  Screening is a
union over stress configurations and is not exact -- on case14_queue one configuration in
62 binds a screened-out contingency and comes out 2% low.  That is a fine trade for a
diagnostic run and a bad one for ground truth, and the full set costs only about 12% more
on case30 and 2.3x on case118.  `--screened` trades exactness back for speed.

Output, identical for both case families so `sweeps.build_target_cache` and
`sweeps.build_dual_cache` ingest either unchanged:

    <out>/mu_cache/part_XXXXXXXX.parquet     id, n_on, constraint, mu   (one row per
        binding dual; a configuration where nothing binds gets one mu = 0 sentinel row)

plus the secondary cache that records what the recipe's relief mechanisms had to buy, in
one schema whatever the parallel axis:

    <out>/snapmax_cache/part_XXXXXXXX.parquet   id, n_on, snapshot, max_mu, max_line,
        max_overload_mw, overload_line, load_shed_mw

For a snapshot-parallel case there is one row per (configuration, snapshot), holding that
single hour's peak mu *before* averaging -- the quantity that compares against real
per-hour shadow prices.  A single-load-level case has one row per configuration with
`snapshot` null.

A large overload or a nonzero shed marks a configuration the shadow-price cap or the
slimmed relief set could not fully serve.  Configuration ids are MSB-first over the
candidate order in the queue file.

Parts that already exist are skipped, so an interrupted sweep resumes where it stopped.
`--shard i/n` splits a full sweep across jobs on chunk boundaries, so each shard writes
exactly the parts an unsharded run would, into the same cache directory: shards are
disjoint, can run concurrently on separate nodes, and can resume each other.

Usage:
    uv run python -m sweeps.enumerate_configs --case case14_queue --workers 8
    uv run python -m sweeps.enumerate_configs --case activsg2000_2025 --shard 3/8
    uv run python -m sweeps.enumerate_configs --case case30_queue --mode random --n 20000
"""

import argparse
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from cases import registry
from model import evaluator

CHUNK = 2000            # configurations dispatched to the workers per part file

MU_COLUMNS = ["id", "n_on", "constraint", "mu"]
SNAPMAX_COLUMNS = ["id", "n_on", "snapshot", "max_mu", "max_line",
                   "max_overload_mw", "overload_line", "load_shed_mw"]


def powers(ngen):
    """MSB-first bit weights -- the id encoding every cache in the project uses."""
    return (1 << np.arange(ngen - 1, -1, -1)).astype(np.int64)


def config_bits(mode, ngen, n, probs, k, seed):
    """The de-duplicated configuration bit matrix for the requested subset mode."""
    if mode == "full":
        ids = np.arange(1 << ngen, dtype=np.int64)
        return ((ids[:, None] & powers(ngen)[None, :]) != 0).astype(np.int8)
    if mode == "leq_k":
        bits = np.array([[1 if c in combo else 0 for c in range(ngen)]
                         for j in range(k + 1) for combo in combinations(range(ngen), j)],
                        dtype=np.int8)
    else:
        rng = np.random.default_rng(seed)
        bits = (rng.random((n, ngen)) < probs[None, :]).astype(np.int8)
    _, keep = np.unique(bits @ powers(ngen), return_index=True)
    return bits[np.sort(keep)]


def shard_bounds(n_configs, shard, n_shards):
    """Contiguous [lo, hi) config range for one shard, aligned to CHUNK boundaries.

    Splitting by whole chunks rather than evenly by configuration is what makes sharded
    and unsharded runs interchangeable: every shard writes exactly the part files a
    single run would, so either can resume the other and no two shards name the same part.
    """
    n_chunks = -(-n_configs // CHUNK)
    lo = shard * n_chunks // n_shards
    hi = (shard + 1) * n_chunks // n_shards
    return lo * CHUNK, min(hi * CHUNK, n_configs)


def snapshot_rows(chunk, results, pw):
    """(long dual rows, snapmax rows) for one chunk from a snapshot-parallel evaluator."""
    averages, per_snapshot = results
    rows, aux = [], []
    for cfg, avg, snaps in zip(chunk, averages, per_snapshot):
        cid, n_on = int(cfg @ pw), int(cfg.sum())
        for con, mu in (avg or {"none": 0.0}).items():
            rows.append((cid, n_on, con, mu))
        for sid, max_mu, max_line, ov_mw, ov_line, shed in snaps:
            aux.append((cid, n_on, sid, max_mu, max_line, ov_mw, ov_line, shed))
    return rows, aux


def config_rows(chunk, results, pw):
    """(long dual rows, snapmax rows) for one chunk from a config-parallel evaluator.

    Same snapmax schema as the snapshot-parallel path, with `snapshot` null: a
    single-load-level case has exactly one implicit snapshot."""
    rows, aux = [], []
    for cfg, (mus, overload, shed) in zip(chunk, results):
        cid, n_on = int(cfg @ pw), int(cfg.sum())
        for con, mu in (mus or {"none": 0.0}).items():
            rows.append((cid, n_on, con, mu))
        con, mu = max(mus.items(), key=lambda kv: kv[1], default=("none", 0.0))
        line, mw = max(overload.items(), key=lambda kv: kv[1], default=("none", 0.0))
        aux.append((cid, n_on, None, mu, con, mw, line, shed))
    return rows, aux


def sweep(case, target, bits, out_dir, offset=0):
    """Solve every configuration in chunks, writing one resumable part pair per chunk.

    `offset` is this shard's position in the global configuration order; parts are named
    by the global index so shards share one cache directory.
    """
    make_rows = snapshot_rows if case.parallel == "snapshot" else config_rows
    aux_columns = SNAPMAX_COLUMNS
    mu_dir, aux_dir = out_dir / "mu_cache", out_dir / "snapmax_cache"
    mu_dir.mkdir(parents=True, exist_ok=True)
    aux_dir.mkdir(parents=True, exist_ok=True)

    pw, t0, solved = powers(bits.shape[1]), time.time(), 0
    for start in range(0, len(bits), CHUNK):
        part = mu_dir / f"part_{offset + start:08d}.parquet"
        aux_part = aux_dir / f"part_{offset + start:08d}.parquet"
        if part.exists() and aux_part.exists():
            continue
        chunk = bits[start:start + CHUNK]
        rows, aux = make_rows(chunk, target.evaluate_detailed(chunk.tolist()), pw)
        pd.DataFrame(rows, columns=MU_COLUMNS).to_parquet(part)
        pd.DataFrame(aux, columns=aux_columns).to_parquet(aux_part)
        solved += len(chunk)
        rate = (time.time() - t0) / solved
        done = start + len(chunk)
        print(f"  {done}/{len(bits)}  {len(rows)} duals  {rate * 1000:.0f} ms/config  "
              f"eta {rate * (len(bits) - done) / 3600:.2f} h", flush=True)
    return mu_dir


def main():
    parser = argparse.ArgumentParser()
    registry.add_case_arg(parser)
    parser.add_argument("--mode", default="full", choices=["full", "random", "leq_k"])
    parser.add_argument("--n", type=int, default=20000, help="draws for --mode random")
    parser.add_argument("--k", type=int, default=5, help="max candidates on for --mode leq_k")
    parser.add_argument("--workers", type=int, default=8,
                        help="config-parallel worker count; snapshot-parallel cases pin "
                             "one worker per snapshot and ignore this")
    parser.add_argument("--threads", type=int, default=None,
                        help="Gurobi threads per worker (default follows the parallel axis)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--screened", action="store_true",
                        help="solve the screened contingency set (faster, not exact)")
    parser.add_argument("--shard", default="0/1",
                        help="i/n: solve only shard i of n (--mode full)")
    args = parser.parse_args()

    shard, n_shards = (int(v) for v in args.shard.split("/"))
    assert 0 <= shard < n_shards, f"--shard {args.shard}: need 0 <= i < n"
    assert n_shards == 1 or args.mode == "full", "--shard only applies to --mode full"

    case = registry.get(args.case)
    bits = config_bits(args.mode, case.ngen, args.n, case.build_p(), args.k, args.seed)
    lo, hi = shard_bounds(len(bits), shard, n_shards) if args.mode == "full" else (0, len(bits))
    out_dir = (case.results_dir if args.mode == "full"
               else Path(case.results_dir).with_name(f"{case.name}_{args.mode}"))

    print(f"{case.name}: {args.mode} sweep, {len(bits)} configs, {case.ngen} candidates, "
          f"{case.parallel}-parallel, "
          f"{'screened' if args.screened else 'full'} contingencies -> {out_dir}", flush=True)
    if n_shards > 1:
        print(f"shard {shard}/{n_shards}: configs [{lo}, {hi}) "
              f"= {-(-(hi - lo) // CHUNK)} chunks", flush=True)

    t0 = time.time()
    with evaluator.for_case(case, n_workers=args.workers, screened=args.screened,
                            threads=args.threads) as target:
        mu_dir = sweep(case, target, bits[lo:hi], out_dir, offset=lo)
    print(f"done in {(time.time() - t0) / 3600:.2f} h -> {mu_dir}")


if __name__ == "__main__":
    main()
