"""Exhaustively solve every queue configuration and stream the duals to a long mu cache.

This is the ground truth the paper's estimates are scored against: with `ncand` candidates
there are 2^ncand configurations, each solved once as an N-1 SCDCOPF, so the per-line tail
probability P[mu_l > c] is known exactly rather than estimated.  Every sampler downstream
then runs as a table lookup against this cache.

The sweep solves the FULL contingency set, not the screened one.  Screening is a union over
stress configurations and is not exact: on case14 one configuration in 62 binds a screened-out
contingency and comes out 2% low.  That is a fine trade for a diagnostic sweep and a bad one
for the ground truth every estimate is scored against, and the full set only costs about 12%
more on case30 and 2.3x on case118.  Pass --screened to trade exactness back for speed.

Output matches the Texas sweep's format, so tail_sampling.build_target_cache and
build_dual_cache ingest it unchanged: one parquet part per CHUNK of configurations, in long
format (id, n_on, constraint, mu), one row per binding dual, with a mu = 0 sentinel row for
a configuration where nothing binds.  Alongside it a stress part records each
configuration's worst thermal overload and its load shed, which is how a configuration that
outran the price cap or the nodal relief shows up.  Configuration ids are MSB-first over
the candidate order in the queue file.

Existing parts are skipped, so an interrupted sweep resumes where it stopped.

Usage:
    uv run python -m ieee_queue.enumerate_configs --case case14_queue
    uv run python -m ieee_queue.enumerate_configs --case case30_queue --workers 16
    uv run python -m ieee_queue.enumerate_configs --case case30_queue --mode random --n 20000
"""

import argparse
import time

import numpy as np
import pandas as pd

from ieee_queue.build_queue_case import NETWORKS
from ieee_queue.sample_harness import RESULTS, ParallelTarget, load_candidates

CHUNK = 2000            # configurations dispatched to the workers per part file


def config_bits(mode, ncand, n, probs, seed):
    """The configuration bit matrix for the requested mode, MSB-first, de-duplicated."""
    if mode == "full":
        ids = np.arange(1 << ncand, dtype=np.int64)
        bits = ((ids[:, None] & powers(ncand)[None, :]) != 0).astype(np.int8)
    else:
        rng = np.random.default_rng(seed)
        bits = (rng.random((n, ncand)) < probs[None, :]).astype(np.int8)
        _, keep = np.unique(bits @ powers(ncand), return_index=True)
        bits = bits[np.sort(keep)]
    return bits


def powers(ncand):
    """MSB-first bit weights, matching the id encoding the tail-sampling caches use."""
    return (1 << np.arange(ncand - 1, -1, -1)).astype(np.int64)


def part_rows(chunk, results, pw):
    """(long dual rows, stress rows) for one chunk of solved configurations."""
    rows, stress = [], []
    for cfg, (mus, overload, shed) in zip(chunk, results):
        cid, n_on = int(cfg @ pw), int(cfg.sum())
        for con, mu in (mus or {"none": 0.0}).items():
            rows.append((cid, n_on, con, mu))
        line, mw = max(overload.items(), key=lambda kv: kv[1], default=("none", 0.0))
        stress.append((cid, n_on, mw, line, shed))
    return rows, stress


def sweep(case, bits, workers, out_dir, screened):
    """Solve every configuration in chunks, writing one resumable part pair per chunk."""
    mu_dir, stress_dir = out_dir / "mu_cache", out_dir / "stress_cache"
    mu_dir.mkdir(parents=True, exist_ok=True)
    stress_dir.mkdir(parents=True, exist_ok=True)
    pw = powers(bits.shape[1])
    target = ParallelTarget(case, n_workers=workers, screened=screened)
    target.start()
    t0 = time.time()
    try:
        for start in range(0, len(bits), CHUNK):
            part = mu_dir / f"part_{start:08d}.parquet"
            stress_part = stress_dir / f"part_{start:08d}.parquet"
            if part.exists() and stress_part.exists():
                continue
            chunk = bits[start:start + CHUNK]
            rows, stress = part_rows(chunk, target.evaluate_detailed(chunk.tolist()), pw)
            pd.DataFrame(rows, columns=["id", "n_on", "constraint", "mu"]).to_parquet(part)
            pd.DataFrame(stress, columns=["id", "n_on", "max_overload_mw", "overload_line",
                                          "load_shed_mw"]).to_parquet(stress_part)
            done = start + len(chunk)
            rate = (time.time() - t0) / done
            print(f"  {done}/{len(bits)}  {len(rows)} duals  "
                  f"{rate * 1000:.0f} ms/config  eta {rate * (len(bits) - done) / 3600:.2f} h",
                  flush=True)
    finally:
        target.close()
    return mu_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="case14_queue", choices=list(NETWORKS))
    parser.add_argument("--mode", default="full", choices=["full", "random"])
    parser.add_argument("--n", type=int, default=20000, help="draws for --mode random")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--screened", action="store_true",
                        help="use the screened contingency set (faster, not exact)")
    args = parser.parse_args()

    candidates = load_candidates(args.case)
    probs = np.array([c["p"] for c in candidates])
    bits = config_bits(args.mode, len(candidates), args.n, probs, args.seed)
    out_dir = RESULTS / args.case if args.mode == "full" else RESULTS / f"{args.case}_random"

    print(f"{args.case}: {args.mode} sweep, {len(bits)} configs, {len(candidates)} "
          f"candidates, {args.workers} workers, "
          f"{'screened' if args.screened else 'full'} contingencies -> {out_dir}", flush=True)
    t0 = time.time()
    mu_dir = sweep(args.case, bits, args.workers, out_dir, args.screened)
    print(f"done in {(time.time() - t0) / 3600:.2f} h -> {mu_dir}")


if __name__ == "__main__":
    main()
