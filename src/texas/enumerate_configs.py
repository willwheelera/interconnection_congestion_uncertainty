"""Exhaustively solve a set of queue configurations and cache S per config id.

Drives configs through the live snapshot-pinned SCDCOPF workers in chunks and caches
the scalar congestion target S(x) = max over lines of load-averaged mu (plus the
driving constraint) for each config id. The same driver serves three subset modes and
the eventual full sweep:

  - random : n iid draws from the nominal Bernoulli law -- an unbiased direct Monte
             Carlo estimate of the tail down to ~1/n, for sanity-checking sample_tail,
  - leq_k  : every config with at most k candidates online (the low-cardinality shell),
  - full   : all 2^ngen configs (the exhaustive ground truth; hours of wall time).

Configs are de-duplicated by id, chunked so the worker queues never hold the whole set,
and the cache is written as a parquet keyed by config id. For the random mode it also
prints the direct exceedance curve and, if sample_tail's npz is present, the sampler's
estimate at matched thresholds.

The full sweep splits across jobs with --shard i/n: shard boundaries fall on CHUNK
boundaries, so each shard writes exactly the part files an unsharded run would, into the
same cache directory. Shards are disjoint and can run concurrently on separate nodes, and
a sharded and unsharded run can resume each other.

Usage:
    uv run python src/texas/enumerate_configs.py --mode random --n 30000
    uv run python src/texas/enumerate_configs.py --mode leq_k --k 5
    uv run python src/texas/enumerate_configs.py --mode full
    uv run python src/texas/enumerate_configs.py --mode full --shard 3/8
"""

import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "texas"))

from sample_harness import ParallelTarget, scalar_target, RESULTS_NAME
from sample_tail import BernoulliLaw

RESULTS = REPO_ROOT / "results" / RESULTS_NAME  # selected by HARNESS_CASE (default 2025)
CACHE = RESULTS / "config_cache.parquet"
MU_DIR = RESULTS / "mu_cache"  # full-sweep long-format mu parts
SNAPMAX_DIR = RESULTS / "snapmax_cache"  # per-snapshot peak mu + line
SAMPLER = RESULTS / "tail_sample.npz"
CHUNK = 2000  # configs sent to the workers per evaluate() call


def generate_configs(mode, n, k, ngen, law, rng):
    """Return the unique config bit-matrix for the requested subset mode."""
    if mode == "random":
        bits = law.draw(n, rng)
    elif mode == "leq_k":
        rows = [[1 if c in combo else 0 for c in range(ngen)]
                for j in range(k + 1) for combo in combinations(range(ngen), j)]
        bits = np.array(rows, dtype=np.int8)
    elif mode == "full":
        ids = np.arange(1 << ngen, dtype=np.int64)
        powers = (1 << np.arange(ngen - 1, -1, -1)).astype(np.int64)
        bits = ((ids[:, None] & powers[None, :]) != 0).astype(np.int8)
    powers = (1 << np.arange(ngen - 1, -1, -1)).astype(np.int64)
    _, keep = np.unique(bits @ powers, return_index=True)
    return bits[np.sort(keep)]


def solve_all(target, bits, powers):
    """Solve every config in chunks; return a DataFrame indexed by config id."""
    rows = []
    for start in range(0, len(bits), CHUNK):
        chunk = bits[start:start + CHUNK]
        for cfg, avg in zip(chunk, target.evaluate(chunk.tolist())):
            con, mu = max(avg.items(), key=lambda kv: kv[1], default=("none", 0.0))
            rows.append((int(cfg @ powers), int(cfg.sum()), scalar_target(avg), con, mu))
        done = start + len(chunk)
        print(f"  solved {done}/{len(bits)}", flush=True)
    df = pd.DataFrame(rows, columns=["id", "n_on", "S", "driver", "driver_mu"])
    return df.set_index("id")


def shard_bounds(n_configs, shard, n_shards):
    """Contiguous [lo, hi) config range for one shard, aligned to CHUNK boundaries.

    Splitting by whole chunks (rather than evenly by config) makes every shard write
    exactly the part files a single unsharded run would write, so sharded and unsharded
    runs are interchangeable: either can resume the other's output, and no two shards
    can name the same part.
    """
    n_chunks = -(-n_configs // CHUNK)
    lo = shard * n_chunks // n_shards
    hi = (shard + 1) * n_chunks // n_shards
    return lo * CHUNK, min(hi * CHUNK, n_configs)


def solve_full_mu(target, bits, powers, offset=0):
    """Solve every config and stream its full load-averaged mu vector to parquet parts.

    Writes one part file per CHUNK of configs in long format (id, n_on, constraint, mu),
    one row per binding dual (a config with none binding gets a single mu=0 sentinel row).
    In parallel writes a snapmax part (id, n_on, snapshot, max_mu, max_line,
    max_overload_mw, overload_line, load_shed_mw) holding each snapshot's single-hour peak
    mu and its line (before averaging -- the quantity to compare against real per-hour
    shadow prices), plus the worst thermal overload the cap allowed and the system-wide
    fallback shed: a large overload or nonzero shed flags a config the slimmed relief set
    could not fully serve, to recheck under full relief. A part pair that already exists
    is skipped, so the sweep resumes after an interruption.

    offset is this shard's position in the global config order; part files are named by
    the global index so shards write into one shared cache directory.
    """
    MU_DIR.mkdir(parents=True, exist_ok=True)
    SNAPMAX_DIR.mkdir(parents=True, exist_ok=True)
    n = len(bits)
    for start in range(0, n, CHUNK):
        gstart = offset + start
        part = MU_DIR / f"part_{gstart:08d}.parquet"
        snap_part = SNAPMAX_DIR / f"part_{gstart:08d}.parquet"
        if part.exists() and snap_part.exists():
            print(f"  skip {start}/{n} (exists)", flush=True)
            continue
        chunk = bits[start:start + CHUNK]
        averages, snap_max = target.evaluate_detailed(chunk.tolist())
        rows, srows = [], []
        for cfg, avg, sm in zip(chunk, averages, snap_max):
            cid, non = int(cfg @ powers), int(cfg.sum())
            for con, mu in (avg or {"none": 0.0}).items():
                rows.append((cid, non, con, mu))
            for sid, max_mu, max_line, ov_mw, ov_line, shed in sm:
                srows.append((cid, non, sid, max_mu, max_line, ov_mw, ov_line, shed))
        pd.DataFrame(rows, columns=["id", "n_on", "constraint", "mu"]).to_parquet(part)
        pd.DataFrame(srows, columns=["id", "n_on", "snapshot", "max_mu", "max_line",
                                     "max_overload_mw", "overload_line", "load_shed_mw"]).to_parquet(snap_part)
        print(f"  wrote {start + len(chunk)}/{n}  {part.name}  ({len(rows)} duals)", flush=True)


def direct_exceedance(S, thresholds):
    """Empirical P[S >= b] from an iid nominal sample (unbiased Monte Carlo)."""
    return (S[:, None] >= thresholds[None, :]).mean(axis=0)


def compare_to_sampler(S_direct):
    """Print the direct MC tail beside sample_tail's subset-simulation estimate."""
    thresholds = np.linspace(0.0, S_direct.max(), 12)
    direct = direct_exceedance(S_direct, thresholds)
    print("\n direct-MC vs subset-simulation exceedance")
    print(f"  {'b ($/MWh)':>10} {'P_direct':>10} {'P_sampler':>10}")
    if SAMPLER.exists():
        d = np.load(SAMPLER)
        sw, ss = d["sample_w"], d["sample_S"]
        samp = np.array([sw[ss >= b].sum() for b in thresholds])
    else:
        samp = np.full(len(thresholds), np.nan)
    for b, pd_, ps in zip(thresholds, direct, samp):
        print(f"  {b:10.1f} {pd_:10.2e} {ps:10.2e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["random", "leq_k", "full"], default="random")
    parser.add_argument("--n", type=int, default=30000, help="draws for --mode random")
    parser.add_argument("--k", type=int, default=5, help="max gens on for --mode leq_k")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--shard", default="0/1",
                        help="i/n: solve only shard i of n (--mode full); shards write "
                             "disjoint parts into one cache, so run them concurrently")
    args = parser.parse_args()

    shard, n_shards = (int(v) for v in args.shard.split("/"))
    assert 0 <= shard < n_shards, f"--shard {args.shard}: need 0 <= i < n"
    assert n_shards == 1 or args.mode == "full", "--shard only applies to --mode full"

    target = ParallelTarget()
    ngen = len(target.candidates)
    law = BernoulliLaw([c["p"] for c in target.candidates])
    powers = (1 << np.arange(ngen - 1, -1, -1)).astype(np.int64)
    rng = np.random.default_rng(args.seed)
    bits = generate_configs(args.mode, args.n, args.k, ngen, law, rng)

    target.start()
    try:
        print(f"mode={args.mode}  unique configs={len(bits)}  ngen={ngen}", flush=True)
        t0 = time.time()
        if args.mode == "full":
            lo, hi = shard_bounds(len(bits), shard, n_shards)
            print(f"shard {shard}/{n_shards}: configs [{lo}, {hi}) "
                  f"= {-(-(hi - lo) // CHUNK)} chunks", flush=True)
            solve_full_mu(target, bits[lo:hi], powers, offset=lo)
            print(f"\nfull sweep of {hi - lo} configs (shard {shard}/{n_shards}) in "
                  f"{time.time() - t0:.0f}s -> {MU_DIR}", flush=True)
        else:
            df = solve_all(target, bits, powers)
            dt = time.time() - t0
            df.to_parquet(CACHE)
            print(f"\nsolved {len(df)} configs in {dt:.0f}s ({dt / len(df):.3f}s each)")
            print(f"S range {df['S'].min():.1f}-{df['S'].max():.1f} $/MWh; "
                  f"top drivers:\n{df.nlargest(5, 'S')[['n_on', 'S', 'driver']]}")
            print(f"wrote {CACHE}")
            if args.mode == "random":
                compare_to_sampler(df["S"].to_numpy())
    finally:
        target.close()


if __name__ == "__main__":
    main()
