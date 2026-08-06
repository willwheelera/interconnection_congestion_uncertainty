"""Solve a set of queue configurations and stream their duals to a resumable cache.

With `--mode full` this is the ground truth every estimate in the project is scored
against: all 2^ngen configurations solved once each, so the per-line tail probability
`P[mu_l > c]` is known exactly rather than estimated, and every sampler downstream runs
as a table lookup.  `--mode random` and `--mode leq_k` sweep subsets instead, for a
direct-Monte-Carlo check or the low-cardinality shell.

`--screened` solves the screened contingency set and stays exact by growing it: every
solve checks the contingencies screening dropped against its own dispatch, and a chunk
that turns any of them up is redone after the set is enlarged and written back to
`data/<case>_screened.json`.  What lands in the cache is therefore always the full
problem's solution, at the cost of the check -- measured on activsg2000_original_v12,
snapshot 2025-02-19 06:00: 26.7 ms/config to solve the 33-contingency screened set,
371.3 with the check, against 570.4 for the full 2756-contingency set outright.

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
One sweep job runs at a time per case: a second concurrent job would race it for the
screened file the augmenting loop rewrites.  Cores are filled by the worker pool, not by
splitting the configuration range across jobs.

Usage:
    uv run python -m src.sweeps.enumerate_configs --case case14_queue --workers 8
    uv run python -m src.sweeps.enumerate_configs --case activsg2000_2025 --screened
    uv run python -m src.sweeps.enumerate_configs --case case30_queue --mode random --n 20000
"""

import argparse
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.cases import registry
from src.cases.screen_contingencies import augment_screened
from src.model import evaluator

CHUNK = 2000            # configurations dispatched to the workers per part file
MAX_AUGMENT_ROUNDS = 10  # redos one chunk may take before the contingency set settles

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


def solve_chunk(case, target, chunk, make_rows, pw, tag):
    """Solve one chunk, growing the contingency set until nothing dropped is violated.

    Every solve checks the contingencies screening dropped against its own dispatch, and
    the parent unions those findings across workers as it drains the results -- so a
    contingency any worker needs is discovered once for all of them, not once each.  A
    non-empty union means this chunk was solved against an incomplete set: the set is
    enlarged on disk, the workers are rebuilt against it (a model's PTDF fixes its
    contingency set at construction), and the chunk is redone before anything is written.

    Only the discovering chunk is redone.  A chunk that came back clean is optimal for the
    full problem already, because the constraints it did not carry were satisfied at its
    optimum and adding satisfied constraints cannot move it.  The set grows monotonically
    within the case's full N-1 list, so discoveries taper and the loop terminates.
    """
    for _ in range(MAX_AUGMENT_ROUNDS):
        target.discovered.clear()
        results = target.evaluate_detailed(chunk.tolist())
        if not target.discovered:
            return make_rows(chunk, results, pw)
        found = sorted(target.discovered)
        n_kept = augment_screened(case, found, tag)
        print(f"  chunk {tag}: +{len(found)} contingencies {found} -> {n_kept} kept; "
              f"rebuilding workers and redoing the chunk", flush=True)
        target.restart()
    raise RuntimeError(f"{case.name}: chunk {tag} was still finding new contingencies after "
                       f"{MAX_AUGMENT_ROUNDS} augmentations")


def sweep(case, target, bits, out_dir):
    """Solve every configuration in chunks, writing one resumable part pair per chunk."""
    make_rows = snapshot_rows if case.parallel == "snapshot" else config_rows
    aux_columns = SNAPMAX_COLUMNS
    mu_dir, aux_dir = out_dir / "mu_cache", out_dir / "snapmax_cache"
    mu_dir.mkdir(parents=True, exist_ok=True)
    aux_dir.mkdir(parents=True, exist_ok=True)

    pw, t0, solved = powers(bits.shape[1]), time.time(), 0
    for start in range(0, len(bits), CHUNK):
        part = mu_dir / f"part_{start:08d}.parquet"
        aux_part = aux_dir / f"part_{start:08d}.parquet"
        if part.exists() and aux_part.exists():
            continue
        chunk = bits[start:start + CHUNK]
        rows, aux = solve_chunk(case, target, chunk, make_rows, pw, start)
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
    parser.add_argument("--workers", type=int, default=None,
                        help="core budget for the worker pool (default: the cores "
                             "Slurm allocated, or this machine's affinity)")
    parser.add_argument("--threads", type=int, default=None,
                        help="Gurobi threads per worker (default follows the parallel axis)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--screened", action="store_true",
                        help="solve the screened contingency set, growing it whenever a "
                             "solve violates a contingency screening dropped")
    args = parser.parse_args()

    case = registry.get(args.case)
    bits = config_bits(args.mode, case.ngen, args.n, case.build_p(), args.k, args.seed)
    out_dir = (case.results_dir if args.mode == "full"
               else Path(case.results_dir).with_name(f"{case.name}_{args.mode}"))

    print(f"{case.name}: {args.mode} sweep, {len(bits)} configs, {case.ngen} candidates, "
          f"{case.parallel}-parallel, "
          f"{'screened' if args.screened else 'full'} contingencies -> {out_dir}", flush=True)

    t0 = time.time()
    with evaluator.for_case(case, n_workers=args.workers, screened=args.screened,
                            threads=args.threads) as target:
        mu_dir = sweep(case, target, bits, out_dir)
    print(f"done in {(time.time() - t0) / 3600:.2f} h -> {mu_dir}")


if __name__ == "__main__":
    main()
