"""Sample a case's congestion-dual distribution under a coupled MRF, resumably, on the cluster.

The enumerated cases solve every one of their 2^ngen configurations and score samplers
against that exact truth (`sweeps.enumerate_configs`, `sampling.run_experiments`).  The full
leveraged queue (ngen ~ 195) has no such reference: 2^195 cannot be enumerated, so its
congestion-dual distribution is *estimated* by drawing configurations from the nominal law
and solving each one -- the whole reason this is a separate pipeline.

How a batch of `B` configurations is drawn depends on the law.  The block-uniform structured
laws (`mrf_cobuild`, `mrf_substitute`) are drawn as **exact iid** samples by the count collapse
of their within-block coupling (`structured_sampling.sample_structured`): at the coupling
`block_c` leaves, a fuel block has a metastable all-on mode a Gibbs sweep cannot cross, so
Gibbs would not reproduce the calibrated marginals.  Every other law is drawn by **Gibbs**
(`mrf.gibbs_sweep`): `bernoulli` (J = 0, so a sweep is already iid) and any dense law with no
count summary.  For Gibbs an ensemble of `B` chains is advanced `--thin` sweeps between solves;
for the exact laws a fresh batch of `B` iid draws is taken each round.  Either way the batch is
solved through the same worker pool and screened-contingency augmentation the enumeration uses
(`evaluator.for_case`, `enumerate_configs.solve_chunk`), so each recorded solve is exact for the
full N-1 problem.  Each law solves against its own screened set, copied from the prepared base
and grown independently (`per_law_case`), so the concurrent law jobs never share a mutable file.
Configurations are keyed by their **bit vector** (packed), not the MSB-first int64 id every
other cache uses -- that id overflows past ngen 63.

Resumable across jobs.  After each round the chain states, the RNG state, and the next round
index are written to `state.json`; a new job reloads them and continues the same chains, so a
run stopped at the walltime is picked up exactly where it left off.  Per-round part files are
named by their round and rewritten idempotently, so a job killed mid-round redoes only that
round on resume.

Output, under `results/<case>/mcmc/<law>/`:

    screened.json                  this law's own screened case, grown as it solves (--screened)
    contingencies.json             the kept-contingency record for that screened case
    state.json                     the checkpoint: X (B x ngen), RNG state, next round, meta
    duals/part_XXXXXXXX.parquet    round, chain, n_on, constraint, mu   (snapshot-averaged)
    snapmax/part_XXXXXXXX.parquet  round, chain, n_on, snapshot, max_mu, ... , load_shed_mw
    configs/part_XXXXXXXX.parquet  round, chain, n_on, bits            (np.packbits of the row)

Usage:
    uv run python -m src.sampling.mcmc_sweep --case activsg2000_v12_levfull --law mrf_cobuild
    uv run python -m src.sampling.mcmc_sweep --case activsg2000_v12_levfull --law bernoulli \
        --screened --chains 256 --thin 10 --max-hours 29
"""

import argparse
import dataclasses
import functools
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.cases import registry
from src.model import evaluator
from src.sampling.distributions import STRUCTURED_NAMES, case_dist
from src.sampling.mrf import gibbs_sweep, _sigmoid
from src.sampling.structured_sampling import sample_structured
from src.sweeps.enumerate_configs import solve_chunk

DUAL_COLUMNS = ["round", "chain", "n_on", "constraint", "mu"]
SNAP_COLUMNS = ["round", "chain", "n_on", "snapshot", "max_mu", "max_line",
                "max_overload_mw", "overload_line", "load_shed_mw"]
CFG_COLUMNS = ["round", "chain", "n_on", "bits"]


def make_rows(round_idx, chunk, results, _pw):
    """(dual, snapmax, config) rows for one solved batch, keyed by (round, chain).

    The snapshot-parallel counterpart of `enumerate_configs.snapshot_rows`, but a chain index
    and a packed bit vector stand in for the int64 config id (which overflows at this ngen).
    """
    averages, per_snapshot = results
    duals, snaps, cfgs = [], [], []
    for chain, (cfg, avg, snaps_r) in enumerate(zip(chunk, averages, per_snapshot)):
        n_on = int(np.sum(cfg))
        for con, mu in (avg or {"none": 0.0}).items():
            duals.append((round_idx, chain, n_on, con, mu))
        for r in snaps_r:
            snaps.append((round_idx, chain, n_on, r.snapshot, r.max_mu, r.max_line,
                          r.max_overload_mw, r.overload_line, r.load_shed_mw))
        cfgs.append((round_idx, chain, n_on, np.packbits(cfg).tobytes()))
    return duals, snaps, cfgs


def write_round(out_dir, round_idx, rows):
    """Write one round's three part files, named by the round so a redo overwrites in place."""
    duals, snaps, cfgs = rows
    for kind, data, cols in [("duals", duals, DUAL_COLUMNS), ("snapmax", snaps, SNAP_COLUMNS),
                             ("configs", cfgs, CFG_COLUMNS)]:
        d = out_dir / kind
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(data, columns=cols).to_parquet(d / f"part_{round_idx:08d}.parquet")


def save_state(out_dir, X, rng, next_round, meta):
    """Checkpoint the chain ensemble, the RNG, and the next round -- written last, atomically."""
    state = {"next_round": next_round, "X": X.tolist(),
             "rng": rng.bit_generator.state, **meta}
    tmp = out_dir / "state.json.tmp"
    tmp.write_text(json.dumps(state))
    tmp.replace(out_dir / "state.json")


def load_or_init(out_dir, ngen, chains, warmup, h, J, seed, iid):
    """Resume the sampler from a checkpoint, or start a fresh one.

    A resumed run restores the RNG and the next round exactly as the last job left them.  A
    fresh Gibbs start draws each chain from the independent marginals and burns in `warmup`
    sweeps so the first solved batch is already at the law's stationary distribution; a fresh
    exact (`iid`) start needs no chain state or warm-up -- each round draws a new batch -- so it
    returns a placeholder ensemble the round loop overwrites.
    """
    ckpt = out_dir / "state.json"
    if ckpt.exists():
        s = json.loads(ckpt.read_text())
        rng = np.random.default_rng()
        rng.bit_generator.state = s["rng"]
        print(f"resuming from round {s['next_round']} ({len(s['X'])} chains)", flush=True)
        return np.array(s["X"], dtype=np.int8), rng, s["next_round"]
    rng = np.random.default_rng(seed)
    if iid:
        print(f"fresh start: {chains} exact iid draws per round", flush=True)
        return np.zeros((chains, ngen), dtype=np.int8), rng, 0
    X = (rng.random((chains, ngen)) < _sigmoid(h)).astype(np.int8)
    for _ in range(warmup):
        gibbs_sweep(h, J, X, rng)
    print(f"fresh start: {chains} chains warmed {warmup} sweeps", flush=True)
    return X, rng, 0


def per_law_case(case, out_dir):
    """A copy of `case` whose screened set lives under `out_dir`, seeded from the shared base.

    Each law grows its own screened contingency set so the concurrent law jobs never write a
    shared file.  Seeded once from the prepared `data/<case>/screened.json`; a resumed link
    finds the grown copy already there and keeps it.
    """
    screened = out_dir / "screened.json"
    contingencies = out_dir / "contingencies.json"
    if not screened.exists():
        shutil.copyfile(case.screened_file, screened)
        shutil.copyfile(case.contingency_file, contingencies)
    return dataclasses.replace(
        case, paths={**case.paths, "screened": screened, "contingencies": contingencies})


def run(case, law_name, args):
    """Advance and solve rounds until the round cap or the wall-time budget, checkpointing each."""
    out_dir = case.results_dir / "mcmc" / law_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.screened:
        case = per_law_case(case, out_dir)
    mrf = case_dist(case, law_name)
    exact = law_name in STRUCTURED_NAMES
    groups = case.candidate_groups()
    X, rng, start = load_or_init(out_dir, case.ngen, args.chains, args.warmup,
                                 mrf.h, mrf.J, args.seed, exact)
    meta = {"case": case.name, "law": law_name, "chains": args.chains, "thin": args.thin,
            "warmup": args.warmup, "screened": args.screened}
    t0, stop = time.time(), start + args.rounds
    with evaluator.for_case(case, case.relief_buses(), screened=args.screened) as target:
        for rnd in range(start, stop):
            if exact:
                X = sample_structured(mrf.h, mrf.J, groups, args.chains, rng)
            else:
                for _ in range(args.thin):
                    gibbs_sweep(mrf.h, mrf.J, X, rng)
            rows = solve_chunk(case, target, X, functools.partial(make_rows, rnd), None, rnd)
            write_round(out_dir, rnd, rows)
            save_state(out_dir, X, rng, rnd + 1, meta)
            done = (rnd - start + 1) * args.chains
            print(f"round {rnd}: {done} configs this job, {len(rows[0])} duals, "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if args.max_hours and time.time() - t0 > args.max_hours * 3600:
                print(f"reached --max-hours {args.max_hours}; checkpointed at round {rnd + 1}")
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, help="a registered sample-only case name")
    parser.add_argument("--law", default="mrf_cobuild", help="a stored nominal law name")
    parser.add_argument("--chains", type=int, default=256, help="parallel Gibbs chains == batch")
    parser.add_argument("--thin", type=int, default=10, help="Gibbs sweeps between solves")
    parser.add_argument("--warmup", type=int, default=100, help="burn-in sweeps on a fresh start")
    parser.add_argument("--rounds", type=int, default=10 ** 9, help="solve rounds this job")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="self-stop after this many hours, checkpointed (for a clean exit "
                             "under the Slurm walltime)")
    parser.add_argument("--screened", action="store_true",
                        help="solve the screened N-1 set, growing it whenever a dropped "
                             "contingency is violated")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    case = registry.get(args.case)
    if not case.sample_only:
        raise SystemExit(f"{case.name} is not sample_only; use sweeps.enumerate_configs")
    run(case, args.law, args)


if __name__ == "__main__":
    main()
