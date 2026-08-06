"""Submit the cluster jobs for one or more sweeps, and chain their continuations.

Everything a submission has to decide lives here rather than in the batch script:
whether the case's inputs exist, how much of its enumeration is already cached, how
many shards to run, and how the links are chained.  `scripts/sweep_vacc.slurm` is left
as a plain runner that solves one shard.

Continuation is Slurm's own `--dependency=afterany`, not a resubmission from inside the
job.  The sweep skips chunks that already exist, so each link picks up where the last
one stopped and a link with nothing left to do exits in seconds.  Two links cover a full
ACTIVSg2000 sweep at eight shards; raise `--links` if the partition's walltime is short.

Run `--test` first on a new cluster: one short job over a few hundred random
configurations, written to `results/<case>_random/`, which exercises the module loads,
the Gurobi license, and the solver path in minutes instead of hours.

Usage:
    uv run python -m src.sweeps.submit_cluster --case activsg2000_v12_lev0 --test
    uv run python -m src.sweeps.submit_cluster --case activsg2000_v12_lev0
    uv run python -m src.sweeps.submit_cluster --case activsg2000_v12_lev0 activsg2000_v12_lev1
    uv run python -m src.sweeps.submit_cluster --case activsg2000_2025 --screened --dry-run
"""

import argparse
import math
import subprocess
from pathlib import Path

from src.cases import registry
from src.paths import REPO_ROOT
from src.sweeps.enumerate_configs import CHUNK

SCRIPT = REPO_ROOT / "scripts" / "sweep_vacc.slurm"
TEST_CONFIGS = 200

# Measured on ACTIVSg2000, one snapshot per worker, one Gurobi thread, warm re-solves.
# Used only to report an estimate at submission time.
MS_PER_CONFIG = {False: 403.0, True: 12.0}   # keyed by whether the set is screened


def check_inputs(case, screened):
    """Fail on the login node, before queueing, if the case cannot be solved."""
    solve = case.solve_file(screened)
    if screened and solve != case.screened_file:
        raise SystemExit(f"{case.name}: --screened needs {case.screened_file.name}; "
                         "build it with cases.screen_contingencies first")
    for label, path in [("case", solve), ("queue", case.candidates_file),
                        ("snapshots", case.snapshots_file)]:
        if path is None or not Path(path).exists():
            raise SystemExit(f"{case.name}: missing {label} file {path}")


def progress(case):
    """(chunks already cached, chunks in a full enumeration) for this case."""
    total = math.ceil((1 << case.ngen) / CHUNK)
    done = len(list(case.mu_cache.glob("part_*.parquet"))) if case.mu_cache.exists() else 0
    return done, total


def print_inputs(case, screened):
    """The three files that decide what a sweep actually solves."""
    print(f"{case.name}: solving {case.solve_file(screened).name} "
          f"| queue {case.candidates_file.name} "
          f"| relief {case.relief_file.name if case.relief_file else 'ALL LOAD BUSES'}")


def print_work(case, screened, shards):
    """How much of the enumeration is left, and roughly how long a shard will take."""
    done, total = progress(case)
    hours = (total - done) * CHUNK * MS_PER_CONFIG[screened] / 1000 / 3600 / shards
    print(f"  {done}/{total} chunks cached, {total - done} left across {shards} shards "
          f"-> ~{hours:.1f} h per shard")
    return total - done


def sbatch_argv(case, shards, sweep_args, depend=None, time=None):
    """The sbatch command for one link of one case's chain.

    Sweep options ride as trailing arguments to the batch script rather than inside
    --export, which is a comma-separated list and a poor place for a quoted string.
    """
    exports = f"ALL,SWEEP_CASE={case.name},SWEEP_SHARDS={shards}"
    argv = ["sbatch", "--parsable", f"--job-name=sweep-{case.name}",
            f"--array=0-{shards - 1}", f"--export={exports}"]
    if depend:
        argv.append(f"--dependency=afterany:{depend}")
    if time:
        argv.append(f"--time={time}")
    return argv + [str(SCRIPT)] + sweep_args


def submit_chain(case, shards, links, sweep_args, time, dry_run):
    """Submit `links` array jobs, each waiting on the one before it."""
    job_id = None
    for link in range(links):
        argv = sbatch_argv(case, shards, sweep_args, depend=job_id, time=time)
        if dry_run:
            print("  " + " ".join(argv))
            job_id = f"<link{link}>"
            continue
        job_id = subprocess.run(argv, cwd=REPO_ROOT, capture_output=True, text=True,
                                check=True).stdout.strip().split(";")[0]
        print(f"  link {link + 1}/{links}: job {job_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", nargs="+", required=True, choices=list(registry.CASES),
                        help="one or more registered cases to sweep")
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--links", type=int, default=2,
                        help="chained jobs per case; each continues the one before it")
    parser.add_argument("--screened", action="store_true",
                        help="solve the screened contingency set (faster, not exact)")
    parser.add_argument("--test", action="store_true",
                        help=f"one short job over {TEST_CONFIGS} random configurations")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    shards, links = (1, 1) if args.test else (args.shards, args.links)
    sweep_args = ["--mode", "random", "--n", str(TEST_CONFIGS)] if args.test else []
    sweep_args += ["--screened"] if args.screened else []
    (REPO_ROOT / "slurm").mkdir(exist_ok=True)

    for name in args.case:
        case = registry.get(name)
        check_inputs(case, args.screened)
        print_inputs(case, args.screened)
        if args.test:
            print(f"  test job: {TEST_CONFIGS} random configs -> {case.name}_random")
        elif print_work(case, args.screened, shards) == 0:
            print("  nothing left to solve; skipping")
            continue
        submit_chain(case, shards, links, sweep_args,
                     "00:20:00" if args.test else None, args.dry_run)


if __name__ == "__main__":
    main()
