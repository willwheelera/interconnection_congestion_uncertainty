"""Submit a congestion-dual sweep to the cluster.

Edit the settings below, then run it:

    uv run python -m src.sweeps.submit_cluster

Each case goes out as an array of SHARDS tasks, submitted LINKS times with each link
waiting on the one before it.  The sweep skips chunks that are already cached, so a
job stopped at the walltime is continued by the next link, and a case with nothing
left to solve is reported and not submitted.
"""

import math
import subprocess

from src.cases import registry
from src.paths import REPO_ROOT
from src.sweeps.enumerate_configs import CHUNK

# ----------------------------------------------------------------- settings

CASES = ["activsg2000_v12_lev0"]   # registered names; see src/cases/registry.py

SHARDS = 8                # array tasks per job, splitting the enumeration evenly
LINKS = 2                 # chained jobs per case, each continuing the last
WALLTIME = "30:00:00"     # per job; must be under the partition's cap

SCREENED = False          # True solves the screened N-1 set: ~34x faster, not exact
DRY_RUN = False           # True prints the sbatch commands instead of submitting

# TEST replaces the enumeration with one short job over a few hundred random
# configurations, written to results/<case>_random/.  Run it once on a new cluster:
# it exercises the module loads, the Gurobi license, and the solver in minutes.
TEST = False
TEST_CONFIGS = 200

# ---------------------------------------------------------------------------

SCRIPT = REPO_ROOT / "scripts" / "sweep_vacc.slurm"

# Measured on ACTIVSg2000: one snapshot per worker, one Gurobi thread, warm
# re-solves.  Only used to print an estimate at submission time.
MS_PER_CONFIG = 12.0 if SCREENED else 403.0

SWEEP_ARGS = ["--screened"] if SCREENED else []
if TEST:
    SHARDS, LINKS, WALLTIME = 1, 1, "00:20:00"
    SWEEP_ARGS += ["--mode", "random", "--n", str(TEST_CONFIGS)]


def check_inputs(case):
    """Fail on the login node, before queueing, if the case cannot be solved."""
    solve = case.solve_file(SCREENED)
    if SCREENED and solve != case.screened_file:
        raise SystemExit(f"{case.name}: SCREENED needs {case.screened_file.name}; "
                         "build it with src.cases.screen_contingencies first")
    for label, path in [("case", solve), ("queue", case.candidates_file),
                        ("snapshots", case.snapshots_file)]:
        if path is None or not path.exists():
            raise SystemExit(f"{case.name}: missing {label} file {path}")
    print(f"{case.name}: solving {solve.name} | queue {case.candidates_file.name} "
          f"| relief {case.relief_file.name if case.relief_file else 'ALL LOAD BUSES'}")


def remaining_chunks(case):
    """Chunks of the enumeration not yet cached, printed with a cost estimate."""
    total = math.ceil((1 << case.ngen) / CHUNK)
    done = len(list(case.mu_cache.glob("part_*.parquet"))) if case.mu_cache.exists() else 0
    hours = (total - done) * CHUNK * MS_PER_CONFIG / 1000 / 3600 / SHARDS
    print(f"  {done}/{total} chunks cached, {total - done} left across {SHARDS} "
          f"shards -> ~{hours:.1f} h per shard")
    return total - done


def submit(case):
    """Submit this case's chain: LINKS array jobs, each waiting on the one before."""
    job_id = None
    for link in range(LINKS):
        argv = ["sbatch", "--parsable",
                f"--job-name=sweep-{case.name}",
                f"--array=0-{SHARDS - 1}",
                f"--time={WALLTIME}",
                f"--export=ALL,SWEEP_CASE={case.name},SWEEP_SHARDS={SHARDS}"]
        if job_id:
            argv.append(f"--dependency=afterany:{job_id}")
        argv += [str(SCRIPT)] + SWEEP_ARGS
        if DRY_RUN:
            print("  " + " ".join(argv))
            job_id = f"<link{link}>"
        else:
            job_id = subprocess.run(argv, cwd=REPO_ROOT, capture_output=True,
                                    text=True, check=True).stdout.strip().split(";")[0]
            print(f"  link {link + 1}/{LINKS}: job {job_id}")


def main():
    (REPO_ROOT / "slurm").mkdir(exist_ok=True)
    for name in CASES:
        case = registry.get(name)
        check_inputs(case)
        if TEST:
            print(f"  test job: {TEST_CONFIGS} random configs -> {case.name}_random")
        elif remaining_chunks(case) == 0:
            print("  nothing left to solve; skipping")
            continue
        submit(case)


if __name__ == "__main__":
    main()
