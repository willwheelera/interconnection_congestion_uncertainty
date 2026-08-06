"""Submit a congestion-dual sweep to the cluster, together with its preparation jobs.

Edit the settings below, then run it:

    uv run python -m src.sweeps.submit_cluster

Each case becomes a chain of Slurm jobs:

    select_relief_set  ->  screen_contingencies  ->  sweep (array of SHARDS) x LINKS

A preparation step is submitted only when its output file is missing, and the sweep
waits on whichever steps were needed.  Both preparation steps depend on the queue, so a
new queue gets its own relief set and its own screened contingency set rather than
inheriting another case's.

The sweep skips chunks that are already cached, so a job stopped at the walltime is
continued by the next link, and a case whose enumeration is complete is not submitted.
"""

import math
import subprocess
from pathlib import Path

from src.cases import registry
from src.paths import REPO_ROOT
from src.sweeps.enumerate_configs import CHUNK

# ----------------------------------------------------------------- settings

CASES = ["activsg2000_v12_lev0"]   # registered names; see src/cases/registry.py

PARTITION = "general"
# Cores per task.  The evaluator fills them with single-threaded workers: CPUS // 8
# replicas of the 8 snapshots.  More than one replica costs a little dual accuracy on
# non-peak constraints -- see the reproducibility note in src/README.md -- so set this
# to 8 if a run needs one worker per snapshot exactly.
CPUS = 40
SHARDS = 8                # array tasks in the sweep, splitting the enumeration
LINKS = 2                 # chained sweep jobs per case, each continuing the last

SWEEP_TIME = "30:00:00"   # per link; must be under the partition's cap
RELIEF_TIME = "12:00:00"  # select_relief_set solves ~6400 configs serially
SCREEN_TIME = "4:00:00"   # screen_contingencies solves 34 stress configs, full N-1 set

SCREENED = True           # solve the screened N-1 set, then audit it after the sweep
DRY_RUN = False           # True prints the sbatch commands instead of submitting

# TEST replaces the enumeration with one short job over a few hundred random
# configurations, written to results/<case>_random/.  Run it once on a new cluster:
# it exercises the module loads, the Gurobi license, and the solver in minutes.
TEST = False
TEST_CONFIGS = 200

# ---------------------------------------------------------------------------

SCRIPT = REPO_ROOT / "src" / "sweeps" / "vacc.slurm"

# Measured on ACTIVSg2000 at one worker per snapshot, one Gurobi thread, warm re-solves.
# Only used to print an estimate; a newly screened set may keep more contingencies than
# the 33 this was measured against.
MS_PER_CONFIG = 12.0 if SCREENED else 403.0

SWEEP_ARGS = ["--screened"] if SCREENED else []
if TEST:
    SHARDS, LINKS, SWEEP_TIME = 1, 1, "00:20:00"
    SWEEP_ARGS += ["--mode", "random", "--n", str(TEST_CONFIGS)]


def check_inputs(case):
    """Report what the case has, and fail before queueing if an input cannot be built."""
    for label, path in [("case", case.case_file), ("queue", case.candidates_file),
                        ("snapshots", case.snapshots_file)]:
        if path is None or not Path(path).exists():
            raise SystemExit(f"{case.name}: missing {label} file {path}")
    screen = (case.screened_file.name if case.screened_file.exists()
              else "to be screened" if SCREENED else "full N-1 set")
    print(f"{case.name}: queue {case.candidates_file.name} "
          f"| relief {case.relief_file.name if case.relief_file else 'to be selected'} "
          f"| contingencies {screen}")


def remaining_chunks(case):
    """Chunks of the enumeration not yet cached, printed with a cost estimate."""
    total = math.ceil((1 << case.ngen) / CHUNK)
    done = len(list(case.mu_cache.glob("part_*.parquet"))) if case.mu_cache.exists() else 0
    replicas = max(1, CPUS // len(case.snapshots()))
    hours = (total - done) * CHUNK * MS_PER_CONFIG / 1000 / 3600 / SHARDS / replicas
    print(f"  {done}/{total} chunks cached, {total - done} left across {SHARDS} shards "
          f"x {replicas} replicas -> ~{hours:.1f} h per shard")
    return total - done


def sbatch(name, module, case, walltime, depend=None, kind="afterok", array=None, args=()):
    """Submit one job of the chain and return its id, or a placeholder under DRY_RUN."""
    exports = f"ALL,JOB_MODULE={module},JOB_CASE={case.name}"
    argv = ["sbatch", "--parsable", f"--job-name={name}", f"--partition={PARTITION}",
            f"--cpus-per-task={CPUS}", f"--time={walltime}"]
    if array:
        argv.append(f"--array={array}")
        exports += f",SWEEP_SHARDS={SHARDS}"
    if depend:
        argv.append(f"--dependency={kind}:{depend}")
    argv += [f"--export={exports}", str(SCRIPT), *args]
    if DRY_RUN:
        print("  " + " ".join(argv))
        return f"<{name}>"
    job = subprocess.run(argv, cwd=REPO_ROOT, capture_output=True, text=True,
                         check=True).stdout.strip().split(";")[0]
    print(f"  {name}: job {job}")
    return job


def prepare(case):
    """Submit the missing preparation steps; return the job the sweep must wait on.

    Screening runs at the case's own recipe, which includes its relief set, so it waits
    on relief selection whenever that is also being built.
    """
    depend = None
    if case.relief_file is None:
        depend = sbatch(f"relief-{case.name}", "src.cases.texas.select_relief_set",
                        case, RELIEF_TIME)
    if SCREENED and not case.screened_file.exists():
        depend = sbatch(f"screen-{case.name}", "src.cases.screen_contingencies",
                        case, SCREEN_TIME, depend=depend)
    return depend


def submit_sweep(case, depend):
    """LINKS array jobs: the first waits on preparation, each on the one before it."""
    kind = "afterok"
    for _ in range(LINKS):
        depend = sbatch(f"sweep-{case.name}", "src.sweeps.enumerate_configs", case,
                        SWEEP_TIME, depend=depend, kind=kind,
                        array=f"0-{SHARDS - 1}", args=SWEEP_ARGS)
        kind = "afterany"


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
        submit_sweep(case, prepare(case))


if __name__ == "__main__":
    main()
