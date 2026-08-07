"""Submit a congestion-dual sweep to the cluster, together with its preparation jobs.

Edit the settings below, then run it:

    uv run python -m src.sweeps.submit_cluster

Each case becomes a short chain of Slurm jobs:

    prepare_case -> sweep x LINKS -> build_target_cache

`prepare_case` selects the relief bus set and then screens the contingencies, skipping
whichever already exists; it is one job because neither step is long.  Both depend on the
queue, so a new queue gets its own pair rather than inheriting another case's.

The sweep skips chunks that are already cached, so a job stopped at the walltime is
continued by the next link, and a case whose enumeration is already complete gets only the
reduction.  The links are chained rather than run side by side: a screened sweep rewrites
its own contingency set as it discovers binders, and two jobs on one case would race for
that file.  Cores are filled by the worker pool within a job.

There is no audit job.  The sweep checks every solve against the contingencies screening
dropped and grows its set whenever one is violated, so the cache is exact when it is
written rather than after something confirms it.
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
# Cores per task.  The evaluator fills them with single-threaded workers: CPUS // 8 on
# each of the 8 snapshot queues, dividing that snapshot's configurations between them.
# More than one worker per snapshot costs a little dual accuracy on non-peak constraints
# -- see the reproducibility note in src/README.md -- so set this to 8 to avoid it.
CPUS = 40
LINKS = 2                 # chained sweep jobs per case, each continuing the last

# Walltimes.  Only the sweep is long, and an inflated request costs queue priority for
# nothing.  Preparation is 840 configurations x the case's snapshots against the full N-1
# set at ~1 s a solve, spread over the node: minutes at 40 cores, and the hour is margin
# for a smaller allocation.
SWEEP_TIME = "30:00:00"   # per link; must be under the partition's cap
PREP_TIME = "1:00:00"     # relief selection and screening together

SCREENED = True           # solve the screened N-1 set, growing it as the sweep runs
DRY_RUN = False           # True prints the sbatch commands instead of submitting

# TEST replaces the enumeration with one short job over a few hundred random
# configurations, written to results/<case>_random/.  Run it once on a new cluster:
# it exercises the module loads, the Gurobi license, and the solver in minutes.
TEST = False
TEST_CONFIGS = 200

# ---------------------------------------------------------------------------

SCRIPT = REPO_ROOT / "src" / "sweeps" / "vacc.slurm"

# Wall ms per configuration with the node filled as below, measured on the cluster rather
# than derived from a single worker: 11 ms for activsg2000_v12_lev0 at 40 cores, screened.
#
# It is tied to CPUS and does not divide by the workers.  A solve is not all LP: the check
# on the contingencies screening dropped costs more than the LP it checks (8.7 ms against
# 11), and every part of it slows down as the node fills -- about 1.5x at one worker per
# physical core, from the all-core clock rather than from memory (see `model.dropped_flows`).
# A per-worker constant divided by the worker count therefore underestimates the sweep.
# Narrowing the check to single precision roughly halved it, so this now overestimates.
MS_PER_CONFIG = 11.0 if SCREENED else 570.0

SWEEP_ARGS = ["--screened"] if SCREENED else []
if TEST:
    LINKS, SWEEP_TIME = 1, "00:20:00"
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
    workers_per_snapshot = max(1, CPUS // len(case.snapshots()))
    hours = (total - done) * CHUNK * MS_PER_CONFIG / 1000 / 3600
    print(f"  {done}/{total} chunks cached, {total - done} left at "
          f"{workers_per_snapshot} workers per snapshot -> ~{hours:.1f} h "
          f"over {LINKS} links of {SWEEP_TIME}")
    return total - done


def sbatch(name, module, case, walltime, depend=None, kind="afterok", args=()):
    """Submit one job of the chain and return its id, or a placeholder under DRY_RUN."""
    exports = f"ALL,JOB_MODULE={module},JOB_CASE={case.name}"
    argv = ["sbatch", "--parsable", f"--job-name={name}", f"--partition={PARTITION}",
            f"--cpus-per-task={CPUS}", f"--time={walltime}"]
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
    """Submit the preparation job when the case is missing either input, else None.

    `prepare_case` decides for itself which of the two steps to run, so this only has to
    decide whether the job is needed at all.
    """
    if case.relief_file is not None and (case.screened_file.exists() or not SCREENED):
        return None
    return sbatch(f"prep-{case.name}", "src.cases.prepare_case", case, PREP_TIME)


def submit_sweep(case, depend):
    """LINKS sweep jobs: the first waits on preparation, each on the one before it.

    Chained, never concurrent: each link resumes the cached chunks the last one left, and
    a screened sweep rewrites the case's contingency set as it runs.
    """
    kind = "afterok"
    for _ in range(LINKS):
        depend = sbatch(f"sweep-{case.name}", "src.sweeps.enumerate_configs", case,
                        SWEEP_TIME, depend=depend, kind=kind, args=SWEEP_ARGS)
        kind = "afterany"
    return depend


def submit_target(case, depend):
    """Reduce the sweep's per-dual cache to the scalar target every sampler reads.

    Chained on `afterany` so it still runs on whatever the last link cached: the
    reduction of a partial sweep still describes the configurations it covers.
    """
    sbatch(f"target-{case.name}", "src.sweeps.build_target_cache", case,
           "2:00:00", depend=depend, kind="afterany")


def main():
    (REPO_ROOT / "slurm").mkdir(exist_ok=True)
    for name in CASES:
        case = registry.get(name)
        check_inputs(case)
        if TEST:
            print(f"  test job: {TEST_CONFIGS} random configs -> {case.name}_random")
            submit_sweep(case, prepare(case))
        elif remaining_chunks(case) == 0:
            print("  enumeration complete; submitting the reduction only")
            submit_target(case, None)
        else:
            submit_target(case, submit_sweep(case, prepare(case)))


if __name__ == "__main__":
    main()
