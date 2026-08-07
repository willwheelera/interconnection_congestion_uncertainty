"""Submit a congestion-dual sweep to the cluster, together with its preparation jobs.

Edit the settings below, then run it:

    uv run python -m src.sweeps.submit_cluster

Each case becomes a chain of Slurm jobs:

    select_relief_set -> screen_contingencies -> sweep x LINKS
                                                     -> build_target_cache -> audit_screening

A preparation step is submitted only when its output file is missing, and the sweep
waits on whichever steps were needed.  Both preparation steps depend on the queue, so a
new queue gets its own relief set and its own screened contingency set rather than
inheriting another case's.

The sweep skips chunks that are already cached, so a job stopped at the walltime is
continued by the next link, and a case whose enumeration is already complete gets only
the reduction and the audit.  The links are chained rather than run side by side: a
screened sweep rewrites its own contingency set as it discovers binders, and two jobs on
one case would race for that file.  Cores are filled by the worker pool within a job.

The chain still ends with the audit, but it no longer carries the exactness argument: the
sweep checks every solve against the dropped contingencies and grows its set when one is
violated, so re-solving 500 of the swept configurations runs that same check a second time
and must come back empty.  It is kept because it costs about a minute and it is the only
thing that would notice the growth loop silently not running.
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

# Walltimes.  Only the sweep is long; the preparation steps are minutes to a couple of
# hours, and an inflated request costs queue priority for nothing.  Counts below are the
# defaults of each module; per-solve costs were measured on the development machine with
# one Gurobi thread and the cluster ran faster, so these carry real margin.
SWEEP_TIME = "30:00:00"   # per link; must be under the partition's cap
# 806 configs x 8 snapshots = 6448 solves, serial in one process, against the full N-1 set
# (screening has not run yet) and mostly at full relief -- ~1.0 s each, so ~2 h.
RELIEF_TIME = "4:00:00"
# 34 stress configs x 8 snapshots = 272 solves against the full N-1 set, but spread over
# the whole node: ~7 solves per worker at ~0.8 s, plus a ~2 s model build.
SCREEN_TIME = "0:30:00"
# 500 configs x 8 snapshots = 4000 screened solves at ~80 ms including the check, spread
# over the node, plus ~4 s per worker to build the full-contingency PTDF and its columns.
AUDIT_TIME = "0:30:00"

AUDIT_TOP = 100           # audit the configurations of largest max_mu ...
AUDIT_RANDOM = 400        # ... and this many uniform draws standing for the rest

SCREENED = True           # solve the screened N-1 set, then audit it after the sweep
DRY_RUN = False           # True prints the sbatch commands instead of submitting

# TEST replaces the enumeration with one short job over a few hundred random
# configurations, written to results/<case>_random/.  Run it once on a new cluster:
# it exercises the module loads, the Gurobi license, and the solver in minutes.
TEST = False
TEST_CONFIGS = 200

# ---------------------------------------------------------------------------

SCRIPT = REPO_ROOT / "src" / "sweeps" / "vacc.slurm"

# Measured on activsg2000_original_v12, one worker per snapshot, one Gurobi thread, warm
# re-solves, on the development machine: 38 ms/config for the screened set plus the
# per-solve check on the contingencies screening dropped (26.4 solve, 11.8 check), 570 for
# the full 2756-contingency set.  The cluster ran the screened solve about twice as fast,
# so these overestimate; only used to print an estimate.
MS_PER_CONFIG = 38.0 if SCREENED else 570.0

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
    hours = (total - done) * CHUNK * MS_PER_CONFIG / 1000 / 3600 / workers_per_snapshot
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


def submit_audit(case, depend):
    """Reduce the sweep, then re-check the contingencies screening dropped.

    Chained on `afterany` so it still runs on whatever the last link cached: an audit of
    a partial sweep is still a statement about the configurations it covers.
    """
    depend = sbatch(f"target-{case.name}", "src.sweeps.build_target_cache", case,
                    "2:00:00", depend=depend, kind="afterany")
    sbatch(f"audit-{case.name}", "src.sweeps.audit_screening", case, AUDIT_TIME,
           depend=depend, args=["--top", str(AUDIT_TOP), "--random", str(AUDIT_RANDOM)])


def main():
    (REPO_ROOT / "slurm").mkdir(exist_ok=True)
    for name in CASES:
        case = registry.get(name)
        check_inputs(case)
        if TEST:
            print(f"  test job: {TEST_CONFIGS} random configs -> {case.name}_random")
            submit_sweep(case, prepare(case))
        elif remaining_chunks(case) == 0:
            print("  enumeration complete; submitting the audit only")
            submit_audit(case, None)
        else:
            submit_audit(case, submit_sweep(case, prepare(case)))


if __name__ == "__main__":
    main()
