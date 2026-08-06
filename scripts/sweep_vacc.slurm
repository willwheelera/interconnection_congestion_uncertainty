#!/usr/bin/env bash
# Run a full ACTIVSg2000 congestion-dual sweep on UVM's VACC.
#
# The sweep is resumable: enumerate_configs skips any chunk whose mu/snapmax part pair
# already exists. This script exploits that by chaining itself -- if the walltime runs
# out before the sweep finishes, it resubmits and the next job picks up where this one
# stopped, losing at most the one chunk that was in flight.
#
# VERIFY BEFORE SUBMITTING (cluster-specific, confirm against current VACC docs):
#   --partition   partition name and its walltime limit
#   --account     your VACC allocation (add a #SBATCH --account line if VACC requires one)
#   uv on PATH    add the module load line your cluster needs, if any
#
# SETUP (once, on the cluster login node -- compute nodes may have no network):
#   git clone <repo> && cd AMPS
#   uv sync                       # builds .venv from uv.lock; fetches EGRET from GitHub
#   mkdir -p slurm                # Slurm opens the --output file before this script runs
#   # Gurobi: see the license check below -- a node-locked license will NOT work here.
#
# Submit:
#   sbatch scripts/sweep_vacc.slurm                                    # default case
#   SWEEP_CASE=activsg2000_v12_lev0 sbatch --export=ALL scripts/sweep_vacc.slurm
#
# COST (measured on one snapshot, one Gurobi thread, warm-started re-solves):
#   full contingency set (2756)     403 ms/config, 0.89 GB per worker  -> 14.7 h per shard
#   screened set (33 on v12)         12 ms/config, 0.24 GB per worker  ->  0.4 h per shard
# The default is the full set, which is what makes the sweep exact ground truth. A
# screened run needs a screened case built for that exact case name first, by
# cases.screen_contingencies -- the v12 set does not carry over to a different queue.
#
# --array splits the config range across independent jobs. Each task keeps the 8-worker
# snapshot-pinned layout, so this is the way to use more than ~10 cores: run more tasks,
# not bigger ones. Shards write disjoint parts into one cache directory, so they need no
# coordination and the derived caches are rebuilt once after all tasks finish.

#SBATCH --job-name=amps-sweep
#SBATCH --partition=bluemoon
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=12G
#SBATCH --time=30:00:00
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH --error=slurm/%x-%A_%a.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# One SCDCOPF worker process per load snapshot; the manager is idle while they solve, so
# the core budget is split among the workers. Threads are derived from the allocation
# rather than hardcoded, so changing --cpus-per-task alone re-tunes the job.
#
# 10 cores (1 thread/worker) rather than 16 (2 threads/worker): measured, a worker draws
# ~115% CPU of the 200% two threads allow, because warm-started dual simplex is
# sequential and extra threads only help through concurrent LP, which a warm start makes
# pointless. The second thread buys ~15% for double the reservation -- scale with more
# array shards instead.
NSNAP=8
CPUS=${SLURM_CPUS_PER_TASK:-16}
THREADS=$(( CPUS / NSNAP < 1 ? 1 : CPUS / NSNAP ))
CASE=${SWEEP_CASE:-activsg2000_2025}

# Empty for the exact full-set sweep; "--screened" for a fast approximate one.
SCREENED=""
[ "${SWEEP_SCREENED:-0}" = "1" ] && SCREENED="--screened"

# numpy/BLAS must stay single-threaded: each worker already gets THREADS Gurobi
# threads, and a nested BLAS pool oversubscribes the node and dominates wall time.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm

# Resolve the script by repo path rather than $0: Slurm runs a spool copy, and the
# resubmission below must name a file that still exists after this job ends.
SCRIPT=${SWEEP_SCRIPT:-scripts/sweep_vacc.slurm}

CHAIN=${SWEEP_CHAIN:-1}       # which link of the resubmission chain this job is
CHAIN_MAX=${SWEEP_CHAIN_MAX:-6}

# Shard count is taken from the array on first submission and then pinned, because a
# resubmission targets one task (--array=$SHARD) and would otherwise see a count of 1.
NSHARDS=${SWEEP_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}
SHARD=${SLURM_ARRAY_TASK_ID:-0}

RESULTS="results/$CASE"
LOG="$RESULTS/sweep_full.shard${SHARD}of${NSHARDS}.log"

echo "=== amps sweep: case=$CASE shard=$SHARD/$NSHARDS chain=$CHAIN/$CHAIN_MAX"
echo "    node=$(hostname)  cpus=$CPUS  threads/worker=$THREADS  contingencies=${SCREENED:-full}"
echo "    results=$RESULTS  script=$SCRIPT"

# --no-sync: the environment is built once on the login node, and a compute node that
# cannot reach GitHub would fail here rather than in the solve.
UV="uv run --no-sync"

# Fail fast and loudly on the two things that kill a job hours in rather than seconds in:
# a missing input file, and a Gurobi license that does not validate off the workstation
# (the harness builds a gurobi_persistent solver with no fallback).
SWEEP_CASE="$CASE" SWEEP_SCREENED="${SWEEP_SCREENED:-0}" $UV python -c "
import os, sys
import gurobipy as gp
from pathlib import Path
from src.cases.registry import get

case = get(os.environ['SWEEP_CASE'])
want_screened = os.environ['SWEEP_SCREENED'] == '1'
solve = case.solve_file(want_screened)
# solve_file falls back to the full case when no screened file exists, which would
# silently turn a requested screened run into a 34x slower one.
if want_screened and solve != case.screened_file:
    sys.exit(f'FATAL: --screened asked for but {case.screened_file} does not exist; '
             'run cases.screen_contingencies for this case first')
missing = [str(p) for p in [solve, case.candidates_file, case.snapshots_file]
           if p is None or not Path(p).exists()]
if missing:
    sys.exit('FATAL: missing case inputs: ' + ', '.join(missing))
m = gp.Model()
m.setParam('OutputFlag', 0)
print('    preflight OK: gurobi', '.'.join(map(str, gp.gurobi.version())),
      '| solving', solve.name, '| queue', case.candidates_file.name,
      '| relief', case.relief_file.name if case.relief_file else 'ALL LOAD BUSES')
" || {
    echo "FATAL: preflight failed on $(hostname). If it was the license:"
    echo "  a node-locked license (HOSTNAME=<workstation>) cannot be used here."
    echo "  Use a VACC-provided Gurobi module/token server, or an academic WLS license."
    exit 1
}

# The sweep prints this sentinel only after every chunk is written.
if [ -f "$LOG" ] && grep -q "^done in" "$LOG"; then
    echo "sweep already complete; nothing to do"
    exit 0
fi

mkdir -p "$RESULTS"

# Stop the sweep a little before the walltime rather than letting Slurm kill the job:
# a Slurm kill takes this script down with it, so the resubmission below would never
# run. Keep SWEEP_TIMEOUT under --time by enough to write the in-flight chunk and
# resubmit. timeout exits 124 when it fires, which is how a walltime stop is told
# apart from a genuine crash (a crash must not resubmit -- that loops forever).
set +e
timeout "${SWEEP_TIMEOUT:-29h}" \
    $UV python -m src.sweeps.enumerate_configs --case "$CASE" --mode full $SCREENED \
        --threads "$THREADS" --shard "$SHARD/$NSHARDS" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e

NPARTS=$(ls "$RESULTS/mu_cache" 2>/dev/null | wc -l || true)   # all shards, not just this one

if [ "$rc" -eq 0 ] && grep -q "^done in" "$LOG"; then
    echo "=== shard $SHARD COMPLETE ($NPARTS chunks cached across all shards)"
    echo "    once every shard is done, on the login node:"
    echo "      uv run python -m src.sweeps.build_target_cache --case $CASE"
    echo "      uv run python -m src.sweeps.build_dual_cache --case $CASE"
elif [ "$rc" -ne 124 ]; then
    echo "=== shard $SHARD FAILED (exit $rc) at $NPARTS chunks; not resubmitting"
    exit "$rc"
elif [ "$CHAIN" -ge "$CHAIN_MAX" ]; then
    echo "=== shard $SHARD incomplete after $CHAIN_MAX jobs; raise SWEEP_CHAIN_MAX"
    exit 1
else
    echo "=== walltime reached; resubmitting shard $SHARD as link $(( CHAIN + 1 ))"
    SWEEP_CHAIN=$(( CHAIN + 1 )) SWEEP_SHARDS=$NSHARDS \
        sbatch --export=ALL --array="$SHARD" "$SCRIPT"
fi
