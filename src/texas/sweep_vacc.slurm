#!/usr/bin/env bash
# Run a full ACTIVSg2000 congestion-dual sweep on UVM's VACC.
#
# The sweep is resumable: enumerate_configs skips any chunk whose mu/snapmax part pair
# already exists. This script exploits that by chaining itself -- if the walltime runs
# out before the sweep finishes, it resubmits and the next job picks up where this one
# stopped, losing at most the one chunk that was in flight (~3 min of work).
#
# VERIFY BEFORE SUBMITTING (cluster-specific, confirm against current VACC docs):
#   --partition   partition name and its walltime limit
#   --account     your VACC allocation
#   module load   the module that provides uv (or drop it; see SETUP below)
#
# SETUP (once, on the cluster):
#   git clone <repo> && cd AMPS
#   uv sync                       # builds .venv from uv.lock
#   # Gurobi: see the license check below -- a node-locked license will NOT work here.
#
# Submit:
#   sbatch src/texas/sweep_vacc.slurm                       # default case (2025)
#   HARNESS_CASE=original_v12 sbatch --export=ALL src/texas/sweep_vacc.slurm

#SBATCH --job-name=amps-sweep
#SBATCH --partition=bluemoon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=30:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# One SCDCOPF worker process per load snapshot; the manager is idle while they solve, so
# the core budget is split among the workers. Threads are derived from the allocation
# rather than hardcoded, so changing --cpus-per-task alone re-tunes the job.
NSNAP=8
CPUS=${SLURM_CPUS_PER_TASK:-16}
export HARNESS_THREADS=$(( CPUS / NSNAP < 1 ? 1 : CPUS / NSNAP ))
export HARNESS_CASE=${HARNESS_CASE:-2025}

# numpy/BLAS must stay single-threaded: each worker already gets HARNESS_THREADS Gurobi
# threads, and a nested BLAS pool oversubscribes the node and dominates wall time.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm

CHAIN=${SWEEP_CHAIN:-1}       # which link of the resubmission chain this job is
CHAIN_MAX=${SWEEP_CHAIN_MAX:-6}
RESULTS="results/$(uv run python -c "
import sys; sys.path.insert(0, 'src/texas')
from sample_harness import RESULTS_NAME; print(RESULTS_NAME)")"
LOG="$RESULTS/sweep_full.log"

echo "=== amps sweep: case=$HARNESS_CASE chain=$CHAIN/$CHAIN_MAX job=${SLURM_JOB_ID:-none}"
echo "    node=$(hostname)  cpus=$CPUS  threads/worker=$HARNESS_THREADS  results=$RESULTS"

# Fail fast and loudly on licensing: Gurobi is required (the harness builds a
# gurobi_persistent solver with no fallback), and a node-locked license issued for a
# workstation will not validate on a compute node.
uv run python -c "
import gurobipy as gp
m = gp.Model()
m.setParam('OutputFlag', 0)
print('    gurobi', '.'.join(map(str, gp.gurobi.version())), 'license OK')
" || {
    echo "FATAL: Gurobi could not obtain a license on $(hostname)."
    echo "  A node-locked license (HOSTNAME=<workstation>) cannot be used here."
    echo "  Use a VACC-provided Gurobi module/token server, or an academic WLS license."
    exit 1
}

# The sweep prints this sentinel only after every chunk is written.
if [ -f "$LOG" ] && grep -q "full sweep of" "$LOG"; then
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
    uv run python src/texas/enumerate_configs.py --mode full 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e

NPARTS=$(ls "$RESULTS/mu_cache" 2>/dev/null | wc -l)

if [ "$rc" -eq 0 ] && grep -q "full sweep of" "$LOG"; then
    echo "=== sweep COMPLETE ($NPARTS chunks)"
    echo "    next: consolidate_s_cache -> reconstruct_lmp -> score_samplers"
elif [ "$rc" -ne 124 ]; then
    echo "=== sweep FAILED (exit $rc) at $NPARTS chunks; not resubmitting"
    exit "$rc"
elif [ "$CHAIN" -ge "$CHAIN_MAX" ]; then
    echo "=== incomplete after $CHAIN_MAX jobs ($NPARTS chunks); raise SWEEP_CHAIN_MAX"
    exit 1
else
    echo "=== walltime reached at $NPARTS chunks; resubmitting link $(( CHAIN + 1 ))"
    SWEEP_CHAIN=$(( CHAIN + 1 )) sbatch --export=ALL "$0"
fi
