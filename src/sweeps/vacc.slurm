#!/usr/bin/env bash
# One job of the pipeline on UVM's VACC: run the module named by JOB_MODULE against the
# case named by JOB_CASE.  Relief selection, contingency screening, and the sweep itself
# all go through here, differing only in the module and the resources requested.
#
# Submitted by `src.sweeps.submit_cluster`, whose settings block holds every decision:
# which cases, the partition, cores, walltimes, and the order the jobs depend on.  Every
# job is a single task on one node; the allocation is filled by the evaluator's worker
# pool, not by splitting the work across tasks.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.out

set -euo pipefail
module load gurobi

# The evaluator fills the allocation with single-threaded Gurobi workers, so numpy's
# BLAS must stay single-threaded too or the nested pools oversubscribe the node.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"

# --no-sync: the environment is built once on the login node by `uv sync`, so a compute
# node that cannot reach GitHub fails here in seconds rather than mid-job.
uv run --no-sync python -m "$JOB_MODULE" --case "$JOB_CASE" "$@"
