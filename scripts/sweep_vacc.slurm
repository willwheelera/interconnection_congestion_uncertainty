#!/usr/bin/env bash
# One shard of a congestion-dual sweep on UVM's VACC.
#
# Submitted by `src.sweeps.submit_cluster`, which checks the case's inputs, sizes the
# array, and chains the links -- everything this script would otherwise have to decide.
# It reads SWEEP_CASE and SWEEP_SHARDS from the submission; any further sweep options
# arrive as arguments to this script.
#
# The sweep skips chunks that are already cached, so a job killed at the walltime loses
# only the chunk in flight and the next link continues from there.

#SBATCH --job-name=amps-sweep
#SBATCH --partition=bluemoon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=12G
#SBATCH --time=30:00:00
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH --error=slurm/%x-%A_%a.out

set -euo pipefail
module load gurobi

# One SCDCOPF worker per load snapshot, sharing the core budget between them. numpy's
# BLAS must stay single-threaded or its nested pool oversubscribes the node on top.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
THREADS=$(( SLURM_CPUS_PER_TASK / 8 < 1 ? 1 : SLURM_CPUS_PER_TASK / 8 ))

cd "$SLURM_SUBMIT_DIR"

# --no-sync: the environment is built once on the login node by `uv sync`, so a compute
# node that cannot reach GitHub fails here in seconds rather than mid-sweep.
uv run --no-sync python -m src.sweeps.enumerate_configs \
    --case "$SWEEP_CASE" \
    --shard "$SLURM_ARRAY_TASK_ID/$SWEEP_SHARDS" \
    --threads "$THREADS" "$@"
