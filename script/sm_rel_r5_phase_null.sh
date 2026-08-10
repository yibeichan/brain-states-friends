#!/bin/bash
#SBATCH --job-name=sm_rel_r5_phase_null
#SBATCH --output=logs/sm_rel_r5_phase_null_%A_%a.out
#SBATCH --error=logs/sm_rel_r5_phase_null_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=ou_bcs_normal
#SBATCH --array=0-5

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
    PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
fi

mkdir -p "${PROJECT_DIR}/logs"

# The Viterbi recursion is a serial Python loop; multi-threaded BLAS only
# oversubscribes cores when the array runs six subjects side by side.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

SUBJECTS=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06)
SUB_ID="${SUB_ID:-${SUBJECTS[${SLURM_ARRAY_TASK_ID:-0}]}}"

echo "=============================================="
echo "sm_rel_r5_phase_null - R5 phase-randomized null"
echo "=============================================="
echo "Subject:  ${SUB_ID}"
echo "n_null:   ${N_NULL:-10000}"
echo "=============================================="

uv run --project "${PROJECT_DIR}" python "${PROJECT_DIR}/script/sm_rel_r5_phase_null.py" \
    --sub_id "$SUB_ID" \
    --n_null "${N_NULL:-10000}"

echo "=============================================="
echo "R5 phase-randomized null complete for ${SUB_ID}"
echo "=============================================="
