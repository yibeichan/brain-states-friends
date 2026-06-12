#!/bin/bash
#SBATCH --job-name=sm_ica_states
#SBATCH --output=logs/sm_ica_states_%A_%a.out
#SBATCH --error=logs/sm_ica_states_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --array=0-5

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
    PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
fi

mkdir -p "${PROJECT_DIR}/logs"

# Single-threaded BLAS so FastICA restarts don't oversubscribe cores
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

SUBJECTS=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06)
SUB_ID="${SUB_ID:-${SUBJECTS[${SLURM_ARRAY_TASK_ID:-0}]}}"
PARCELLATION="${PARCELLATION:-atlas-4S156Parcels}"
VT="${VT:-0.95}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate friends-states

echo "=============================================="
echo "sm_ica_states - ICA Supplementary Analysis"
echo "=============================================="
echo "Subject:       ${SUB_ID}"
echo "Parcellation:  ${PARCELLATION}"
echo "VT:            ${VT}"
echo "=============================================="

# EXTRA_K accepts colon- or comma-separated K(s). Colons are translated to
# commas here so callers can pass them via `sbatch --export=ALL,EXTRA_K=41:42:..`
# without SLURM comma-splitting the --export list into separate variables.
# Default to empty first so the substitution is safe under `set -u` when EXTRA_K
# is unset (a normal base-sweep run passes no EXTRA_K).
EXTRA_K="${EXTRA_K:-}"
EXTRA_K_ARG="${EXTRA_K//:/,}"

uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/sm_ica_states.py" \
    --sub_id "$SUB_ID" \
    --parcellation "$PARCELLATION" \
    --vt "$VT" \
    ${EXTRA_K:+--extra_k "$EXTRA_K_ARG"} \
    ${FORCE:+--force}

echo "=============================================="
echo "ICA supplementary analysis complete!"
echo "=============================================="
