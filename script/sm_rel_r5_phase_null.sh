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

# uv needs a writable cache; the default below works on a fresh clone. Set
# UV_CACHE_DIR in the submitting shell to relocate it (never hardcode a site path here).
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
if ! (mkdir -p "$UV_CACHE_DIR" && touch "$UV_CACHE_DIR/.writable" && rm -f "$UV_CACHE_DIR/.writable"); then
    echo "UV_CACHE_DIR=$UV_CACHE_DIR is not writable; export a writable path before submitting" >&2
    exit 1
fi

# The Viterbi recursion is a serial Python loop; multi-threaded BLAS only
# oversubscribes cores when the array runs six subjects side by side.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

SUBJECTS=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06)
SUB_ID="${SUB_ID:-${SUBJECTS[${SLURM_ARRAY_TASK_ID:-0}]}}"
STIMULUS="${STIMULUS:-movie10}"
PER_GROUP_FLAG=""
if [ "${PER_GROUP:-0}" = "1" ]; then PER_GROUP_FLAG="--per_group"; fi

echo "=============================================="
echo "sm_rel_r5_phase_null - phase-randomized null"
echo "=============================================="
echo "Subject:  ${SUB_ID}"
echo "Stimulus: ${STIMULUS}"
echo "n_null:   ${N_NULL:-10000}"
echo "per_group: ${PER_GROUP:-0}"
echo "=============================================="
uv run --project "${PROJECT_DIR}" python "${PROJECT_DIR}/script/sm_rel_r5_phase_null.py" \
    --sub_id "$SUB_ID" \
    --stimulus "$STIMULUS" \
    --n_null "${N_NULL:-10000}" \
    $PER_GROUP_FLAG
echo "=============================================="
echo "phase-randomized null complete for ${SUB_ID}/${STIMULUS}"
echo "=============================================="
