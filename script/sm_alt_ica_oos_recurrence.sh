#!/bin/bash
#SBATCH --job-name=sm_alt_ica_oos_rec
#SBATCH --output=logs/sm_alt_ica_oos_rec_%A_%a.out
#SBATCH --error=logs/sm_alt_ica_oos_rec_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=ou_bcs_normal
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

echo "=============================================="
echo "sm_alt_ica_oos_recurrence - ICA OOS Recurrence"
echo "=============================================="
echo "Subject:       ${SUB_ID}"
echo "Stimuli:       ${STIMULI:-movie10 harrypotter petitprince}"
echo "n_null:        ${N_NULL:-1000}"
echo "=============================================="

# Run each stimulus independently. A clean skip (e.g. sub-04 has no HP/PP)
# exits 0 inside the Python; a genuine per-stimulus failure must NOT abort the
# sibling stimuli under `set -e`, so guard the call and remember a nonzero rc.
rc=0
for STIM in ${STIMULI:-movie10 harrypotter petitprince}; do
    echo "--- ${SUB_ID} / ${STIM} ---"
    if ! uv run --project "${PROJECT_DIR}" python "${PROJECT_DIR}/script/sm_alt_ica_oos_recurrence.py" \
        --sub_id "$SUB_ID" \
        --stimulus "$STIM" \
        --n_null "${N_NULL:-1000}"; then
        echo "WARNING: ${SUB_ID}/${STIM} failed; continuing with remaining stimuli" >&2
        rc=1
    fi
done

echo "=============================================="
echo "ICA OOS recurrence analysis complete (rc=${rc})!"
echo "=============================================="
exit $rc
