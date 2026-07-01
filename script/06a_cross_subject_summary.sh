#!/bin/bash
#SBATCH --job-name=06a_cross_subj
#SBATCH --output=logs/06a_cross_subj_%j.out
#SBATCH --error=logs/06a_cross_subj_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=ou_bcs_normal,pi_satra

# =============================================================================
# 06a - Cross-Subject Summary
# =============================================================================
# Generates multi-panel summary figures aggregating 06a results across all
# 6 subjects. Run AFTER all per-subject 06a jobs complete.
#
# Usage:
#   sbatch script/06a_cross_subject_summary.sh
#   sbatch --dependency=afterok:<array_job_id> script/06a_cross_subject_summary.sh
# =============================================================================

set -e

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
fi

mkdir -p "${PROJECT_DIR}/logs"

# Ensure the user-local uv install is on PATH (SLURM jobs may not inherit it)
export PATH="$HOME/.local/bin:$PATH"

PARCELLATION="${PARCELLATION:-atlas-4S156Parcels}"
VT="${VT:-0.95}"

echo "=============================================="
echo "06a - Cross-Subject Summary"
echo "=============================================="
echo "Parcellation: ${PARCELLATION}"
echo "VT: ${VT}"
echo "=============================================="

VT_ARG=""
if [ -n "${VT}" ]; then
    VT_ARG="--vt ${VT}"
fi

uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/06a_state_temp_dynamics.py" \
    --mode cross_subject_summary \
    --parcellation "${PARCELLATION}" \
    ${VT_ARG}

echo "=============================================="
echo "Cross-subject summary complete!"
echo "=============================================="
