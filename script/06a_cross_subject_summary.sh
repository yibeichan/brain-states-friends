#!/bin/bash
#SBATCH --job-name=06a_cross_subj
#SBATCH --output=logs/06a_cross_subj_%j.out
#SBATCH --error=logs/06a_cross_subj_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=ou_bcs_normal,pi_satra

# Shared preamble: PROJECT_DIR/SCRIPT_DIR, uv on PATH, logs/ (utils/_env.sh)
_ENV="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)/script/utils/_env.sh"
[ -f "$_ENV" ] || _ENV="${SLURM_SUBMIT_DIR:-.}/script/utils/_env.sh"
source "$_ENV" || { echo "ERROR: cannot locate script/utils/_env.sh — submit from the repo root" >&2; exit 1; }

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
