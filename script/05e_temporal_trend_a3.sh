#!/bin/bash
#SBATCH --job-name=05e_trend_a3
#SBATCH --output=logs/05e_trend_a3_%A_%a.out
#SBATCH --error=logs/05e_trend_a3_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --array=0-5

# =============================================================================
# 05e_temporal_trend_a3 - Within-Session FO Habituation (LME)
# =============================================================================
# Tests whether FO for each brain state trends systematically across runs
# within scanning sessions using a random-intercept linear mixed-effects model
# with permutation-based inference.
#
# See also: 05e_temporal_trend_a1.sh - cross-episode temporal trends
#           05e_temporal_trend_a2.sh - within-run temporal position
#
# Prerequisites:
#   - Step 04 completed in 'select' mode (decoded_states.pkl)
#   - Step 05a completed (fractional_occupancy.pkl, recurrence_summary.json)
#   - 00_get_scan completed (run_acquisition_times.csv)
#
# Usage:
#   sbatch --export=SUB_ID=sub-01 script/05e_temporal_trend_a3.sh
#   sbatch --export=SUB_ID=sub-01,VT=0.99 script/05e_temporal_trend_a3.sh
#   bash script/05e_temporal_trend_a3.sh
# =============================================================================

set -e

# Determine project directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
fi

mkdir -p "${PROJECT_DIR}/logs"

# Ensure the user-local uv install is on PATH (SLURM jobs may not inherit it)
export PATH="$HOME/.local/bin:$PATH"

# Configuration (can be overridden via --export)
subjects=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06)
SUB_ID="${SUB_ID:-${subjects[${SLURM_ARRAY_TASK_ID:-0}]}}"
PARCELLATION="${PARCELLATION:-atlas-4S156Parcels}"
N_PERM="${N_PERM:-5000}"
VT="${VT:-0.95}"

# Print header
echo "=============================================="
echo "05e_a3 - Within-Session FO Habituation (LME)"
echo "=============================================="
echo "Subject:       ${SUB_ID}"
echo "Parcellation:  ${PARCELLATION}"
echo "Permutations:  ${N_PERM}"
echo "VT:            ${VT}"
echo "=============================================="

VT_ARG=""
if [ -n "${VT}" ]; then
    VT_ARG="--vt ${VT}"
fi

# Primary run: exclude sub-HRF states (default)
echo "--- Primary run (exclude sub-HRF) ---"
uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/05e_temporal_trend_a3.py" \
    --sub_id "${SUB_ID}" \
    --parcellation "${PARCELLATION}" \
    --n_perm "${N_PERM}" \
    ${VT_ARG}

# Sensitivity run: include all states
echo "--- Sensitivity run (all states) ---"
uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/05e_temporal_trend_a3.py" \
    --sub_id "${SUB_ID}" \
    --parcellation "${PARCELLATION}" \
    --n_perm "${N_PERM}" \
    --no-exclude_sub_hrf \
    ${VT_ARG}

echo "=============================================="
echo "05e_a3 within-session habituation complete (both versions)!"
echo "=============================================="
