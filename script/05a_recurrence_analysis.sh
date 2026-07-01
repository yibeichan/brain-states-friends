#!/bin/bash
#SBATCH --job-name=05a_recurrence
#SBATCH --output=logs/05a_recurrence_%A_%a.out
#SBATCH --error=logs/05a_recurrence_%A_%a.err
#SBATCH --time=00:05:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --array=0-5

# =============================================================================
# 05a - Brain State Recurrence Analysis
# =============================================================================
# Scores each brain state's recurrence across episodes (continuous gradient),
# from states active in nearly every episode to states appearing in only a few.
#
# Computes:
#   - Fractional occupancy per state per episode
#   - Recurrence score (fraction of episodes where state is active)
#   - Subject-specific recurring threshold from the recurrence distribution (default)
#   - Season specificity index
#   - Permutation test + FDR correction (Benjamini-Hochberg)
#
# Prerequisites:
#   - Step 04 completed in 'select' mode (decoded_states.pkl + final_results.json)
#
# Usage:
#   sbatch --export=SUB_ID=sub-01 script/05a_recurrence_analysis.sh
#   sbatch --export=SUB_ID=sub-01,PARCELLATION=atlas-4S456Parcels script/05a_recurrence_analysis.sh
#   bash 05a_recurrence_analysis.sh
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
VT="${VT:-0.95}"

# Print header
echo "=============================================="
echo "05a - Brain State Recurrence Analysis"
echo "=============================================="
echo "Subject: ${SUB_ID}"
echo "Parcellation: ${PARCELLATION}"
echo "=============================================="

# Run recurrence analysis
VT_ARG=""
if [ -n "${VT}" ]; then
    VT_ARG="--vt ${VT}"
fi

uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/05a_recurrence_analysis.py" \
    --sub_id "${SUB_ID}" \
    --parcellation "${PARCELLATION}" \
    ${VT_ARG}

echo "=============================================="
echo "Recurrence analysis complete!"
echo "=============================================="
