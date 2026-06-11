#!/bin/bash
#SBATCH --job-name=05e_trend_a2
#SBATCH --output=logs/05e_trend_a2_%A_%a.out
#SBATCH --error=logs/05e_trend_a2_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --array=0-5

# =============================================================================
# 05e_temporal_trend_a2 - Within-Run Temporal Position Analysis
# =============================================================================
# Detects whether brain states cluster at specific temporal positions within
# scanner runs, using the a/b suffix design to disentangle a-specific from
# shared run-onset effects.  Reports structural observations only.
#
# See also: 05e_temporal_trend_a1.sh — cross-episode temporal trends
#
# Prerequisites:
#   - Step 04 completed in 'select' mode (decoded_states.pkl)
#   - Step 05a completed (recurrence_summary.json)
#
# Usage:
#   sbatch --export=SUB_ID=sub-01 script/05e_temporal_trend_a2.sh
#   sbatch --export=SUB_ID=sub-01,PARCELLATION=atlas-4S456Parcels script/05e_temporal_trend_a2.sh
#   bash script/05e_temporal_trend_a2.sh
# =============================================================================

set -e

# Determine project directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
fi

mkdir -p "${PROJECT_DIR}/logs"

eval "$(micromamba shell hook --shell bash)"
micromamba activate friends-states

# Configuration (can be overridden via --export)
subjects=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06)
SUB_ID="${SUB_ID:-${subjects[${SLURM_ARRAY_TASK_ID:-0}]}}"
PARCELLATION="${PARCELLATION:-atlas-4S156Parcels}"
N_PERMUTATIONS="${N_PERMUTATIONS:-2000}"
VT="${VT:-0.95}"

# Print header
echo "=============================================="
echo "05e_a2 - Within-Run Temporal Position Analysis"
echo "=============================================="
echo "Subject:       ${SUB_ID}"
echo "Parcellation:  ${PARCELLATION}"
echo "Permutations:  ${N_PERMUTATIONS}"
echo "=============================================="

VT_ARG=""
if [ -n "${VT}" ]; then
    VT_ARG="--vt ${VT}"
fi

# Run position analysis (sub-HRF states excluded by default)
uv run --project "${PROJECT_DIR}" --extra gpu python "${PROJECT_DIR}/script/05e_temporal_trend_a2.py" \
    --sub_id "${SUB_ID}" \
    --parcellation "${PARCELLATION}" \
    --n_permutations "${N_PERMUTATIONS}" \
    ${VT_ARG}

echo "=============================================="
echo "05e_a2 temporal position analysis complete!"
echo "=============================================="
