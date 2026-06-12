#!/bin/bash
#SBATCH --job-name=05e_trend_a1
#SBATCH --output=logs/05e_trend_a1_%A_%a.out
#SBATCH --error=logs/05e_trend_a1_%A_%a.err
#SBATCH --time=00:45:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=1
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --array=0-5

# =============================================================================
# 05e_temporal_trend_a1 - Cross-Episode Temporal Trends
# =============================================================================
# Tests whether brain states exhibit systematic temporal trends at four
# hierarchical scales + two diagnostics:
#   Scale 1 (cross-season): Mann-Kendall tau (n=6, exploratory)
#   Scale 2 (within-season): permutation test on mean Spearman rho
#   Scale 2.5 (session habituation): run-level FO within BIDS sessions
#   Scale 3 (variance partition): semi-partial R^2 with permutation
#   Diag 1: motion confound check (FD trend + partial correlations)
#   Diag 2: anti-correlated state pair analysis
#
# See also: 05e_temporal_trend_a2.sh - within-run temporal position (theme song)
#
# Prerequisites:
#   - Step 04 completed (decoded_states.pkl, best_model.pkl)
#   - Step 05a completed (fractional_occupancy.pkl, per_season_mean_fo.json)
#
# Usage:
#   sbatch --export=SUB_ID=sub-01 script/05e_temporal_trend_a1.sh
#   sbatch --export=SUB_ID=sub-01,VT=0.99 script/05e_temporal_trend_a1.sh
#   bash script/05e_temporal_trend_a1.sh
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
SUB_ID_ORIG="${SUB_ID}"  # capture before default assignment
subjects=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06)
SUB_ID="${SUB_ID:-${subjects[${SLURM_ARRAY_TASK_ID:-0}]}}"
PARCELLATION="${PARCELLATION:-atlas-4S156Parcels}"
VT="${VT:-0.95}"

# If SUB_ID was explicitly set, only run array task 0 (others exit early)
if [ -n "$SLURM_JOB_ID" ] && [ -n "${SUB_ID_ORIG}" ] && [ "${SLURM_ARRAY_TASK_ID:-0}" -ne 0 ]; then
    echo "SUB_ID explicitly set to ${SUB_ID}; skipping array task ${SLURM_ARRAY_TASK_ID}"
    exit 0
fi

# Print header
echo "=============================================="
echo "05e_a1 - Cross-Episode Temporal Trends"
echo "=============================================="
echo "Subject:       ${SUB_ID}"
echo "Parcellation:  ${PARCELLATION}"
echo "VT:            ${VT}"
echo "=============================================="

VT_ARG=""
if [ -n "${VT}" ]; then
    VT_ARG="--vt ${VT}"
fi

uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/05e_temporal_trend_a1.py" \
    --sub_id "${SUB_ID}" \
    --parcellation "${PARCELLATION}" \
    ${VT_ARG}

echo "=============================================="
echo "05e_a1 temporal trend analysis complete!"
echo "=============================================="
