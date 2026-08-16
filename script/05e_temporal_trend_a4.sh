#!/bin/bash
#SBATCH --job-name=05e_trend_a4
#SBATCH --output=logs/05e_trend_a4_%A_%a.out
#SBATCH --error=logs/05e_trend_a4_%A_%a.err
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --array=0-5

# Shared preamble: PROJECT_DIR/SCRIPT_DIR, uv on PATH, logs/ (utils/_env.sh)
source "${SLURM_SUBMIT_DIR:-.}/script/utils/_env.sh" 2>/dev/null || source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/script/utils/_env.sh" || exit 1

# =============================================================================
# 05e_temporal_trend_a4 - State Flag Synthesis
# =============================================================================
# Synthesizes per-state temporal metrics from a1, a2, a3 into boolean tags
# and a summary category for downstream filtering.
#
# Prerequisites:
#   - Step 05a completed (recurrence_scores.npy, eligible_states.json)
#   - Step 05e_a1 completed (temporal_trend_metrics.csv)
#   - Step 05e_a2 completed (temporal_position_metrics.csv)
#   - Step 05e_a3 completed (habituation_metrics.csv)
#
# Usage:
#   sbatch --export=SUB_ID=sub-01 script/05e_temporal_trend_a4.sh
#   sbatch --export=SUB_ID=sub-01,VT=0.99 script/05e_temporal_trend_a4.sh
#   bash script/05e_temporal_trend_a4.sh
# =============================================================================

set -e


# Configuration (can be overridden via --export)
subjects=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06)
SUB_ID="${SUB_ID:-${subjects[${SLURM_ARRAY_TASK_ID:-0}]}}"
PARCELLATION="${PARCELLATION:-atlas-4S156Parcels}"
VT="${VT:-0.95}"

# Print header
echo "=============================================="
echo "05e_a4 - State Flag Synthesis"
echo "=============================================="
echo "Subject:       ${SUB_ID}"
echo "Parcellation:  ${PARCELLATION}"
echo "VT:            ${VT}"
echo "=============================================="

VT_ARG=""
if [ -n "${VT}" ]; then
    VT_ARG="--vt ${VT}"
fi

uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/05e_temporal_trend_a4.py" \
    --sub_id "${SUB_ID}" \
    --parcellation "${PARCELLATION}" \
    ${VT_ARG}

echo "=============================================="
echo "05e_a4 state flag synthesis complete!"
echo "=============================================="
