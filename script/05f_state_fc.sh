#!/bin/bash
#SBATCH --job-name=05f_state_fc
#SBATCH --output=logs/05f_state_fc_%A_%a.out
#SBATCH --error=logs/05f_state_fc_%A_%a.err
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --array=0-5

# Shared preamble: PROJECT_DIR/SCRIPT_DIR, uv on PATH, logs/ (utils/_env.sh)
source "${SLURM_SUBMIT_DIR:-.}/script/utils/_env.sh" 2>/dev/null || source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/script/utils/_env.sh" || exit 1

# =============================================================================
# 05f - Empirical Within-State Functional Connectivity
# =============================================================================
# Computes empirical state-conditioned FC from parcel timeseries using
# Ledoit-Wolf shrinkage covariance, delta_R (state - grand mean), and
# network-level aggregation.
#
# Prerequisites:
#   - 04_combined_hdphmm.py (mode: select) completed
#   - 02_extract_parcel_ts.py completed
#   - 05a_recurrence_analysis.py completed
#
# Usage:
#   sbatch --export=SUB_ID=sub-01 script/05f_state_fc.sh
#   sbatch --export=SUB_ID=sub-01,VT=0.99 script/05f_state_fc.sh
# =============================================================================

set -e


subjects=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06)
SUB_ID="${SUB_ID:-${subjects[${SLURM_ARRAY_TASK_ID:-0}]}}"
PARCELLATION="${PARCELLATION:-atlas-4S156Parcels}"
VT="${VT:-0.95}"

echo "=============================================="
echo "05f - Empirical State FC"
echo "=============================================="
echo "Subject: ${SUB_ID}"
echo "Parcellation: ${PARCELLATION}"
echo "=============================================="

VT_ARG=""
if [ -n "${VT}" ]; then
    VT_ARG="--vt ${VT}"
    echo "Variance threshold: ${VT}"
fi

uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/05f_state_fc.py" \
    --sub_id "${SUB_ID}" \
    --parcellation "${PARCELLATION}" \
    ${VT_ARG}

echo "=============================================="
echo "Empirical state FC complete!"
echo "=============================================="
