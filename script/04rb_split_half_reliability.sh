#!/bin/bash
#SBATCH --job-name=04rb_split_half
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

# Shared preamble: PROJECT_DIR/SCRIPT_DIR, uv on PATH, logs/ (utils/_env.sh)
source "${SLURM_SUBMIT_DIR:-.}/script/utils/_env.sh" 2>/dev/null || source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/script/utils/_env.sh" || exit 1

# =============================================================================
# Split-Half Reliability Analysis - SLURM Wrapper
# =============================================================================
#
# Compares structural invariants between two independently-fit HMM halves.
#
# Prerequisites:
#   - 04 split_half_fit completed for both halves (A and B)
#
# Usage:
#   sbatch --export=SUBJECT_ID=sub-01 script/04rb_split_half_reliability.sh
#
# =============================================================================


SUBJECT_ID=${SUBJECT_ID:-"sub-01"}
PARCELLATION=${PARCELLATION:-"atlas-4S156Parcels"}

echo "=========================================="
echo "04rb Split-Half Reliability"
echo "=========================================="
echo "Subject:      $SUBJECT_ID"
echo "Parcellation: $PARCELLATION"
echo "=========================================="

uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/04rb_split_half_reliability.py" \
    --sub_id "$SUBJECT_ID" \
    --parcellation "$PARCELLATION"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Failed (exit code $EXIT_CODE)"
    exit $EXIT_CODE
else
    echo "SUCCESS"
fi

seff $SLURM_JOB_ID 2>/dev/null || true
