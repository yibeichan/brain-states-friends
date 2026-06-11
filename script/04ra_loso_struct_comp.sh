#!/bin/bash
#SBATCH --job-name=04ra_loso_struct_comp
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

# =============================================================================
# LOSO Structural Comparison — SLURM Wrapper
# =============================================================================
#
# Compares structural invariants across 6 LOSO folds.
#
# Prerequisites: 04 loso_fit completed for all 6 seasons.
#
# Usage:
#   sbatch --export=SUBJECT_ID=sub-01 script/04ra_loso_struct_comp.sh
#
# =============================================================================

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
fi

mkdir -p "${PROJECT_DIR}/logs"

eval "$(micromamba shell hook --shell bash)"
micromamba activate friends-states

SUBJECT_ID=${SUBJECT_ID:-"sub-01"}
PARCELLATION=${PARCELLATION:-"atlas-4S156Parcels"}

echo "=========================================="
echo "04r LOSO Structural Comparison"
echo "=========================================="
echo "Subject:      $SUBJECT_ID"
echo "Parcellation: $PARCELLATION"
echo "=========================================="

uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/04ra_loso_struct_comp.py" \
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
