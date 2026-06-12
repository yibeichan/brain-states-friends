#!/bin/bash
#SBATCH --job-name=04rc_fc
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-7
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

# =============================================================================
# Reliability FC Computation - SLURM Wrapper
# =============================================================================
#
# Computes empirical within-state FC for LOSO folds and split-half halves.
# Array tasks 0-5 = LOSO seasons 1-6, tasks 6-7 = split-half A/B.
#
# Prerequisites:
#   - 04 loso_fit and split_half_fit completed
#   - 02_extract_parcel_ts completed
#
# Usage:
#   sbatch --export=SUBJECT_ID=sub-01 script/04rc_reliability_fc.sh
#
#   For sub-04 (4 seasons): override array
#   sbatch --export=SUBJECT_ID=sub-04 --array=0-5 script/04rc_reliability_fc.sh
#   (tasks 4-5 map to split-half; tasks 0-3 map to LOSO seasons 1-4)
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
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "=========================================="
echo "04rc Reliability FC"
echo "=========================================="
echo "Subject:      $SUBJECT_ID"
echo "Parcellation: $PARCELLATION"
echo "Task ID:      $TASK_ID"

if [ "$TASK_ID" -le 5 ]; then
    FOLD=$((TASK_ID + 1))
    echo "Mode:         loso (season $FOLD)"
    echo "=========================================="
    uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/04rc_reliability_fc.py" \
        --sub_id "$SUBJECT_ID" \
        --parcellation "$PARCELLATION" \
        --mode loso \
        --fold "$FOLD"
elif [ "$TASK_ID" -eq 6 ]; then
    echo "Mode:         split_half A"
    echo "=========================================="
    uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/04rc_reliability_fc.py" \
        --sub_id "$SUBJECT_ID" \
        --parcellation "$PARCELLATION" \
        --mode split_half \
        --half A
elif [ "$TASK_ID" -eq 7 ]; then
    echo "Mode:         split_half B"
    echo "=========================================="
    uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/04rc_reliability_fc.py" \
        --sub_id "$SUBJECT_ID" \
        --parcellation "$PARCELLATION" \
        --mode split_half \
        --half B
fi

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Failed (exit code $EXIT_CODE)"
    exit $EXIT_CODE
else
    echo "SUCCESS"
fi

seff $SLURM_JOB_ID 2>/dev/null || true
