#!/bin/bash
#SBATCH --job-name=07a_physio_friends
#SBATCH --output=logs/07a_physio_friends_%A_%a.out
#SBATCH --error=logs/07a_physio_friends_%A_%a.err
#SBATCH --time=00:20:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --array=0-5

# =============================================================================
# 07a - Physiological Feature Extraction (Friends)
# =============================================================================
# Extracts TR-aligned physiological features (HR, HRV, breathing rate, RVT,
# EDA tonic/phasic, SCR) from Friends physprep data.
# SLURM array job: one task per subject (array index 0-5 → sub-01 to sub-06).
#
# Fully independent of the brain pipeline — discovers runs from physprep
# and derives TR counts from recording length.
#
# Prerequisites:
#   - Friends physprep data available
#
# Usage:
#   sbatch script/07a_physio_features.sh                    # all 6 subjects
#   sbatch --array=0 script/07a_physio_features.sh          # sub-01 only
#   sbatch --array=0,2,5 script/07a_physio_features.sh      # sub-01,03,06
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

# Map array index to subject ID
SUBJECTS=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06)
SUB_ID="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"

echo "=============================================="
echo "07a - Physio Feature Extraction (Friends)"
echo "=============================================="
echo "Subject: ${SUB_ID}"
echo "=============================================="

uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/07a_physio_features.py" \
    --sub_id "${SUB_ID}" \
    --stimulus friends

echo "=============================================="
echo "Friends physio extraction complete!"
echo "=============================================="
