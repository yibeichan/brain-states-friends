#!/bin/bash
#SBATCH --job-name=m10_07a_physio
#SBATCH --output=logs/m10_07a_physio_%A_%a.out
#SBATCH --error=logs/m10_07a_physio_%A_%a.err
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --array=0-5

# =============================================================================
# m10_07a - Physiological Feature Extraction (Movie10)
# =============================================================================
# Extracts TR-aligned physiological features from Movie10 physprep data.
# SLURM array job: one task per subject (array index 0-5 → sub-01 to sub-06).
#
# Fully independent of the brain pipeline - discovers runs from physprep
# and derives TR counts from recording length.
#
# Prerequisites:
#   - Movie10 physprep data available
#
# Usage:
#   sbatch script/m10_07a_physio_features.sh                    # all 6 subjects
#   sbatch --array=0 script/m10_07a_physio_features.sh          # sub-01 only
#   sbatch --array=0,2,5 script/m10_07a_physio_features.sh      # sub-01,03,06
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
echo "m10_07a - Physio Feature Extraction (Movie10)"
echo "=============================================="
echo "Subject: ${SUB_ID}"
echo "=============================================="

uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/07a_physio_features.py" \
    --sub_id "${SUB_ID}" \
    --stimulus movie10

echo "=============================================="
echo "Movie10 physio extraction complete!"
echo "=============================================="
