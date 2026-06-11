#!/bin/bash
#SBATCH --job-name=08a_content
#SBATCH --output=logs/08a_content_%A_%a.out
#SBATCH --error=logs/08a_content_%A_%a.err
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --array=0-5

# =============================================================================
# 08a - Content Feature Extraction
# =============================================================================
# Extracts TR-aligned content features (speech presence, dialogue rate,
# speakers, silence, scene boundaries) from te-charnet annotations.
#
# SLURM array job: one task per subject (array index 0-5 -> sub-01 to sub-06).
#
# Prerequisites:
#   - 04_combined_hdphmm.py (mode: select) completed
#   - te-charnet annotations available
#
# Usage:
#   sbatch script/08a_content_features.sh                    # all 6 subjects
#   sbatch --array=0 script/08a_content_features.sh          # sub-01 only
#   sbatch --export=SUB_ID=sub-01 script/08a_content_features.sh  # single subject
# =============================================================================

set -e

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
fi

mkdir -p "${PROJECT_DIR}/logs"

eval "$(micromamba shell hook --shell bash)"
micromamba activate friends-states

# Subject ID: from export or array index
if [ -z "$SUB_ID" ]; then
    SUBJECTS=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06)
    SUB_ID="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"
fi

PARCELLATION="${PARCELLATION:-atlas-4S156Parcels}"

echo "=============================================="
echo "08a - Content Feature Extraction"
echo "=============================================="
echo "Subject: ${SUB_ID}"
echo "Parcellation: ${PARCELLATION}"
echo "=============================================="

CMD="uv run --project ${PROJECT_DIR} --no-sync python ${PROJECT_DIR}/script/08a_content_features.py \
    --sub_id ${SUB_ID} \
    --parcellation ${PARCELLATION}"

# Optional: variance threshold
if [ -n "$VT" ]; then
    CMD="${CMD} --vt ${VT}"
    echo "Variance threshold: ${VT}"
fi

eval $CMD

echo "=============================================="
echo "Content feature extraction complete!"
echo "=============================================="
