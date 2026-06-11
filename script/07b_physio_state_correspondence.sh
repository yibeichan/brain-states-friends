#!/bin/bash
#SBATCH --job-name=07b_physio_corr
#SBATCH --output=logs/07b_physio_corr_%j.out
#SBATCH --error=logs/07b_physio_corr_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=ou_bcs_normal,pi_satra

# =============================================================================
# 07b - Physio-State Correspondence Analysis (Friends)
# =============================================================================
# Tests whether brain states have distinct autonomic signatures.
# 5 analyses: state profiles, multi-lag, TTAs, cross-episode consistency,
# arousal-diversity.
#
# Prerequisites:
#   - 07a completed (Friends physio features)
#   - 04 select + 05a completed
#
# Usage:
#   sbatch --export=SUB_ID=sub-01 script/07b_physio_state_correspondence.sh
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

SUB_ID="${SUB_ID:-sub-01}"
PARCELLATION="${PARCELLATION:-atlas-4S156Parcels}"
VT="${VT:-0.95}"

echo "=============================================="
echo "07b - Physio-State Correspondence (Friends)"
echo "=============================================="
echo "Subject: ${SUB_ID}"
echo "Parcellation: ${PARCELLATION}"
echo "=============================================="

VT_ARG=""
if [ -n "${VT}" ]; then
    VT_ARG="--vt ${VT}"
fi

uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/07b_physio_state_correspondence.py" \
    --sub_id "${SUB_ID}" \
    --parcellation "${PARCELLATION}" \
    ${VT_ARG} \
    --sensitivity_analysis

echo "=============================================="
echo "Physio-state correspondence complete!"
echo "=============================================="
