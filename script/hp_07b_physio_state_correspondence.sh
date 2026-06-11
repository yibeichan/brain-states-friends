#!/bin/bash
#SBATCH --job-name=hp_07b_physio_corr
#SBATCH --output=logs/hp_07b_physio_corr_%j.out
#SBATCH --error=logs/hp_07b_physio_corr_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=ou_bcs_normal,pi_satra

# =============================================================================
# hp_07b - Physio-State Correspondence Analysis (Harry Potter)
# =============================================================================
# Tests whether brain states have distinct autonomic signatures in Harry Potter.
# Uses Friends recurrence taxonomy (state categories from 05a).
# 5 analyses: state profiles, multi-lag, TTAs, cross-episode consistency,
# arousal-diversity.
#
# Prerequisites:
#   - hp_07a completed (Harry Potter physio features)
#   - hp_04 decoded_states.pkl
#   - Friends 05a recurrence_summary.json
#
# Usage:
#   sbatch --export=SUB_ID=sub-01 script/hp_07b_physio_state_correspondence.sh
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
echo "hp_07b - Physio-State Correspondence (Harry Potter)"
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
    --stimulus harrypotter \
    ${VT_ARG} \
    --sensitivity_analysis

echo "=============================================="
echo "Harry Potter physio-state correspondence complete!"
echo "=============================================="
