#!/bin/bash
#SBATCH --job-name=07c_xstim_physio
#SBATCH --output=logs/07c_xstim_physio_%j.out
#SBATCH --error=logs/07c_xstim_physio_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=ou_bcs_normal,pi_satra

# =============================================================================
# 07c - Cross-Stimulus Physio Correspondence
# =============================================================================
# Tests whether brain states maintain consistent autonomic signatures across
# Friends and Movie10. Four analyses: C1 signature stability, C2 genre profiles,
# C3 arousal modulation, C4 cross-stimulus TTAs.
#
# Prerequisites:
#   - 07a completed for both Friends and Movie10
#   - 04 select + 05a + m10_04 completed
#
# Usage:
#   sbatch --export=SUB_ID=sub-01,VT=0.99 script/07c_cross_stimulus_physio.sh
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
echo "07c - Cross-Stimulus Physio Correspondence"
echo "=============================================="
echo "Subject: ${SUB_ID}"
echo "Parcellation: ${PARCELLATION}"
echo "VT: ${VT}"
echo "=============================================="

uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/07c_cross_stimulus_physio.py" \
    --sub_id "${SUB_ID}" \
    --parcellation "${PARCELLATION}" \
    --vt "${VT}"

echo "=============================================="
echo "Cross-stimulus physio correspondence complete!"
echo "=============================================="
