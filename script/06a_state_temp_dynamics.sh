#!/bin/bash
#SBATCH --job-name=06a_temp_dyn
#SBATCH --output=logs/06a_temp_dyn_%A_%a.out
#SBATCH --error=logs/06a_temp_dyn_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --array=0-5

# Shared preamble: PROJECT_DIR/SCRIPT_DIR, uv on PATH, logs/ (utils/_env.sh)
_ENV="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)/script/utils/_env.sh"
[ -f "$_ENV" ] || _ENV="${SLURM_SUBMIT_DIR:-.}/script/utils/_env.sh"
source "$_ENV" || { echo "ERROR: cannot locate script/utils/_env.sh — submit from the repo root" >&2; exit 1; }

# =============================================================================
# 06a - State Temporal Dynamics Analysis
# =============================================================================
# Computes sequence blocks, start/end times, dwell times, and transition properties
# for each state category, mapping them back to the decoded HMM trace.
#
# Requires GPU partition because the pickled HMM model depends on JAX.
#
# Prerequisites:
#   - Step 04 completed (04_combined_hdphmm.py)
#   - Step 05a completed (05a_recurrence_analysis.py)
#
# Usage:
#   sbatch 06a_state_temp_dynamics.sh
#   OR
#   bash 06a_state_temp_dynamics.sh
# =============================================================================

set -e

# JAX configuration
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# Configuration (override via --export on sbatch)
subjects=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06)
SUB_ID="${SUB_ID:-${subjects[${SLURM_ARRAY_TASK_ID:-0}]}}"
PARCELLATION="${PARCELLATION:-atlas-4S156Parcels}"
VT="${VT:-0.95}"

# Print header
echo "=============================================="
echo "06a - State Temporal Dynamics Analysis"
echo "=============================================="
echo "Subject: ${SUB_ID}"
echo "Parcellation: ${PARCELLATION}"
echo "=============================================="

# Run analysis
VT_ARG=""
if [ -n "${VT}" ]; then
    VT_ARG="--vt ${VT}"
fi

uv run --project "${PROJECT_DIR}" --extra gpu python "${PROJECT_DIR}/script/06a_state_temp_dynamics.py" \
    --sub_id "${SUB_ID}" \
    --parcellation "${PARCELLATION}" \
    ${VT_ARG}

echo "=============================================="
echo "Temporal Dynamics Analysis complete!"
echo "=============================================="
