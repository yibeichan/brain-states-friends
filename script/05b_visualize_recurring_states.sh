#!/bin/bash
#SBATCH --job-name=05b_viz
#SBATCH --output=logs/05b_viz_%j.out
#SBATCH --error=logs/05b_viz_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --array=0-5
#SBATCH --cpus-per-task=2
#SBATCH --partition=mit_normal

# =============================================================================
# 05b - Visualize Recurring Brain States with yabplot (Cortical + Subcortical)
# =============================================================================
# Creates multi-panel figure showing top recurring brain states on:
#   - Cortical surface (yab.plot_cortical, Schaefer-100 atlas)
#   - Subcortical 3D structures (yab.plot_subcortical, custom 4s156_subcortical atlas)
#
# Prerequisites:
#   - Step 05a completed (recurrence_summary.json, fractional_occupancy.pkl)
#   - Step 04 (select mode) completed (state_means_parcel.npy)
#   - yabplot installed (git dependency; https://github.com/yibeichan/yabplot)
#   - Subcortical VTK meshes built via yabplot's
#       tools/build_4s156_subcortical.py
#
# Usage:
#   sbatch --export=SUB_ID=sub-01 script/05b_visualize_recurring_states.sh
#   OR
#   bash script/05b_visualize_recurring_states.sh
# =============================================================================

set -euo pipefail

# Determine project directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
fi

mkdir -p "${PROJECT_DIR}/logs"

eval "$(micromamba shell hook --shell bash)"
micromamba activate friends-states

# Configuration (override via --export on sbatch)
subjects=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06)
SUB_ID="${SUB_ID:-${subjects[${SLURM_ARRAY_TASK_ID:-0}]}}"
PARCELLATION="${PARCELLATION:-atlas-4S156Parcels}"
N_STATES="${N_STATES:-5}"
VT="${VT:-0.95}"

# PyVista headless rendering — OSMesa software renderer (no GPU needed)
export PYOPENGL_PLATFORM=osmesa
export DISPLAY=""

echo "=============================================="
echo "05b - Visualize Recurring Brain States (yabplot)"
echo "=============================================="
echo "Subject:        ${SUB_ID}"
echo "Parcellation:   ${PARCELLATION}"
echo "Number of states: ${N_STATES}"
echo "Rendering backend: OSMesa (headless)"
echo "=============================================="

VT_ARG=""
if [ -n "${VT}" ]; then
    VT_ARG="--vt ${VT}"
fi

uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/05b_visualize_recurring_states.py" \
    --sub_id "${SUB_ID}" \
    --parcellation "${PARCELLATION}" \
    --n_states "${N_STATES}" \
    ${VT_ARG}

echo "=============================================="
echo "05b visualization complete!"
echo "=============================================="
