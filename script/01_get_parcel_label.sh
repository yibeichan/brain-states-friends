#!/bin/bash
#SBATCH --job-name=01_get_parcel_label
#SBATCH --partition=mit_normal
#SBATCH --output=logs/01_get_parcel_label_%A_%a.out
#SBATCH --error=logs/01_get_parcel_label_%A_%a.err
#SBATCH --array=0-10
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# Shared preamble: PROJECT_DIR/SCRIPT_DIR, uv on PATH, logs/ (utils/_env.sh)
_ENV="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)/script/utils/_env.sh"
[ -f "$_ENV" ] || _ENV="${SLURM_SUBMIT_DIR:-.}/script/utils/_env.sh"
source "$_ENV" || { echo "ERROR: cannot locate script/utils/_env.sh — submit from the repo root" >&2; exit 1; }

parcellations=("Schaefer2018" "4S156" "4S256" "4S356" "4S456" "4S556" "4S656" "4S756" "4S856" "4S956" "4S1056")

# A SLURM array task id is required to select the parcellation; without
# this guard an unset id silently selects index 0 (Schaefer2018 only).
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set. Submit via sbatch, or test a"
    echo "single task with: SLURM_ARRAY_TASK_ID=0 bash script/01_get_parcel_label.sh"
    exit 1
fi

TASK_ID=${parcellations[$SLURM_ARRAY_TASK_ID]}

echo "Processing parcellation: $TASK_ID"

uv run --project "${PROJECT_DIR}" --no-sync python "${SCRIPT_DIR}/01_get_parcel_label.py" --parcellation "$TASK_ID"
