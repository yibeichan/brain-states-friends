#!/bin/bash
#SBATCH --job-name=01_get_parcel_label
#SBATCH --partition=mit_normal
#SBATCH --output=logs/01_get_parcel_label_%A_%a.out
#SBATCH --error=logs/01_get_parcel_label_%A_%a.err
#SBATCH --array=0-10
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# Determine the project directory (do this early for path resolution).
# Under sbatch, BASH_SOURCE points to the SLURM spool dir, so prefer
# SLURM_SUBMIT_DIR; fall back to the script's own location otherwise.
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
fi
SCRIPT_DIR="${PROJECT_DIR}/script"

# Create logs directory using absolute path (logs/ is gitignored)
mkdir -p "${PROJECT_DIR}/logs"

# Ensure the user-local uv install is on PATH (SLURM jobs may not inherit it)
export PATH="$HOME/.local/bin:$PATH"

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
