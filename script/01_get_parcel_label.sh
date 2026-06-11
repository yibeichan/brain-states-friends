#!/bin/bash
#SBATCH --job-name=01_get_parcel_label
#SBATCH --partition=mit_normal
#SBATCH --output=logs/01_get_parcel_label_%A_%a.out
#SBATCH --error=logs/01_get_parcel_label_%A_%a.err
#SBATCH --array=0-10
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

mkdir -p logs

source ~/.bashrc
micromamba activate friends-states

parcellations=("Schaefer2018" "4S156" "4S256" "4S356" "4S456" "4S556" "4S656" "4S756" "4S856" "4S956" "4S1056")

TASK_ID=${parcellations[$SLURM_ARRAY_TASK_ID]}

echo "Processing parcellation: $TASK_ID"

# Note: BASH_SOURCE points to SLURM spool dir, so use SLURM_SUBMIT_DIR instead
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/script"
uv run --no-sync python "${SCRIPT_DIR}/01_get_parcel_label.py" --parcellation "$TASK_ID"
