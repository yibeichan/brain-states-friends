#!/bin/bash
#
# SLURM submission script for hcptrt resting-state fMRIPrep post-processing
#
# Purpose: Apply minimal confound regression to hcptrt restingstate CIFTI outputs.
#          Reuses 00_postproc.py with --fmriprep_dir + --bids_task restingstate.
#
# Resting state: one ~15 min run (~600 TRs, TR 1.49 s) per hcptrt session;
#                5/5/5/5/4/6 runs for sub-01..06 (sub-04 present, unlike HP/PP)
#
# Output: {SCRATCH_DIR}/output/00_postproc/{sub_id}/*task-restingstate*_bold_cleaned.dtseries.nii
#         (coexists with Friends/Movie10/HP/PP files in same directory)
#
# Usage:
#   sbatch script/rest_00_postproc.sh

#SBATCH --job-name=rest_postproc
#SBATCH --partition=mit_normal
#SBATCH --output=logs/rest_00_postproc_%A_%a.out
#SBATCH --error=logs/rest_00_postproc_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-5

# Ensure the user-local uv install is on PATH (SLURM jobs may not inherit it)
export PATH="$HOME/.local/bin:$PATH"

# Determine project directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
fi
SCRIPT_DIR="${PROJECT_DIR}/script"
mkdir -p "${PROJECT_DIR}/logs"

# hcptrt fMRIPrep derivatives (read-only, datalad-retrieved)
HCPTRT_FMRIPREP="/orcd/data/satra/002/datasets/all_about_cneuromod/cneuromod.processed/fmriprep/hcptrt"

# All six subjects (rest has sub-04, unlike HP/PP)
sub_ids=("sub-01" "sub-02" "sub-03" "sub-04" "sub-05" "sub-06")
TASK_ID=${sub_ids[$SLURM_ARRAY_TASK_ID]}

uv run --project "${PROJECT_DIR}" --no-sync python "${SCRIPT_DIR}/00_postproc.py" "${TASK_ID}" "hcptrt" \
    --fmriprep_dir "${HCPTRT_FMRIPREP}/${TASK_ID}" \
    --bids_task restingstate \
    --n_jobs "$SLURM_CPUS_PER_TASK"
