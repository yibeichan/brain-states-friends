#!/bin/bash
#
# SLURM submission script for fMRIPrep post-processing (Script 00)
#
# Purpose: Apply minimal confound regression to fMRIPrep CIFTI outputs
#
# Preprocessing Strategy (November 2025):
#   - Minimal confounds: 6 motion + 2 WM/CSF + high-pass filter
#   - Voxel-wise z-scoring (essential for unbiased parcellation)
#   - No global signal regression (preserves global patterns)
#   - No scrubbing (maintains narrative continuity)
#
# Documentation: the design notes
#
# Usage:
#   sbatch script/00_postproc.sh
#
# To auto-save outputs with DataLad after all subjects finish:
#   JOB=$(sbatch --parsable script/00_postproc.sh)
#   sbatch --dependency=afterok:$JOB --export=STAGE=00,SUBJECT_ID=all script/utils/datalad_save.sh
#
# Output: {SCRATCH_DIR}/output/00_postproc/{sub_id}/*_cleaned.dtseries.nii

#SBATCH --job-name=postproc_friends
#SBATCH --partition=mit_normal,pi_satra
#SBATCH --output=logs/00_postproc_%A_%a.out
#SBATCH --error=logs/00_postproc_%A_%a.err
# Friends is ~290 runs/subject vs ~61 for movie10 at 02:00 — linear scaling
# gives ~9.5h, so allow 12h.
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-5
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=YOUR_EMAIL@example.com

# Shared preamble: PROJECT_DIR/SCRIPT_DIR, uv on PATH, logs/ (utils/_env.sh)
_ENV="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)/script/utils/_env.sh"
[ -f "$_ENV" ] || _ENV="${SLURM_SUBMIT_DIR:-.}/script/utils/_env.sh"
source "$_ENV" || { echo "ERROR: cannot locate script/utils/_env.sh — submit from the repo root" >&2; exit 1; }

# Subject array (6 subjects)
sub_ids=("sub-01" "sub-02" "sub-03" "sub-04" "sub-05" "sub-06")

# A SLURM array task id is required to select the subject; without this
# guard an unset id silently selects index 0 (sub-01).
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set. Submit via sbatch, or test a"
    echo "single task with: SLURM_ARRAY_TASK_ID=0 bash script/00_postproc.sh"
    exit 1
fi

# Get subject ID for this array task
TASK_ID=${sub_ids[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo "fMRIPrep Post-processing - Minimal Confound Strategy"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Subject: $TASK_ID"
echo "Node: $HOSTNAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "============================================================"
echo ""

# Run post-processing with minimal confound strategy
# Uses --n_jobs to match SLURM allocation for efficient parallelization
uv run --project "${PROJECT_DIR}" --no-sync python "${SCRIPT_DIR}/00_postproc.py" "${TASK_ID}" "friends" --n_jobs "${SLURM_CPUS_PER_TASK:-4}"

exit_code=$?

echo ""
echo "============================================================"
echo "Job completed with exit code: $exit_code"
echo "End time: $(date)"
echo "============================================================"

exit $exit_code
