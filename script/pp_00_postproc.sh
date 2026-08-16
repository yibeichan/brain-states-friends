#!/bin/bash
#
# SLURM submission script for Petit Prince fMRIPrep post-processing
#
# Purpose: Apply minimal confound regression to Petit Prince fMRIPrep CIFTI outputs.
#          Reuses 00_postproc.py with --fmriprep_dir pointing to petit-prince.fmriprep/.
#
# Petit Prince: audiobook listening (French lppFR + English lppEN), ~18 runs per subject
# Subjects: sub-01, sub-02, sub-03, sub-05, sub-06 (no sub-04)
#
# Output: {SCRATCH_DIR}/output/00_postproc/{sub_id}/*task-lpp{FR,EN}*_cleaned.dtseries.nii
#         (coexists with Friends, Movie10, and HP files in same directory)
#
# Prerequisites: datalad get petit-prince.fmriprep data before running
#
# Usage:
#   sbatch script/pp_00_postproc.sh
#
# Documentation: the design notes

#SBATCH --job-name=pp_postproc
#SBATCH --partition=pi_satra
#SBATCH --output=logs/pp_00_postproc_%A_%a.out
#SBATCH --error=logs/pp_00_postproc_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-4
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=YOUR_EMAIL@example.com

# Shared preamble: PROJECT_DIR/SCRIPT_DIR, uv on PATH, logs/ (utils/_env.sh)
_ENV="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)/script/utils/_env.sh"
[ -f "$_ENV" ] || _ENV="${SLURM_SUBMIT_DIR:-.}/script/utils/_env.sh"
source "$_ENV" || { echo "ERROR: cannot locate script/utils/_env.sh — submit from the repo root" >&2; exit 1; }

# Load environment for DATA_DIR
set -a; source "${PROJECT_DIR}/.env"; set +a

# Subject array (5 subjects - no sub-04 in Petit Prince)
sub_ids=("sub-01" "sub-02" "sub-03" "sub-05" "sub-06")
TASK_ID=${sub_ids[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo "Petit Prince Post-processing - Minimal Confound Strategy"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Subject: $TASK_ID"
echo "Task: petit-prince"
echo "Node: $HOSTNAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "============================================================"
echo ""

# Run post-processing with --fmriprep_dir (PP lives outside cneuromod.processed/)
uv run --project "${PROJECT_DIR}" --no-sync python "${SCRIPT_DIR}/00_postproc.py" "${TASK_ID}" "petit-prince" \
    --fmriprep_dir "${DATA_DIR}/petit-prince.fmriprep/${TASK_ID}" \
    --n_jobs $SLURM_CPUS_PER_TASK

exit_code=$?

echo ""
echo "============================================================"
echo "Job completed with exit code: $exit_code"
echo "End time: $(date)"
echo "============================================================"

exit $exit_code
