#!/bin/bash
#SBATCH --job-name=04_combined_hdphmm
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-4          # Stage 1 default: 5 vt configs; override per stage
#SBATCH --time=13:55:00      # fit default; see NOTES ON RESOURCES below
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8    # needed for joblib parallel forward-backward
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

# =============================================================================
# Combined weak-limit HMM - SLURM Wrapper (Two-Stage Model Selection)
# =============================================================================
#
# Two-stage pipeline controlled by MODE and STAGE environment variables:
#
# STAGE 1 FIT: Select variance threshold (PCA dimensionality)
#   Array: 0-4 (5 vt configs × 5 seeds each)
#   sbatch --export=SUBJECT_ID=sub-01,STAGE=1 script/04_combined_hdphmm.sh
#
# STAGE 2 FIT: Select K/gamma/cov at the chosen vt
#   Array size depends on selected vt (12-24 configs). Check stage1_result.json.
#   For vt with full+diag cov (vt≤0.85): 3 gamma × 2 cov × 4 nc = 24 configs → --array=0-23
#   For vt with diag only (vt≥0.90):     3 gamma × 1 cov × 4 nc = 12 configs → --array=0-11
#   sbatch --array=0-11 --export=SUBJECT_ID=sub-01,STAGE=2 script/04_combined_hdphmm.sh
#
# SELECT MODE: Two-stage selection + final refit + decode (single task)
#   sbatch --array=0 --export=SUBJECT_ID=sub-01,MODE=select \
#          --time=10:00:00 script/04_combined_hdphmm.sh
#
# SELECT MODE WITH FIXED VT: bypass Stage 1 metric, use externally validated vt
#   sbatch --array=0 --export=SUBJECT_ID=sub-01,MODE=select,FIXED_VT=0.95 \
#          --time=10:00:00 script/04_combined_hdphmm.sh
#
# STAGE 2 FIT WITH FIXED VT: run Stage 2 without stage1_result.json on disk
#   sbatch --array=0-11 --export=SUBJECT_ID=sub-01,STAGE=2,FIXED_VT=0.95 \
#          script/04_combined_hdphmm.sh
#
# SELECT SEED MODE: Parallelize final refit (one seed per task)
#   sbatch --array=0-9 --export=SUBJECT_ID=sub-01,MODE=select_seed \
#          --time=10:00:00 script/04_combined_hdphmm.sh
#
# SELECT FINALIZE MODE: Decode + save after all seeds done
#   sbatch --array=0 --export=SUBJECT_ID=sub-01,MODE=select_finalize \
#          --time=01:00:00 script/04_combined_hdphmm.sh
#
# LOSO FIT MODE: Strategy B - refit best config for one held-out season
#   sbatch --array=1-6 --export=SUBJECT_ID=sub-01,MODE=loso_fit \
#          --time=20:00:00 script/04_combined_hdphmm.sh
#   # With explicit vt:
#   sbatch --array=1-6 --export=SUBJECT_ID=sub-01,MODE=loso_fit,FIXED_VT=0.95 \
#          --time=20:00:00 script/04_combined_hdphmm.sh
#   # sub-04: only seasons 1-4
#   sbatch --array=1-4 --export=SUBJECT_ID=sub-04,MODE=loso_fit \
#          --time=20:00:00 script/04_combined_hdphmm.sh
#
# SPLIT-HALF FIT MODE: Fit best config on each interleaved half
#   sbatch --array=0-1 --export=SUBJECT_ID=sub-01,MODE=split_half_fit \
#          --time=20:00:00 script/04_combined_hdphmm.sh
#   # Array 0 = half A, 1 = half B
#
# ALL SUBJECTS (Stage 1 example):
#   for sub in 01 02 03 04 05 06; do
#       sbatch --export=SUBJECT_ID=sub-${sub},STAGE=1 script/04_combined_hdphmm.sh
#   done
#
# NOTES ON RESOURCES (with parallel E-step, --cpus-per-task=8):
#   Stage 1 fit (any vt, diag, nc=60):   --time=14:00:00, --mem=32G
#   Stage 2 fit vt≤0.85, full:           --time=14:00:00, --mem=32G
#   Stage 2 fit vt≤0.90, diag:           --time=14:00:00, --mem=32G
#   Stage 2 fit vt=0.95/0.99, diag:      --time=20:00:00, --mem=32G
#   select:                               --time=10:00:00, --mem=32G
#   loso_fit:                             --time=20:00:00, --mem=32G
#
# Per-seed JSON checkpointing is built in for all fit modes:
# if a task times out and is resubmitted, completed seeds are skipped.
# =============================================================================

# Determine project directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
fi

mkdir -p "${PROJECT_DIR}/logs"

# Ensure the user-local uv install is on PATH (SLURM jobs may not inherit it)
export PATH="$HOME/.local/bin:$PATH"

# =============================================================================
# Configuration (override with --export on sbatch command line)
# =============================================================================

SUBJECT_ID=${SUBJECT_ID:-"sub-01"}
PARCELLATION=${PARCELLATION:-"atlas-4S156Parcels"}
MODE=${MODE:-"fit"}
STAGE=${STAGE:-1}
FIXED_VT=${FIXED_VT:-""}      # Optional: skip Stage 1, use this vt (e.g., 0.95)
N_FIT_SEEDS=${N_FIT_SEEDS:-5}
N_FINAL_SEEDS=${N_FINAL_SEEDS:-10}
N_JOBS=${N_JOBS:-8}

# Map short parcellation names to full names
case "$PARCELLATION" in
    156)  PARCELLATION="atlas-4S156Parcels" ;;
    456)  PARCELLATION="atlas-4S456Parcels" ;;
    1056) PARCELLATION="atlas-4S1056Parcels" ;;
    *)    ;;  # Already full name
esac

# =============================================================================
# Validate prerequisites
# =============================================================================

SCRATCH_DIR=${SCRATCH_DIR:-$(grep SCRATCH_DIR "${PROJECT_DIR}/.env" | cut -d'=' -f2 | tr -d '"' | tr -d "'")}

COMBINED_DIR="${SCRATCH_DIR}/output/03a_pca4combined_hmm/${PARCELLATION}/${SUBJECT_ID}"
if [ ! -f "${COMBINED_DIR}/n_pcs_lookup.json" ]; then
    echo "ERROR: 03a_pca4combined_hmm output not found at ${COMBINED_DIR}"
    echo "Run script/03a_pca4combined_hmm.py for ${SUBJECT_ID} first."
    exit 1
fi

if [ ! -f "${COMBINED_DIR}/splits/primary.json" ]; then
    echo "ERROR: Primary split not found at ${COMBINED_DIR}/splits/primary.json"
    echo "Run script/03a_pca4combined_hmm.py for ${SUBJECT_ID} first."
    exit 1
fi

OUTPUT_DIR="${SCRATCH_DIR}/output/04_combined_hdphmm/${PARCELLATION}/${SUBJECT_ID}"

# Stage 2 requires Stage 1 result (unless FIXED_VT bypasses Stage 1)
if [ "$MODE" = "fit" ] && [ "$STAGE" = "2" ] && [ -z "${FIXED_VT}" ]; then
    if [ ! -f "${OUTPUT_DIR}/stage1_result.json" ]; then
        echo "ERROR: Stage 1 result not found at ${OUTPUT_DIR}/stage1_result.json"
        echo "Run Stage 1 fit + select first, or set FIXED_VT=0.95 to bypass Stage 1."
        exit 1
    fi
fi

# loso_fit requires final results from select mode
# Resolve vt: from FIXED_VT or stage1_result.json
if [ "$MODE" = "loso_fit" ]; then
    if [ -n "${FIXED_VT}" ]; then
        _VT="${FIXED_VT}"
    elif [ -f "${OUTPUT_DIR}/stage1_result.json" ]; then
        _VT=$(uv run --project "${PROJECT_DIR}" --no-sync python3 -c "import json; print(json.load(open('${OUTPUT_DIR}/stage1_result.json'))['selected_vt'])")
    else
        echo "ERROR: Cannot determine vt for loso_fit."
        echo "Set FIXED_VT or ensure stage1_result.json exists."
        exit 1
    fi
    _VT_FMT=$(printf "%.2f" "$_VT")
    if [ ! -f "${OUTPUT_DIR}/final/vt${_VT_FMT}/final_results.json" ]; then
        echo "ERROR: Select-mode results not found at ${OUTPUT_DIR}/final/vt${_VT_FMT}/final_results.json"
        echo "Run MODE=select first."
        exit 1
    fi
fi

# =============================================================================
# Print configuration
# =============================================================================

echo "=========================================="
echo "04 Combined weak-limit HMM"
echo "=========================================="
echo "SLURM Job ID:     $SLURM_JOB_ID"
echo "Array Task ID:    ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Subject:          $SUBJECT_ID"
echo "Parcellation:     $PARCELLATION"
echo "Mode:             $MODE"
case "$MODE" in
    fit)
        echo "Stage:            $STAGE"
        echo "Task index:       ${SLURM_ARRAY_TASK_ID}"
        echo "Seeds per config: $N_FIT_SEEDS"
        [ -n "${FIXED_VT}" ] && echo "Fixed vt:         $FIXED_VT (bypassing Stage 1)"
        ;;
    select)
        echo "Final seeds:      $N_FINAL_SEEDS"
        [ -n "${FIXED_VT}" ] && echo "Fixed vt:         $FIXED_VT (bypassing Stage 1)"
        ;;
    select_seed)
        echo "Seed index:       ${SLURM_ARRAY_TASK_ID}"
        ;;
    select_finalize)
        echo "Final seeds:      $N_FINAL_SEEDS"
        ;;
    loso_fit)
        echo "LOSO season:      ${SLURM_ARRAY_TASK_ID}"
        echo "Final seeds:      $N_FINAL_SEEDS"
        [ -n "${FIXED_VT}" ] && echo "Fixed vt:         $FIXED_VT"
        ;;
    split_half_fit)
        HALF_LABEL=$( [ "${SLURM_ARRAY_TASK_ID}" = "0" ] && echo "A" || echo "B" )
        echo "Half:             ${HALF_LABEL} (task ${SLURM_ARRAY_TASK_ID})"
        echo "Final seeds:      $N_FINAL_SEEDS"
        [ -n "${FIXED_VT}" ] && echo "Fixed vt:         $FIXED_VT"
        ;;
esac
echo "=========================================="

# =============================================================================
# Run
# =============================================================================

FORCE_FLAG=""
if [ "${FORCE_REFIT}" = "true" ]; then
    FORCE_FLAG="--force_refit"
fi

FIXED_VT_ARG=""
if [ -n "${FIXED_VT}" ]; then
    FIXED_VT_ARG="--fixed_vt ${FIXED_VT}"
fi

if [ "$MODE" = "fit" ]; then
    uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/04_combined_hdphmm.py" \
        --sub_id "$SUBJECT_ID" \
        --parcellation "$PARCELLATION" \
        --mode fit \
        --stage "$STAGE" \
        --task_index "$SLURM_ARRAY_TASK_ID" \
        --n_fit_seeds "$N_FIT_SEEDS" \
        --n_jobs "$N_JOBS" \
        $FIXED_VT_ARG

elif [ "$MODE" = "select" ]; then
    uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/04_combined_hdphmm.py" \
        --sub_id "$SUBJECT_ID" \
        --parcellation "$PARCELLATION" \
        --mode select \
        --n_final_seeds "$N_FINAL_SEEDS" \
        --n_jobs "$N_JOBS" \
        $FORCE_FLAG \
        $FIXED_VT_ARG

elif [ "$MODE" = "select_seed" ]; then
    uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/04_combined_hdphmm.py" \
        --sub_id "$SUBJECT_ID" \
        --parcellation "$PARCELLATION" \
        --mode select_seed \
        --seed_index "$SLURM_ARRAY_TASK_ID" \
        --n_jobs "$N_JOBS" \
        $FORCE_FLAG \
        $FIXED_VT_ARG

elif [ "$MODE" = "select_finalize" ]; then
    uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/04_combined_hdphmm.py" \
        --sub_id "$SUBJECT_ID" \
        --parcellation "$PARCELLATION" \
        --mode select_finalize \
        --n_final_seeds "$N_FINAL_SEEDS" \
        --n_jobs "$N_JOBS" \
        $FORCE_FLAG \
        $FIXED_VT_ARG

elif [ "$MODE" = "loso_fit" ]; then
    uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/04_combined_hdphmm.py" \
        --sub_id "$SUBJECT_ID" \
        --parcellation "$PARCELLATION" \
        --mode loso_fit \
        --loso_season "$SLURM_ARRAY_TASK_ID" \
        --n_final_seeds "$N_FINAL_SEEDS" \
        --n_jobs "$N_JOBS" \
        $FORCE_FLAG \
        $FIXED_VT_ARG

elif [ "$MODE" = "split_half_fit" ]; then
    # Map SLURM array task ID to half label: 0 -> A, 1 -> B
    HALF_LABEL=$( [ "${SLURM_ARRAY_TASK_ID}" = "0" ] && echo "A" || echo "B" )
    uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/04_combined_hdphmm.py" \
        --sub_id "$SUBJECT_ID" \
        --parcellation "$PARCELLATION" \
        --mode split_half_fit \
        --half "$HALF_LABEL" \
        --n_final_seeds "$N_FINAL_SEEDS" \
        --n_jobs "$N_JOBS" \
        $FORCE_FLAG \
        $FIXED_VT_ARG

else
    echo "ERROR: Unknown MODE '$MODE'. Use 'fit', 'select', 'select_seed', 'select_finalize', 'loso_fit', or 'split_half_fit'."
    exit 1
fi

EXIT_CODE=$?

# =============================================================================
# Report
# =============================================================================

echo ""
if [ $EXIT_CODE -ne 0 ]; then
    echo "=========================================="
    echo "ERROR: Failed (exit code $EXIT_CODE)"
    echo "=========================================="
    exit $EXIT_CODE
else
    echo "=========================================="
    echo "SUCCESS"
    echo "=========================================="
fi

echo ""
echo "========== SLURM Job Resource Usage =========="
seff $SLURM_JOB_ID 2>/dev/null || true
echo "==============================================="
