#!/bin/bash
#SBATCH --job-name=04_combined_hdphmm_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-4          # Stage 1 default: 5 vt configs; override per stage
#SBATCH --time=07:00:00      # GPU fits: most complete in 2-5h, some up to 6.5h
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

# Shared preamble: PROJECT_DIR/SCRIPT_DIR, uv on PATH, logs/ (utils/_env.sh)
source "${SLURM_SUBMIT_DIR:-.}/script/utils/_env.sh" 2>/dev/null || source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/script/utils/_env.sh" || exit 1

# =============================================================================
# Combined weak-limit HMM - GPU SLURM Wrapper (JAX backend)
# =============================================================================
#
# Same interface as 04_combined_hdphmm.sh but requests a GPU and uses JAX.
# The Python script auto-detects JAX+GPU and switches to the JAX backend.
#
# Usage is identical to the CPU version:
#   sbatch --export=SUBJECT_ID=sub-01,STAGE=1 script/04_combined_hdphmm_gpu.sh
#   sbatch --array=0-11 --export=SUBJECT_ID=sub-01,STAGE=2,FIXED_VT=0.95 script/04_combined_hdphmm_gpu.sh
#   sbatch --array=0 --export=SUBJECT_ID=sub-01,MODE=select,FIXED_VT=0.95 --time=02:00:00 script/04_combined_hdphmm_gpu.sh
#   sbatch --array=1-6 --export=SUBJECT_ID=sub-01,MODE=loso_fit --time=04:00:00 script/04_combined_hdphmm_gpu.sh
#   sbatch --array=0-1 --export=SUBJECT_ID=sub-01,MODE=split_half_fit --time=04:00:00 script/04_combined_hdphmm_gpu.sh
#
# NOTES ON GPU RESOURCES:
#   Stage 1/2 fit:  --time=04:00:00, --mem=32G, --gres=gpu:1
#   select:         --time=02:00:00, --mem=32G, --gres=gpu:1
#   loso_fit:       --time=04:00:00, --mem=32G, --gres=gpu:1
#
# To request a specific GPU type:
#   sbatch --gres=gpu:h100:1 ...
#   sbatch --gres=gpu:a100:1 ...
# =============================================================================


# JAX configuration
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # avoid OOM on shared GPUs
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8    # limit to 80% GPU memory

# =============================================================================
# Configuration (override with --export on sbatch command line)
# =============================================================================

SUBJECT_ID=${SUBJECT_ID:-"sub-01"}
PARCELLATION=${PARCELLATION:-"atlas-4S156Parcels"}
MODE=${MODE:-"fit"}
STAGE=${STAGE:-1}
FIXED_VT=${FIXED_VT:-""}
N_FIT_SEEDS=${N_FIT_SEEDS:-5}
N_FINAL_SEEDS=${N_FINAL_SEEDS:-10}
N_JOBS=${N_JOBS:-1}  # GPU: no joblib parallelism needed, JAX handles it

# Map short parcellation names to full names
case "$PARCELLATION" in
    156)  PARCELLATION="atlas-4S156Parcels" ;;
    456)  PARCELLATION="atlas-4S456Parcels" ;;
    1056) PARCELLATION="atlas-4S1056Parcels" ;;
    *)    ;;
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

if [ "$MODE" = "fit" ] && [ "$STAGE" = "2" ] && [ -z "${FIXED_VT}" ]; then
    if [ ! -f "${OUTPUT_DIR}/stage1_result.json" ]; then
        echo "ERROR: Stage 1 result not found at ${OUTPUT_DIR}/stage1_result.json"
        echo "Run Stage 1 fit + select first, or set FIXED_VT=0.95 to bypass Stage 1."
        exit 1
    fi
fi

# loso_fit / split_half_fit require final results from select mode
if [ "$MODE" = "loso_fit" ] || [ "$MODE" = "split_half_fit" ]; then
    if [ -n "${FIXED_VT}" ]; then
        _VT="${FIXED_VT}"
    elif [ -f "${OUTPUT_DIR}/stage1_result.json" ]; then
        _VT=$(uv run --project "${PROJECT_DIR}" --no-sync python3 -c "import json; print(json.load(open('${OUTPUT_DIR}/stage1_result.json'))['selected_vt'])")
    else
        echo "ERROR: Cannot determine vt for ${MODE}."
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
echo "04 Combined weak-limit HMM (GPU/JAX)"
echo "=========================================="
echo "SLURM Job ID:     $SLURM_JOB_ID"
echo "Array Task ID:    ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Subject:          $SUBJECT_ID"
echo "Parcellation:     $PARCELLATION"
echo "Mode:             $MODE"

# Report GPU info
echo "GPU:              $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

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

SELECT_CONFIG_ARG=""
if [ -n "${SELECT_CONFIG}" ]; then
    SELECT_CONFIG_ARG="--select_config ${SELECT_CONFIG}"
fi

if [ "$MODE" = "fit" ]; then
    uv run --project "${PROJECT_DIR}" --extra gpu python "${PROJECT_DIR}/script/04_combined_hdphmm.py" \
        --sub_id "$SUBJECT_ID" \
        --parcellation "$PARCELLATION" \
        --mode fit \
        --stage "$STAGE" \
        --task_index "$SLURM_ARRAY_TASK_ID" \
        --n_fit_seeds "$N_FIT_SEEDS" \
        --n_jobs "$N_JOBS" \
        $FIXED_VT_ARG

elif [ "$MODE" = "select" ]; then
    uv run --project "${PROJECT_DIR}" --extra gpu python "${PROJECT_DIR}/script/04_combined_hdphmm.py" \
        --sub_id "$SUBJECT_ID" \
        --parcellation "$PARCELLATION" \
        --mode select \
        --n_final_seeds "$N_FINAL_SEEDS" \
        --n_jobs "$N_JOBS" \
        $FORCE_FLAG \
        $FIXED_VT_ARG \
        $SELECT_CONFIG_ARG

elif [ "$MODE" = "select_seed" ]; then
    uv run --project "${PROJECT_DIR}" --extra gpu python "${PROJECT_DIR}/script/04_combined_hdphmm.py" \
        --sub_id "$SUBJECT_ID" \
        --parcellation "$PARCELLATION" \
        --mode select_seed \
        --seed_index "$SLURM_ARRAY_TASK_ID" \
        --n_jobs "$N_JOBS" \
        $FORCE_FLAG \
        $FIXED_VT_ARG \
        $SELECT_CONFIG_ARG

elif [ "$MODE" = "select_finalize" ]; then
    uv run --project "${PROJECT_DIR}" --extra gpu python "${PROJECT_DIR}/script/04_combined_hdphmm.py" \
        --sub_id "$SUBJECT_ID" \
        --parcellation "$PARCELLATION" \
        --mode select_finalize \
        --n_final_seeds "$N_FINAL_SEEDS" \
        --n_jobs "$N_JOBS" \
        $FORCE_FLAG \
        $FIXED_VT_ARG \
        $SELECT_CONFIG_ARG

elif [ "$MODE" = "loso_fit" ]; then
    uv run --project "${PROJECT_DIR}" --extra gpu python "${PROJECT_DIR}/script/04_combined_hdphmm.py" \
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
    uv run --project "${PROJECT_DIR}" --extra gpu python "${PROJECT_DIR}/script/04_combined_hdphmm.py" \
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
