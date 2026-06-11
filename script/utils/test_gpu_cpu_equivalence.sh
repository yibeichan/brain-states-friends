#!/bin/bash
#SBATCH --job-name=gpu_cpu_equiv
#SBATCH --partition=ou_bcs_low,pi_satra
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

# =============================================================================
# GPU vs CPU Equivalence Test — SLURM Wrapper
# =============================================================================
#
# Usage:
#   sbatch script/utils/test_gpu_cpu_equivalence.sh
#   sbatch --export=SUB_ID=sub-02,SEED=0 script/utils/test_gpu_cpu_equivalence.sh
#   sbatch --export=N_ITER=200 script/utils/test_gpu_cpu_equivalence.sh  # quick 200-iter check
# =============================================================================

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &> /dev/null && pwd)
fi

mkdir -p "${PROJECT_DIR}/logs"

eval "$(micromamba shell hook --shell bash)"
micromamba activate friends-states

# JAX configuration
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

SUB_ID=${SUB_ID:-"sub-01"}
PARCELLATION=${PARCELLATION:-"atlas-4S156Parcels"}
CONFIG_NAME=${CONFIG_NAME:-"vt0.80_covdiag_nc60_g5"}
SEED=${SEED:-0}
N_ITER=${N_ITER:-""}

echo "=========================================="
echo "GPU vs CPU Equivalence Test"
echo "=========================================="
echo "SLURM Job ID:  $SLURM_JOB_ID"
echo "Subject:       $SUB_ID"
echo "Config:        $CONFIG_NAME"
echo "Seed:          $SEED"
echo "N_ITER:        ${N_ITER:-convergence}"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "=========================================="

N_ITER_ARG=""
if [ -n "$N_ITER" ]; then
    N_ITER_ARG="--n_iter $N_ITER"
fi

uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/utils/test_gpu_cpu_equivalence.py" \
    --sub_id "$SUB_ID" \
    --parcellation "$PARCELLATION" \
    --config_name "$CONFIG_NAME" \
    --seed "$SEED" \
    $N_ITER_ARG

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -ne 0 ]; then
    echo "=========================================="
    echo "FAILED (exit code $EXIT_CODE)"
    echo "=========================================="
else
    echo "=========================================="
    echo "SUCCESS"
    echo "=========================================="
fi

echo ""
seff $SLURM_JOB_ID 2>/dev/null || true
