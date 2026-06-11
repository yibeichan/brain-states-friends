#!/bin/bash
#SBATCH --job-name=08g_transformer_convergence
#SBATCH --output=logs/08g_transformer_convergence_%A_%a.out
#SBATCH --error=logs/08g_transformer_convergence_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --array=0-5

# =============================================================================
# 08g — Transformer convergence analyses
# =============================================================================
# D5 (08b ↔ 08d per-state), cross-modality dissociation (Friends/Movie10),
# and recurrence × depth interaction. Depends on 08b and 08d outputs already
# existing for the relevant (subject, stimulus, model) combinations.
#
# Required exports:
#   MODELS   — space-separated model list (default: "w2v-bert-2.0 dinov2-large llama-3.2-3b")
#   VT       — 05e_a4 VT suffix (default: 0.95)
#   N_PERMS  — permutation count for D5 (default: 1000)
# =============================================================================

set -e

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
fi

mkdir -p "${PROJECT_DIR}/logs"

eval "$(micromamba shell hook --shell bash)"
micromamba activate friends-states

MODELS="${MODELS:-w2v-bert-2.0 dinov2-large llama-3.2-3b}"
VT="${VT:-0.95}"
N_PERMS="${N_PERMS:-1000}"

SUBJECTS=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06)
SUB_ID="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"

echo "=============================================="
echo "08g — Transformer convergence"
echo "=============================================="
echo "Subject:   ${SUB_ID}"
echo "Models:    ${MODELS}"
echo "VT:        ${VT}"
echo "N perms:   ${N_PERMS}"
echo "=============================================="

uv run --project "${PROJECT_DIR}" --no-sync python \
    "${PROJECT_DIR}/script/08g_transformer_convergence.py" \
    --sub_id "${SUB_ID}" \
    --vt "${VT}" \
    --n_permutations "${N_PERMS}" \
    --models ${MODELS}

echo "=============================================="
echo "08g complete!"
echo "=============================================="
