#!/bin/bash
#SBATCH --job-name=08f_cross_stim_per_state
#SBATCH --output=logs/08f_cross_stim_per_state_%A_%a.out
#SBATCH --error=logs/08f_cross_stim_per_state_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --array=0-5

# =============================================================================
# 08f — Per-state cross-stimulus signature consistency (D3c + D4-lang)
# =============================================================================
# Reads 08d D2 outputs for two stimuli and computes per-state Spearman ρ of
# layer-wise AUC profiles. No feature reloading.
#
# Required exports:
#   STIMULUS_A  — e.g. friends            (default: friends)
#   STIMULUS_B  — e.g. movie10            (default: movie10)
#   MODEL       — transformer model       (default: llama-3.2-3b)
#   VT          — 05e_a4 VT suffix        (default: 0.95)
#   D4_LANG     — if "1", add --d4_lang   (default: unset)
#
# Usage examples:
#   # D3c Friends vs Movie10
#   sbatch script/08f_transformer_cross_stim_per_state.sh
#
#   # D3c Friends vs Harry Potter (text model only)
#   sbatch --export=STIMULUS_B=harrypotter script/08f_transformer_cross_stim_per_state.sh
#
#   # D4-lang PP-FR vs PP-EN (w2v-bert)
#   sbatch --export=STIMULUS_A=petitprince_fr,STIMULUS_B=petitprince_en,MODEL=w2v-bert-2.0,D4_LANG=1 \
#       script/08f_transformer_cross_stim_per_state.sh
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

STIMULUS_A="${STIMULUS_A:-friends}"
STIMULUS_B="${STIMULUS_B:-movie10}"
MODEL="${MODEL:-llama-3.2-3b}"
VT="${VT:-0.95}"

SUBJECTS=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06)
SUB_ID="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"

# HP and PP datasets do not include sub-04 — this is a CNeuroMod data-provenance
# limitation, NOT a quality-control exclusion. Auto-skip so the uniform
# --array=0-5 works for every stimulus without re-launching with a trimmed array.
if [[ "${STIMULUS_A}" == "harrypotter" || "${STIMULUS_A}" == "petitprince_fr" || "${STIMULUS_A}" == "petitprince_en" \
   || "${STIMULUS_B}" == "harrypotter" || "${STIMULUS_B}" == "petitprince_fr" || "${STIMULUS_B}" == "petitprince_en" ]] \
   && [[ "${SUB_ID}" == "sub-04" ]]; then
    echo "Skipping ${SUB_ID} for ${STIMULUS_A}/${STIMULUS_B} (subject not in dataset — CNeuroMod data-provenance limitation, not QC)"
    exit 0
fi

echo "=============================================="
echo "08f — Per-state cross-stim consistency"
echo "=============================================="
echo "Subject:     ${SUB_ID}"
echo "Stimulus A:  ${STIMULUS_A}"
echo "Stimulus B:  ${STIMULUS_B}"
echo "Model:       ${MODEL}"
echo "D4-lang:     ${D4_LANG:-0}"
echo "=============================================="

CMD=(uv run --project "${PROJECT_DIR}" --no-sync python
     "${PROJECT_DIR}/script/08f_transformer_cross_stim_per_state.py"
     --sub_id "${SUB_ID}"
     --stimulus_a "${STIMULUS_A}"
     --stimulus_b "${STIMULUS_B}"
     --model "${MODEL}"
     --vt "${VT}")
if [ "${D4_LANG:-0}" = "1" ]; then
    CMD+=(--d4_lang)
fi

"${CMD[@]}"

echo "=============================================="
echo "08f complete!"
echo "=============================================="
