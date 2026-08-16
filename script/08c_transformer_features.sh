#!/bin/bash
#SBATCH --job-name=08c_transformer
#SBATCH --output=logs/08c_transformer_%A_%a.out
#SBATCH --error=logs/08c_transformer_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --array=0-291

# Shared preamble: PROJECT_DIR/SCRIPT_DIR, uv on PATH, logs/ (utils/_env.sh)
_ENV="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)/script/utils/_env.sh"
[ -f "$_ENV" ] || _ENV="${SLURM_SUBMIT_DIR:-.}/script/utils/_env.sh"
source "$_ENV" || { echo "ERROR: cannot locate script/utils/_env.sh — submit from the repo root" >&2; exit 1; }

# =============================================================================
# 08c - Transformer Feature Extraction (per-episode parallelization)
# =============================================================================
# Each SLURM array task processes ONE episode for ONE model.
# MODEL is passed via --export (required).
# SLURM array index maps to episode via stimulus-specific episode list file.
#
# Required exports:
#   MODEL          Model key (w2v-bert-2.0, dinov2-large, llama-3.2-3b)
#
# Optional exports:
#   STIMULUS              Dataset (default: friends)
#   DOWNLOAD_ONLY         Set to 1 for cache warmup (downloads model, no extraction)
#   WINDOW_TRS            LLaMA local-window span in TRs (default: 4)
#   OUTPUT_SUBDIR_SUFFIX  Suffix for the 08c output directory (default: empty)
#                         Used by the W-sweep to avoid colliding with the
#                         production path: e.g. "_sweep_w3" writes to
#                         08c_transformer_features_sweep_w3/...
#
# Prerequisites:
#   ffmpeg installed (for audio extraction from mkv)
#
# Usage (two-step: warmup + fan-out):
#   # Step 0: Download ALL models to shared HF cache (CPU, no GPU needed)
#   #   Can run on login node or as a CPU job:
#   HF_HOME=/path/to/hf_cache uv run python script/08c_transformer_features.py \
#       --model all --download_only --device cpu
#
#   # Step 1: Fan-out - 292 parallel GPU jobs per model
#   sbatch --export=MODEL=llama-3.2-3b --array=0-291 script/08c_transformer_features.sh
#   sbatch --export=MODEL=dinov2-large --array=0-291 script/08c_transformer_features.sh
#   sbatch --export=MODEL=w2v-bert-2.0 --array=0-291 script/08c_transformer_features.sh
#
#   # Movie10 text:
#   sbatch --export=STIMULUS=movie10,MODEL=llama-3.2-3b \
#       --array=0-60 script/08c_transformer_features.sh
#
#   # HP text:
#   sbatch --export=STIMULUS=harrypotter,MODEL=llama-3.2-3b \
#       --array=0-6 script/08c_transformer_features.sh
#
#   # PP French audio:
#   sbatch --export=STIMULUS=petitprince_fr,MODEL=w2v-bert-2.0 \
#       --array=0-8 script/08c_transformer_features.sh
#
#   # PP English audio:
#   sbatch --export=STIMULUS=petitprince_en,MODEL=w2v-bert-2.0 \
#       --array=0-8 script/08c_transformer_features.sh
# =============================================================================

set -e

module load ffmpeg/5.1.4
# Re-prepend after module load so user-local uv stays ahead of module bins
# (the shared preamble's PATH export ran before the module prepended its own).
export PATH="$HOME/.local/bin:$PATH"

# NOTE: Do NOT run `uv sync` here - it strips NVIDIA .so files (known uv bug).
# Run `uv sync --extra torch --extra gpu` once manually, then fix with:
#   uv pip install --reinstall nvidia-cudnn-cu13 nvidia-nccl-cu13

# NVIDIA libs installed by pip (cuDNN, cuBLAS, NCCL, etc.) - not on system LD path.
# Glob the python* dir so this works whichever interpreter uv provisions
# (requires-python allows 3.11 or 3.12).
NVIDIA_LD=""
for subdir in "${PROJECT_DIR}"/.venv/lib/python*/site-packages/nvidia/*/lib; do
    [ -d "$subdir" ] && NVIDIA_LD="${subdir}:${NVIDIA_LD}"
done
# FFmpeg shared libs (torchcodec needs libavutil.so etc. on LD_LIBRARY_PATH).
# Set FFMPEG_LIB to your ffmpeg lib dir (e.g. a spack/module install); if unset,
# torchcodec must find ffmpeg some other way (system libs, module load, etc.).
FFMPEG_LIB="${FFMPEG_LIB:-}"
export LD_LIBRARY_PATH="${FFMPEG_LIB:+${FFMPEG_LIB}:}${NVIDIA_LD}${LD_LIBRARY_PATH:-}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# Verify cudnn is findable (venv python, via uv)
uv run --project "${PROJECT_DIR}" --no-sync python -c "import ctypes; ctypes.CDLL('libcudnn.so.9')" 2>/dev/null && echo "cudnn: OK" || echo "cudnn: NOT FOUND"

# Shared HuggingFace cache (override HF_HUB_CACHE to point at a shared location)
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"

# --- Validate MODEL ---
if [ -z "${MODEL}" ]; then
    echo "ERROR: MODEL must be set via --export (e.g., --export=MODEL=llama-3.2-3b)"
    exit 1
fi

# Default stimulus
STIMULUS="${STIMULUS:-friends}"

# LLaMA local-window readout - see 2026-05-01_08c_llama_local_window_design.md
WINDOW_TRS="${WINDOW_TRS:-4}"
OUTPUT_SUBDIR_SUFFIX="${OUTPUT_SUBDIR_SUFFIX:-}"

# --- Map stimulus to episode list file ---
case "${STIMULUS}" in
    friends)        EPISODE_FILE="${PROJECT_DIR}/episode_ids.txt" ;;
    movie10)        EPISODE_FILE="${PROJECT_DIR}/movie10_episode_ids.txt" ;;
    harrypotter)    EPISODE_FILE="${PROJECT_DIR}/harrypotter_episode_ids.txt" ;;
    petitprince_fr) EPISODE_FILE="${PROJECT_DIR}/petitprince_fr_episode_ids.txt" ;;
    petitprince_en) EPISODE_FILE="${PROJECT_DIR}/petitprince_en_episode_ids.txt" ;;
    *)
        echo "ERROR: Unknown stimulus '${STIMULUS}'"
        exit 1
        ;;
esac

if [ ! -f "$EPISODE_FILE" ]; then
    echo "ERROR: Episode file '${EPISODE_FILE}' does not exist."
    exit 1
fi

# --- Map array index to episode run_id ---
LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 1))
RUN_ID=$(sed -n "${LINE_NUMBER}p" "$EPISODE_FILE")

if [ -z "$RUN_ID" ]; then
    echo "No episode found on line ${LINE_NUMBER} of '${EPISODE_FILE}'."
    echo "Task ${SLURM_ARRAY_TASK_ID} exiting gracefully (array larger than episode count)."
    exit 0
fi

# Audio cache directory (persist extracted WAVs)
AUDIO_CACHE="${SCRATCH_DIR:-/tmp}/08c_audio_cache/${STIMULUS}"
mkdir -p "${AUDIO_CACHE}"

# --- Download-only mode (for cache warmup) ---
DOWNLOAD_ARGS=""
if [ "${DOWNLOAD_ONLY}" = "1" ]; then
    DOWNLOAD_ARGS="--download_only"
fi

echo "=============================================="
echo "08c - Transformer Feature Extraction"
echo "=============================================="
echo "Stimulus:          ${STIMULUS}"
echo "Model:             ${MODEL}"
echo "Run ID:            ${RUN_ID}"
echo "Device:            cuda"
echo "Download:          ${DOWNLOAD_ONLY:-0}"
echo "Window TRs:        ${WINDOW_TRS}"
echo "Output suffix:     '${OUTPUT_SUBDIR_SUFFIX}'"
echo "=============================================="

uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/08c_transformer_features.py" \
    --stimulus "${STIMULUS}" \
    --model "${MODEL}" \
    --run_id "${RUN_ID}" \
    --device cuda \
    --audio_cache_dir "${AUDIO_CACHE}" \
    --window_trs "${WINDOW_TRS}" \
    --output_subdir_suffix "${OUTPUT_SUBDIR_SUFFIX}" \
    ${DOWNLOAD_ARGS}

echo "=============================================="
echo "Feature extraction complete!"
echo "=============================================="
