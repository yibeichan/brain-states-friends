#!/bin/bash
#SBATCH --job-name=datalad_save
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

# =============================================================================
# DataLad Save & Push - Save pipeline outputs and sync to persistent storage
# =============================================================================
#
# Usage (manual):
#   bash script/utils/datalad_save.sh --stage 04 --subject sub-01
#   bash script/utils/datalad_save.sh --stage 04 --subject sub-01 --no-push
#   bash script/utils/datalad_save.sh --stage 04 --subject sub-01 --parcellation atlas-4S456Parcels
#
# Usage (SLURM dependency):
#   JOB=$(sbatch --parsable --export=SUBJECT_ID=sub-01,MODE=select script/04_combined_hdphmm.sh)
#   sbatch --dependency=afterok:$JOB --export=STAGE=04,SUBJECT_ID=sub-01 script/utils/datalad_save.sh
#
# Usage (save all subjects for a stage):
#   bash script/utils/datalad_save.sh --stage 05a --subject all
#
# =============================================================================

set -euo pipefail

# --- Parse arguments (support both --flag and SLURM env vars) ---
STAGE="${STAGE:-}"
SUBJECT_ID="${SUBJECT_ID:-}"
PARCELLATION="${PARCELLATION:-atlas-4S156Parcels}"
PUSH="${PUSH:-true}"
MESSAGE="${MESSAGE:-}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --stage) STAGE="$2"; shift 2 ;;
        --subject) SUBJECT_ID="$2"; shift 2 ;;
        --parcellation) PARCELLATION="$2"; shift 2 ;;
        --message) MESSAGE="$2"; shift 2 ;;
        --push) PUSH="true"; shift ;;
        --no-push) PUSH="false"; shift ;;
        -h|--help)
            echo "Usage: datalad_save.sh --stage STAGE --subject SUBJECT_ID [--parcellation PARC] [--no-push] [--message MSG]"
            echo ""
            echo "Stages: 00, 02, 03a, 03b, 04, 05a, 05b, 05c, 05d, 05e_a1, 05e_a2, 06, m10_03, m10_04, hp_04, pp_04, diag"
            echo "Subject: sub-01 through sub-06, or 'all'"
            exit 0
            ;;
        *) echo "ERROR: Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$STAGE" ] || [ -z "$SUBJECT_ID" ]; then
    echo "ERROR: --stage and --subject are required"
    echo "Run with --help for usage"
    exit 1
fi

# --- Determine directories ---
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    CODE_DIR="$SLURM_SUBMIT_DIR"
else
    CODE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &> /dev/null && pwd)
fi

SCRATCH_DIR=$(grep SCRATCH_DIR "${CODE_DIR}/.env" | cut -d'=' -f2 | tr -d '"' | tr -d "'")
OUTPUT_DIR="${SCRATCH_DIR}/output"

if [ ! -d "$OUTPUT_DIR/.datalad" ]; then
    echo "ERROR: ${OUTPUT_DIR} is not a DataLad dataset. Run 'datalad create' first."
    exit 1
fi

# --- Map stage to output subdirectory ---
declare -A STAGE_MAP=(
    ["00"]="00_postproc"
    ["02"]="02_parcel_ts_avg"
    ["03a"]="03a_pca4combined_hmm"
    ["03b"]="03b_pca_loadings"
    ["04"]="04_combined_hdphmm"
    ["04ra"]="04ra_loso_struct_comp"
    ["04rb"]="04rb_split_half"
    ["04rv"]="04rv_reliability_vis"
    ["05a"]="05a_recurrence_analysis"
    ["05b"]="05b_recurring_states_visualization"
    ["05c"]="05c_episode_decodability"
    ["05d"]="05d_state_similarity"
    ["05e_a1"]="05e_temporal_trend_a1"
    ["05e_a2"]="05e_temporal_trend_a2"
    ["05e_a3"]="05e_temporal_trend_a3"
    ["05e_a4"]="05e_temporal_trend_a4"
    ["05f"]="05f_state_fc"
    ["06"]="06a_state_temp_dynamics"
    ["06b"]="06b_transition_structure"
    ["06c"]="06c_higher_order_transitions"
    ["06d"]="06d_preserved_chains"
    ["08c"]="08c_transformer_features"
    ["08c_w1"]="08c_transformer_features_sweep_w1"
    ["08c_w3"]="08c_transformer_features_sweep_w3"
    ["08c_w6"]="08c_transformer_features_sweep_w6"
    ["08c_w9"]="08c_transformer_features_sweep_w9"
    ["08c_legacy"]="08c_transformer_features_legacy_cumulative_pool"
    ["08d"]="08d_transformer_depth"
    ["08d_w1"]="08d_transformer_depth_sweep_w1"
    ["08d_w3"]="08d_transformer_depth_sweep_w3"
    ["m10_03"]="m10_03_projected"
    ["m10_04"]="m10_04_decoded"
    ["m10_05"]="m10_05_cross_validation"
    ["hp_03"]="hp_03_projected"
    ["hp_04"]="hp_04_decoded"
    ["hp_05"]="hp_05_cross_validation"
    ["pp_03"]="pp_03_projected"
    ["pp_04"]="pp_04_decoded"
    ["pp_05"]="pp_05_cross_validation"
    ["pp_annotations"]="pp_annotations"
    ["manuscript_figures"]="manuscript_figures"
    ["diag"]="diagnostics"
)

STAGE_DIR="${STAGE_MAP[$STAGE]:-}"
if [ -z "$STAGE_DIR" ]; then
    echo "ERROR: Unknown stage '$STAGE'. Valid: ${!STAGE_MAP[*]}"
    exit 1
fi

# --- Build save path(s) ---
# Stages without parcellation level (output is {stage}/{sub}/)
if [ "$STAGE" = "00" ]; then
    if [ "$SUBJECT_ID" = "all" ]; then
        SAVE_PATH="${STAGE_DIR}/"
    else
        SAVE_PATH="${STAGE_DIR}/${SUBJECT_ID}/"
    fi
elif [ "$SUBJECT_ID" = "all" ]; then
    SAVE_PATH="${STAGE_DIR}/${PARCELLATION}/"
else
    SAVE_PATH="${STAGE_DIR}/${PARCELLATION}/${SUBJECT_ID}/"
fi

if [ ! -d "${OUTPUT_DIR}/${SAVE_PATH}" ]; then
    echo "ERROR: Directory does not exist: ${OUTPUT_DIR}/${SAVE_PATH}"
    exit 1
fi

# --- Get code provenance ---
CODE_COMMIT=$(git -C "${CODE_DIR}" rev-parse HEAD)
CODE_BRANCH=$(git -C "${CODE_DIR}" rev-parse --abbrev-ref HEAD)
CODE_SHORT=$(git -C "${CODE_DIR}" rev-parse --short HEAD)

# --- Get input data provenance ---
DATA_DIR=$(grep DATA_DIR "${CODE_DIR}/.env" | cut -d'=' -f2 | tr -d '"' | tr -d "'")
INPUT_DIR="${DATA_DIR}/cneuromod.processed/fmriprep/friends"
INPUT_COMMIT=""
if [ -d "${INPUT_DIR}/.git" ] || [ -d "${INPUT_DIR}/.datalad" ]; then
    INPUT_COMMIT=$(git -C "${INPUT_DIR}" rev-parse HEAD 2>/dev/null || echo "unknown")
fi

SLURM_INFO=""
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SLURM_INFO="slurm_job_id: ${SLURM_JOB_ID}"
fi

# --- Build commit message ---
INPUT_LINE=""
if [ -n "$INPUT_COMMIT" ]; then
    INPUT_LINE="input_data: cneuromod.processed/fmriprep/friends@${INPUT_COMMIT}"
fi

if [ -z "$MESSAGE" ]; then
    COMMIT_MSG="[${STAGE}] ${SUBJECT_ID} ${PARCELLATION}

code_repo: yibeichan/brain-states-friends
code_commit: ${CODE_COMMIT}
code_branch: ${CODE_BRANCH}
${INPUT_LINE}
${SLURM_INFO}"
else
    COMMIT_MSG="${MESSAGE}

code_repo: yibeichan/brain-states-friends
code_commit: ${CODE_COMMIT}
code_branch: ${CODE_BRANCH}
${INPUT_LINE}
${SLURM_INFO}"
fi

# --- Environment ---
# Ensure the user-local uv install is on PATH (SLURM jobs may not inherit it)
export PATH="$HOME/.local/bin:$PATH"
# datalad is an optional extra (pyproject.toml); --project resolves the repo venv
# even though we run from OUTPUT_DIR (the DataLad dataset).
DATALAD="uv run --project ${CODE_DIR} --extra datalad datalad"

# datalad shells out to git-annex, a Haskell binary that uv/PyPI cannot provide.
# It must be fetched once into user space (no root) with datalad-installer.
if ! command -v git-annex &> /dev/null; then
    echo "ERROR: git-annex not found on PATH. datalad needs it and uv cannot install it."
    echo "Fetch it once (no root) with datalad-installer, then re-run this job:"
    echo "  uv run --project ${CODE_DIR} --extra datalad datalad-installer \\"
    echo "    -E ~/.local/share/git-annex-env.sh \\"
    echo "    git-annex -m datalad/git-annex:release --install-dir ~/.local"
    echo "  source ~/.local/share/git-annex-env.sh   # or add its PATH line to ~/.bashrc"
    exit 1
fi

# --- Save ---
cd "${OUTPUT_DIR}"
echo "Saving ${SAVE_PATH} ..."
echo "Code: ${CODE_BRANCH}@${CODE_SHORT}"

${DATALAD} save -m "${COMMIT_MSG}" "${SAVE_PATH}"
echo "Saved."

# --- Push ---
if [ "$PUSH" = "true" ]; then
    echo "Pushing to ria-storage ..."
    ${DATALAD} push --to ria-storage
    echo "Pushed."
fi

echo "Done."
