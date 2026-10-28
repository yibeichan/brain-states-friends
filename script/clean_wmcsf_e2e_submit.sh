#!/bin/bash
#
# Submit the CNeuroMod clean-vs-corrupt CSF/WM end-to-end check: re-clean,
# parcellate, PCA and refit the combined HMM (full two-stage selection) for one or
# more subjects, once with the shared CSF/WM table's corrupt-mask regressors and
# once with its clean ones, then compare the two fits (clean_wmcsf_compare.py).
# Both variants use the same rebuild recipe, so their difference isolates the
# trilinear tissue-map defect.
#
# Each variant gets its own SCRATCH_DIR (RUN_ROOT/<variant>), so every stage reads
# and writes RUN_ROOT/<variant>/output/... and never touches the main outputs.
# Stage 2 and the final refit use FIXED_VT=0.95, as the published fits did. Stage 1
# still runs so the compare can report its vt pick; stage 2 leaves out the config
# that stage 1 also fits (the same directory), so no two tasks write it at once.
#
# Chain per subject and variant: 00 -> 02 -> 03a -> 04 stage 1 and stage 2, each
# as up to FIT_ATTEMPTS attempts of 48 h -> fit check -> select_seed (one task per
# final seed) -> select_finalize. Once both variants of a subject are final, its
# compare runs and writes RUN_ROOT/compare/<sub>.json.
#
# Fit attempts: a fit task runs its config's 5 seeds one after another and saves
# each seed when it finishes; a later attempt skips saved seeds and exits in
# seconds when none is left. Measured on the 2026-10 run, one config needed up to
# 75 h (nc100), against pi_satra's 48 h maximum, so attempts are linked with
# afterany and clean_wmcsf_check_fits.py, run after the last one, stops the chain
# unless every stage-1 and stage-2 config has all its seeds.
#
# Usage (from the repo root of a frozen worktree with its own .venv and .env):
#   WMCSF_TABLE=<table dir> RUN_ROOT=<new absolute dir> \
#       bash script/clean_wmcsf_e2e_submit.sh sub-03
# (one subject per call: the per-user submit limit holds about one subject's chain)
#
# Every check runs before the first sbatch. RUN_ROOT must be absolute (a symlink is
# resolved), and each variant root and its output/ must be real directories, all
# outside the home directory and the main output dataset; each subject is given
# once; the jobs must fit under the partition's per-user submit limit together with
# what is already queued (one subject is about 160 array tasks); and each subject
# and variant claims RUN_ROOT/<variant>/.claim_<sub> with mkdir, so a second
# submission for the same subject is refused.
#
# Recovery: if the chain stops before the fits (00, 02 or 03a), use a new RUN_ROOT,
# since stage 02 skips parcel files that already exist. If it stops in the fits
# (the fit check lists the incomplete configs), the same root can be reused, but
# not through this script, which refuses a root that already holds fits: submit by
# hand, with the same options as below, the fit arrays (MODE=fit, STAGE=1 and 2),
# then clean_wmcsf_check_fits.py, select_seed, select_finalize and the compare, in
# that order. Fit tasks resume from their saved seeds; a seed that raised is saved
# as failed and skipped on resume, so delete its seeds/seed_<i>.json to retry it.
#
# Jobs inherit the calling shell's environment except the variables the stages read
# as options or inputs (STAGE_VARS); those come only from this script or from the
# worktree's .env. UV_PROJECT_ENVIRONMENT is among them, so jobs use ./.venv.
#
# Partition: pi_satra, --no-requeue. Stage 00 overwrites its outputs and 02 skips
# existing files without atomic writes, so neither is safe to requeue mid-run.
# Limits (measured on the 2026-10 run): stage 00 took minutes per subject; a stage-1
# config up to 20 h and a stage-2 config up to 75 h over its 5 seeds; one final
# seed 5-32 h, depending on the node.

set -euo pipefail
shopt -s inherit_errexit   # a failed sbatch inside $(submit ...) stops the script
MAX_SUBMIT=${MAX_SUBMIT:-256}   # pi_satra QOS MaxSubmitJobsPerUser; array tasks count

: "${WMCSF_TABLE:?set WMCSF_TABLE to the clean CSF/WM table directory}"
: "${RUN_ROOT:?set RUN_ROOT to a new directory for this check}"
[ "$#" -ge 1 ] || { echo "usage: $0 sub-XX [sub-YY ...]" >&2; exit 1; }

SUBJECTS=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06)
VARIANTS=(corrupt clean)
PARCELLATION=atlas-4S156Parcels
FIXED_VT=0.95
N_FIT_SEEDS=5
N_FINAL_SEEDS=10
FIT_ATTEMPTS=3
COMMON=(--parsable --partition=pi_satra --no-requeue --mail-type=NONE)
UV_PY="PATH=\$HOME/.local/bin:\$PATH uv run --project ${PWD} --no-sync python"
# Options and inputs the stage wrappers and scripts read from the environment.
# BASE_DIR, DATA_DIR and ATLAS_DIR then come from ./.env, which the stages load.
STAGE_VARS=(SCRATCH_DIR SUBJECT_ID PARCELLATION MODE STAGE FIXED_VT N_FIT_SEEDS
            N_FINAL_SEEDS N_JOBS FORCE_REFIT SPLIT_MODE WMCSF_TABLE WMCSF_VARIANT
            BASE_DIR DATA_DIR ATLAS_DIR UV_PROJECT_ENVIRONMENT UV_PYTHON VIRTUAL_ENV
            PYTHONPATH)
UNSET=()
for v in "${STAGE_VARS[@]}"; do UNSET+=(-u "$v"); done

die() { echo "ERROR: $*" >&2; exit 1; }

# submit VAR=value ... -- <sbatch args>: sbatch with only these stage options set.
# Job ids also go to SUBMIT_LOG (created just before submission), since submit runs
# in command substitutions.
submit() {
    local vars=() id
    while [ "$1" != "--" ]; do vars+=("$1"); shift; done
    shift
    id=$(env "${UNSET[@]}" "${vars[@]}" sbatch "${COMMON[@]}" --export=ALL "$@")
    [[ "${id}" =~ ^[0-9]+$ ]] || { echo "ERROR: sbatch returned '${id}'" >&2; return 1; }
    echo "${id}" >> "${SUBMIT_LOG}"
    echo "${id}"
}

# --- checks (nothing is submitted until all pass) ---------------------------------

[ -f script/00_postproc.sh ] || die "run from the repo root"
[ -f "${WMCSF_TABLE}/manifest.tsv" ] || die "no manifest.tsv in ${WMCSF_TABLE}"
[ -f .env ] || die "no .env in $(pwd); the stages read BASE_DIR, DATA_DIR and ATLAS_DIR from it"
command -v python3 > /dev/null || die "python3 is not on PATH"
[ -x .venv/bin/python ] || die "no .venv in $(pwd); jobs run 'uv run --no-sync' against it"
.venv/bin/python -c "import numpy, scipy, sklearn, nibabel, nilearn, dotenv" 2> /dev/null \
    || die ".venv in $(pwd) lacks the pipeline's packages (run uv sync there first)"

env_value() {  # the value of KEY in ./.env, quotes and a trailing comment removed
    sed -n "s/^[[:space:]]*\(export[[:space:]]\{1,\}\)\{0,1\}$1=//p" .env | tail -n 1 \
        | sed 's/[[:space:]]#.*$//' | tr -d "\"'" | sed 's/[[:space:]]*$//'
}
for key in BASE_DIR DATA_DIR ATLAS_DIR SCRATCH_DIR; do
    [ -n "$(env_value "${key}")" ] || die "no ${key} line in ./.env"
done

[[ "${RUN_ROOT}" = /* ]] || die "RUN_ROOT must be an absolute path, got ${RUN_ROOT}"
run_root=$(realpath -m "${RUN_ROOT}")
# Both paths are spliced unquoted into --wrap commands.
for p in "${run_root}" "${PWD}"; do
    [[ "$p" =~ ^[A-Za-z0-9._/+-]+$ ]] || die "path has characters other than letters, digits and ._/+-: $p"
done
home=$(realpath -m "${HOME}")
prod_outputs=("$(realpath -m "$(env_value SCRATCH_DIR)")/output")
[ -n "${SCRATCH_DIR:-}" ] && prod_outputs+=("$(realpath -m "${SCRATCH_DIR}")/output")

check_root() {  # a path the chain writes under must stay out of home and the main outputs
    case "$1/" in
        "${home}/"*) die "$1 is inside the home directory (${home}); a chain writes hundreds of GB" ;;
    esac
    for out in "${prod_outputs[@]}"; do
        case "$1/output/" in "${out}/"*) die "$1/output would be the main output dataset ${out}" ;; esac
        case "$1/" in "${out}/"*) die "$1 is inside the main output dataset ${out}" ;; esac
    done
}
check_root "${run_root}"

declare -A seen
for sub in "$@"; do
    case " ${SUBJECTS[*]} " in *" ${sub} "*) ;; *) die "unknown subject ${sub}" ;; esac
    [ -z "${seen[$sub]:-}" ] || die "${sub} is given twice"
    seen[$sub]=1
done

grid=$(python3 -B - "${FIXED_VT}" <<'EOF' || true
import sys
sys.path.insert(0, "script/config")
import combined_hmm_config as c
s1 = [c.config_name(x) for x in c.build_stage1_grid()]
s2 = [c.config_name(x) for x in c.build_stage2_grid(float(sys.argv[1]))]
print(len(s1), len(s2), ",".join(str(i) for i, n in enumerate(s2) if n in s1) or "-")
EOF
)
read -r N_STAGE1 N_STAGE2 STAGE2_SKIP <<< "${grid}" || true
[[ "${N_STAGE1:-}" =~ ^[0-9]+$ && "${N_STAGE2:-}" =~ ^[0-9]+$ ]] \
    || die "could not read the stage-1/stage-2 grids from script/config"
STAGE2_TASKS=()
for ((i = 0; i < N_STAGE2; i++)); do
    [[ ",${STAGE2_SKIP}," == *",${i},"* ]] || STAGE2_TASKS+=("$i")
done
STAGE2_ARRAY=$(IFS=,; echo "${STAGE2_TASKS[*]}")

# Array tasks this call submits, against the per-user submit limit.
per_variant=$((3 + FIT_ATTEMPTS * (N_STAGE1 + ${#STAGE2_TASKS[@]}) + 1 + N_FINAL_SEEDS + 1))
needed=$(( $# * (2 * per_variant + 1) ))
queue=$(squeue -h -r -u "$(id -un)" -p pi_satra) \
    || die "squeue failed; cannot check the submit limit"
queued=$(printf '%s' "${queue}" | grep -c . || true)
(( queued + needed <= MAX_SUBMIT )) || die "this needs ${needed} pi_satra jobs (array tasks) and ${queued} are queued, over the limit of ${MAX_SUBMIT}; submit fewer subjects per call"

claimed=()
release() { for c in "${claimed[@]}"; do rmdir "$c"; done; }
for sub in "$@"; do
    for variant in "${VARIANTS[@]}"; do
        root="${run_root}/${variant}"
        for d in "${root}" "${root}/output" "${root}/logs"; do
            if [ "$(realpath -m "${d}")" != "${d}" ]; then
                release; die "${d} resolves to $(realpath -m "${d}"); it must be a real directory"
            fi
        done
        check_root "${root}"
        mkdir -p "${root}/logs"
        if [ -e "${root}/output/04_combined_hdphmm/${PARCELLATION}/${sub}" ]; then
            release; die "${root} already holds ${sub} fits; use a new RUN_ROOT"
        fi
        if ! mkdir "${root}/.claim_${sub}" 2>/dev/null; then
            release
            die "${root}/.claim_${sub} exists: a chain for ${sub} was already submitted here; use a new RUN_ROOT"
        fi
        claimed+=("${root}/.claim_${sub}")
    done
done
mkdir -p "${run_root}/compare/logs"

# If submission stops partway (a failed sbatch or a signal), say what is already
# queued, or how to check when no job id was recorded.
SUBMIT_LOG=$(mktemp)
on_exit() {
    local st=$1 ids
    ids=$(tr "\n" " " < "${SUBMIT_LOG}"); rm -f "${SUBMIT_LOG}"
    [ "${st}" -eq 0 ] && return
    if [ -n "${ids}" ]; then
        echo "ERROR: submission stopped partway; already queued: ${ids}" >&2
        echo "Cancel them with: scancel ${ids}; then use a new RUN_ROOT." >&2
    else
        # An sbatch interrupted mid-call may have queued a job whose id was lost,
        # so the claims stay until squeue shows nothing from this call.
        echo "ERROR: submission stopped before any job id was recorded. Check" >&2
        echo "'squeue -u $(id -un) -n <job name>' for wcsf-* jobs; if there are none," >&2
        echo "remove ${claimed[*]:-the claims} and rerun." >&2
    fi
}
trap 'on_exit $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# fit_attempts <stage> <array> <first dependency> VAR=... : FIT_ATTEMPTS fit jobs,
# with the given stage options,
# each after the previous one ends in any state; prints the last job id.
fit_attempts() {
    local stage=$1 array=$2 dep=$3 a id
    shift 3
    for ((a = 1; a <= FIT_ATTEMPTS; a++)); do
        id=$(submit "$@" MODE=fit STAGE="${stage}" N_FIT_SEEDS="${N_FIT_SEEDS}" -- \
            --job-name="${tag}-s${stage}a${a}" --dependency="${dep}" \
            --array="${array}" --cpus-per-task=8 --mem=32G --time=48:00:00 \
            --output="${logs}/04s${stage}_%A_%a.out" --error="${logs}/04s${stage}_%A_%a.err" \
            script/04_combined_hdphmm.sh)
        dep="afterany:${id}"
    done
    echo "${id}"
}

# --- submission --------------------------------------------------------------------

echo "stage 1: ${N_STAGE1} configs; stage 2: tasks ${STAGE2_ARRAY} (skips ${STAGE2_SKIP}, fit by stage 1)"
for sub in "$@"; do
    idx=-1
    for i in "${!SUBJECTS[@]}"; do [ "${SUBJECTS[$i]}" = "$sub" ] && idx=$i; done
    fin=()
    for variant in "${VARIANTS[@]}"; do
        root="${run_root}/${variant}"
        logs="${root}/logs"
        base=(SCRATCH_DIR="${root}" SUBJECT_ID="${sub}" PARCELLATION="${PARCELLATION}")
        tag="wcsf-${variant:0:2}-${sub#sub-}"

        j00=$(submit "${base[@]}" WMCSF_TABLE="${WMCSF_TABLE}" WMCSF_VARIANT="${variant}" -- \
            --job-name="${tag}-00" --array="${idx}" \
            --cpus-per-task=8 --mem=48G --time=12:00:00 \
            --output="${logs}/00_%A_%a.out" --error="${logs}/00_%A_%a.err" \
            script/00_postproc.sh)

        j02=$(submit "${base[@]}" -- --job-name="${tag}-02" --dependency="afterok:${j00}" \
            --cpus-per-task=2 --mem=16G --time=08:00:00 \
            --output="${logs}/02_%j.out" --error="${logs}/02_%j.err" \
            --wrap="${UV_PY} ${PWD}/script/02_extract_parcel_ts.py --subject_id ${sub} --parcellation ${PARCELLATION}")

        j03=$(submit "${base[@]}" -- --job-name="${tag}-03a" --dependency="afterok:${j02}" \
            --array="${idx}" --cpus-per-task=1 --mem=16G --time=01:00:00 \
            --output="${logs}/03a_%A_%a.out" --error="${logs}/03a_%A_%a.err" \
            script/03a_pca4combined_hmm.sh)

        j41=$(fit_attempts 1 "0-$((N_STAGE1 - 1))" "afterok:${j03}" "${base[@]}")
        j42=$(fit_attempts 2 "${STAGE2_ARRAY}" "afterok:${j03}" "${base[@]}" FIXED_VT="${FIXED_VT}")
        jchk=$(submit "${base[@]}" -- --job-name="${tag}-fitchk" \
            --dependency="afterany:${j41}:${j42}" --cpus-per-task=1 --mem=2G --time=00:10:00 \
            --output="${logs}/04fitchk_%j.out" --error="${logs}/04fitchk_%j.err" \
            --wrap="${UV_PY} ${PWD}/script/clean_wmcsf_check_fits.py --scratch_dir ${root} --sub_id ${sub} --parcellation ${PARCELLATION} --fixed_vt ${FIXED_VT} --n_fit_seeds ${N_FIT_SEEDS}")

        jseed=$(submit "${base[@]}" MODE=select_seed FIXED_VT="${FIXED_VT}" -- \
            --job-name="${tag}-seed" --dependency="afterok:${jchk}" \
            --array=0-$((N_FINAL_SEEDS - 1)) --cpus-per-task=8 --mem=32G --time=48:00:00 \
            --output="${logs}/04seed_%A_%a.out" --error="${logs}/04seed_%A_%a.err" \
            script/04_combined_hdphmm.sh)
        jfin=$(submit "${base[@]}" MODE=select_finalize FIXED_VT="${FIXED_VT}" \
            N_FINAL_SEEDS="${N_FINAL_SEEDS}" -- \
            --job-name="${tag}-fin" --dependency="afterok:${jseed}" \
            --array=0 --cpus-per-task=8 --mem=32G --time=04:00:00 \
            --output="${logs}/04fin_%A_%a.out" --error="${logs}/04fin_%A_%a.err" \
            script/04_combined_hdphmm.sh)
        fin+=("${jfin}")

        echo "${sub} ${variant}: 00=${j00} 02=${j02} 03a=${j03} s1=..${j41} s2=..${j42}" \
             "fitcheck=${jchk} seeds=${jseed} finalize=${jfin} root=${root}"
    done

    jcmp=$(submit SCRATCH_DIR="${run_root}" -- --job-name="wcsf-cmp-${sub#sub-}" \
        --dependency="afterok:${fin[0]}:${fin[1]}" \
        --cpus-per-task=1 --mem=8G --time=01:00:00 \
        --output="${run_root}/compare/logs/${sub}_%j.out" \
        --error="${run_root}/compare/logs/${sub}_%j.err" \
        --wrap="${UV_PY} ${PWD}/script/clean_wmcsf_compare.py --run_root ${run_root} --sub_id ${sub} --n_final_seeds ${N_FINAL_SEEDS} --out ${run_root}/compare/${sub}.json")
    echo "${sub} compare: ${jcmp} -> ${run_root}/compare/${sub}.json"
done
