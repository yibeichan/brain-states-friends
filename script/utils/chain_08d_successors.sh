#!/bin/bash
# Chain successor jobs for all running 08d D1_lag jobs.
# Each successor fires after the predecessor (afterany = complete/timeout/fail).
# With per-layer checkpointing, the successor resumes from the last saved layer.
# Uses optimized code (Opts A+C+D) already in the repo.
#
# Usage: bash script/utils/chain_08d_successors.sh [--dry-run]
#
# Options:
#   --dry-run   Print sbatch commands without submitting

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

PARTITION="mit_preemptable"
EXCLUDE_NODES="node2803,node3805"   # GPU-equipped nodes silently stall CPU jobs (may24 lesson #390)
TIME="1-12:00:00"   # 36h — lag=0 has timed out at 24h (may22 lesson #338)
MEM="48G"
CPUS=8       # layers are single-threaded (OMP/MKL/OPENBLAS pinned to 1); 1 core per joblib worker
N_JOBS=8     # 4× the prior 2-wide layer parallelism (was CPUS=4/N_JOBS=2 → 2 cores idle)
N_PERMS=1000
VT="0.95"
SCRIPT="script/08d_transformer_depth.sh"

count=0
# Match all D1_lag job-name variants — D1_lag (PERLAGS), D1w_lag (wave), D1s_lag (successor) —
# but only RUNNING ones. Pending tasks already have a parent in flight that will chain
# them when it terminates; chaining a successor-of-a-PD-successor is just queue clutter.
while IFS='|' read jobid name state; do
    # Parse: name = "08d_D1{,w,s}_lagN_MODEL", jobid = "BASEID_ARRAYIDX"
    lag=$(echo "$name" | sed 's/.*lag\([0-9]*\)_.*/\1/')
    model=$(echo "$name" | sed 's/.*lag[0-9]*_//')
    # Strip brackets defensively — squeue shows pending array tasks as "BASEID_[N]"
    array_idx=$(echo "$jobid" | sed 's/.*_//; s/[][]//g')

    successor_name="08d_D1s_lag${lag}_${model}"
    log_prefix="logs/${successor_name}"

    cmd="sbatch --parsable \
        --dependency=afterany:${jobid} \
        --job-name=${successor_name} \
        --output=${log_prefix}_%A_%a.out \
        --error=${log_prefix}_%A_%a.err \
        --time=${TIME} \
        --mem=${MEM} \
        --cpus-per-task=${CPUS} \
        --partition=${PARTITION} \
        --exclude=${EXCLUDE_NODES} \
        --requeue \
        --array=${array_idx} \
        --export=ALL,ANALYSES=D1,LAGS=${lag},MODEL=${model},STIMULUS=friends,VT=${VT},N_PERMS=${N_PERMS},N_JOBS=${N_JOBS},PERLAGS= \
        ${SCRIPT}"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY] $cmd"
    else
        new_jobid=$(eval "$cmd")
        echo "  ${jobid} (${name}) → successor ${new_jobid}_${array_idx} (${successor_name})"
    fi
    count=$((count + 1))
done < <(squeue -u "$USER" --format="%i|%j|%t" --noheader --states=R 2>/dev/null | grep -E "08d_D1[a-z0-9]*_lag")

echo ""
echo "Chained ${count} successor jobs (n_jobs=${N_JOBS}, time=${TIME})"
