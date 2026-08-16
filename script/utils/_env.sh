#!/bin/bash
# Shared SLURM-wrapper preamble for script/*.sh — source, don't execute.
#
# Sets PROJECT_DIR and SCRIPT_DIR from this file's own on-disk location,
# puts the user-local uv install on PATH, and creates the logs/ directory
# (gitignored) that the wrappers' #SBATCH --output lines point at.
#
# How the wrappers locate this file (the canonical 3-line preamble below):
# they probe the wrapper's own location first (correct for manual runs and
# git worktrees, regardless of cwd), then fall back to SLURM_SUBMIT_DIR.
# Under sbatch the wrapper runs from the SLURM spool dir, so only the
# SLURM_SUBMIT_DIR path can resolve — sbatch jobs must therefore be
# submitted from the root of the intended checkout, and the relative
# #SBATCH --output=logs/... path is likewise resolved against the
# submit-time cwd. The pyproject guard below turns a submit dir that is
# not a usable project root into a loud, immediate error.
#
# Canonical wrapper preamble (keep in sync with the 47 call sites):
#   # Shared preamble: PROJECT_DIR/SCRIPT_DIR, uv on PATH, logs/ (utils/_env.sh)
#   _ENV="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)/script/utils/_env.sh"
#   [ -f "$_ENV" ] || _ENV="${SLURM_SUBMIT_DIR:-.}/script/utils/_env.sh"
#   source "$_ENV" || { echo "ERROR: cannot locate script/utils/_env.sh — submit from the repo root" >&2; exit 1; }

_ENV_SH_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_DIR=$(cd -- "${_ENV_SH_DIR}/../.." &> /dev/null && pwd)
unset _ENV_SH_DIR
SCRIPT_DIR="${PROJECT_DIR}/script"

# Guard: the checkout this file lives in must be a usable project root
# (catches script/-only copies and DataLad clones where pyproject.toml is
# still a dangling annex symlink).
if [ -z "$PROJECT_DIR" ] || [ ! -f "${PROJECT_DIR}/pyproject.toml" ]; then
    echo "ERROR: ${PROJECT_DIR:-<unresolved>} is not a usable project root" >&2
    echo "       (no pyproject.toml). Submit wrappers from the repo root." >&2
    exit 1
fi

# Ensure the user-local uv install is on PATH (SLURM jobs may not inherit
# it). Wrappers that `module load` afterwards should re-prepend if the
# module ships conflicting binaries.
export PATH="$HOME/.local/bin:$PATH"

# Logs directory for #SBATCH --output/--error. Note: this helps the NEXT
# submission and manual runs — SLURM opens the output file before the job
# body runs, so a missing logs/ already failed the current job at launch.
# Warn-don't-fail, and keep the sourced file's exit status at 0.
if ! mkdir -p "${PROJECT_DIR}/logs"; then
    echo "WARNING: could not create ${PROJECT_DIR}/logs" >&2
fi
