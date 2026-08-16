#!/bin/bash
# Shared SLURM-wrapper preamble for script/*.sh — source, don't execute.
#
# Resolves PROJECT_DIR from this file's own on-disk location. BASH_SOURCE of
# a *sourced* file is its real repo path even under sbatch (only the
# submitted wrapper is copied to the SLURM spool dir), so PROJECT_DIR is
# correct regardless of the submit-time cwd. Also puts the user-local uv
# install on PATH and creates the logs/ directory (gitignored) that the
# wrappers' #SBATCH --output lines point at.
#
# Note: sbatch still resolves the relative #SBATCH --output=logs/... path
# against the submit-time cwd, so wrappers should be submitted from the
# repo root; the pyproject.toml guard below turns a wrong submit dir into
# a loud, immediate error instead of a silently wrong environment.
#
# Usage — first non-SBATCH line of every wrapper in script/:
#   source "${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}/script/utils/_env.sh" || exit 1

_ENV_SH_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_DIR=$(cd -- "${_ENV_SH_DIR}/../.." &> /dev/null && pwd)
unset _ENV_SH_DIR
SCRIPT_DIR="${PROJECT_DIR}/script"

if [ -z "$PROJECT_DIR" ] || [ ! -f "${PROJECT_DIR}/pyproject.toml" ]; then
    echo "ERROR: could not resolve the project root (no pyproject.toml at" >&2
    echo "       '${PROJECT_DIR}'). Submit wrappers from the repo root." >&2
    exit 1
fi

# Ensure the user-local uv install is on PATH (SLURM jobs may not inherit it)
export PATH="$HOME/.local/bin:$PATH"

# Logs directory for #SBATCH --output/--error (must exist at job launch)
mkdir -p "${PROJECT_DIR}/logs"
