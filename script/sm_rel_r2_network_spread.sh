#!/bin/bash
#SBATCH --job-name=sm_rel_r2_network_spread
#SBATCH --output=logs/sm_rel_r2_network_spread_%A.out
#SBATCH --error=logs/sm_rel_r2_network_spread_%A.err
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=ou_bcs_normal

# Single job, not an array: the analysis is a few seconds per subject and the
# published-medians gate needs all six subjects loaded together anyway.

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
    PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
fi

mkdir -p "${PROJECT_DIR}/logs"

echo "=============================================="
echo "sm_rel_r2_network_spread - R2 network-spread null"
echo "=============================================="
echo "n_draws:  ${N_DRAWS:-10000}"
echo "=============================================="

uv run --project "${PROJECT_DIR}" python "${PROJECT_DIR}/script/sm_rel_r2_network_spread.py" \
    --n_draws "${N_DRAWS:-10000}" \
    --n_group "${N_GROUP:-10000}"

echo "=============================================="
echo "R2 network-spread null complete"
echo "=============================================="
