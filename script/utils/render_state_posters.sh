#!/bin/bash
# Regenerate individual per-state brain-render PNGs for the project-page viewer.
# Requires yabplot atlas meshes (built once via yabplot tools) and the per-subject
# state artifacts under $SCRATCH/output. The public .venv lacks the meshes, so run
# this in an atlas-equipped environment.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PARC="${PARC:-atlas-4S156Parcels}"
NSTATES="${NSTATES:-5}"
OUT="${OUT:-$CODE_DIR/site-staging/static/states}"
mkdir -p "$OUT"

# One-time atlas mesh build (idempotent; comment out if already built):
#   uv run --project "$CODE_DIR" python /home/yibei/yabplot/tools/build_schaefer100_cortical.py
#   uv run --project "$CODE_DIR" python /home/yibei/yabplot/tools/build_4s156_subcortical.py

for SUB in sub-01 sub-02 sub-03 sub-04 sub-05 sub-06; do
  echo "Rendering $SUB ..."
  uv run --project "$CODE_DIR" python "$CODE_DIR/script/05b_visualize_recurring_states.py" \
    --sub_id "$SUB" --parcellation "$PARC" --n_states "$NSTATES" --poster_output_dir "$OUT"
done

echo "Renders written to $OUT"
echo "Next: hand-write/update states.json referencing the produced PNGs, then re-run"
echo "site Task 2 (copy states + manifest onto the gh-pages worktree)."
