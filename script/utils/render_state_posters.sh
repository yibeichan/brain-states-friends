#!/bin/bash
# Regenerate individual per-state brain-render PNGs for the project-page viewer.
#
# Self-contained on the PUBLIC repo. The cortical atlas (schaefer100_cortical)
# ships with the yabplot commit pinned in pyproject.toml. The 4S subcortical
# meshes derive from the CC BY-ND 4S atlas, so yabplot does not ship them: this
# script builds them into the installed yabplot once, with
# tools/build_4s156_subcortical.py from a yabplot checkout ($YABPLOT_TOOLS) and
# the 4S atlas under $ATLAS_DIR. It then renders each subject's top states under
# a headless X server. Also needs the per-subject HMM artifacts under
# $SCRATCH/output (shared scratch, auto-loaded from .env).
#
# Output: per-state cortical+subcortical PNGs in site-staging/static/states/.
# After running, update states.json to reference the produced PNGs, then copy
# both onto the gh-pages worktree (site Task 2).
#
# Env overrides: PARC, NSTATES, VT, OUT, WORK, YABPLOT_TOOLS, ATLAS_DIR.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PARC="${PARC:-atlas-4S156Parcels}"
NSTATES="${NSTATES:-5}"
VT="${VT:-0.95}"   # variance-threshold subdir of 05a/04 outputs (final/vt{VT}); "" for legacy layout
OUT="${OUT:-$CODE_DIR/site-staging/static/states}"
# 05b also writes a combined montage + individual_states to --output_dir; we don't
# need those for the viewer, and its default scratch path may hold read-only
# (git-annex) files. Send that byproduct to a fresh, writable, gitignored work dir.
WORK="${WORK:-$CODE_DIR/site-staging/.render_work}"
YABPLOT_TOOLS="${YABPLOT_TOOLS:-$HOME/yabplot/tools}"
mkdir -p "$OUT"

run() { uv run --project "$CODE_DIR" "$@"; }

# --- yabplot atlases: check the shipped cortical one, build the subcortical one ---
ATLAS_BASE="$(run python -c 'import os, yabplot; print(os.path.join(os.path.dirname(yabplot.__file__), "data", "atlases"))')"

if [ -z "$(ls -A "$ATLAS_BASE/schaefer100_cortical" 2>/dev/null)" ]; then
  echo "ERROR: yabplot atlas schaefer100_cortical missing from $ATLAS_BASE." >&2
  echo "Reinstall yabplot at the commit pinned in pyproject.toml: uv sync --reinstall-package yabplot" >&2
  exit 1
fi

if ! ls "$ATLAS_BASE"/4s_subcortical/*.vtk >/dev/null 2>&1; then
  ATLAS_DIR="${ATLAS_DIR:-$(run python -c 'import sys; from dotenv import dotenv_values; print(dotenv_values(sys.argv[1]).get("ATLAS_DIR") or "")' "$CODE_DIR/.env")}"
  if [ ! -f "$YABPLOT_TOOLS/build_4s156_subcortical.py" ] || [ -z "$ATLAS_DIR" ]; then
    echo "ERROR: 4s_subcortical meshes missing from $ATLAS_BASE." >&2
    echo "Set YABPLOT_TOOLS to the tools/ dir of a yabplot checkout at the pinned commit" >&2
    echo "and ATLAS_DIR (in .env) to the directory holding atlas-4S156Parcels/." >&2
    exit 1
  fi
  echo "Building 4s_subcortical meshes into $ATLAS_BASE/4s_subcortical ..."
  run python "$YABPLOT_TOOLS/build_4s156_subcortical.py" \
    --atlas-root "$ATLAS_DIR" --out-dir "$ATLAS_BASE/4s_subcortical"
fi

# --- Headless rendering ---
# 05b self-starts an offscreen context (osmesa, else pv.start_xvfb()); xvfb-run
# provides a real DISPLAY as a safety net when osmesa is unavailable.
XVFB=""
if command -v xvfb-run >/dev/null 2>&1; then
  XVFB="xvfb-run -a"
fi

# --vt selects the variance-threshold subdir (final/vt{VT}); omit it entirely for the legacy layout.
VT_ARG=()
[ -n "$VT" ] && VT_ARG=(--vt "$VT")

for SUB in sub-01 sub-02 sub-03 sub-04 sub-05 sub-06; do
  echo "Rendering $SUB ..."
  # Poster components go to a per-subject subdir: state IDs are per-subject, so a
  # flat dir would collide/overwrite across subjects and lose subject identity.
  $XVFB uv run --project "$CODE_DIR" python "$CODE_DIR/script/05b_visualize_recurring_states.py" \
    --sub_id "$SUB" --parcellation "$PARC" --n_states "$NSTATES" "${VT_ARG[@]}" \
    --output_dir "$WORK/$SUB" --poster_output_dir "$OUT/$SUB"
done

echo "Poster components written to $OUT/<sub-XX>/ (montage byproduct in $WORK)"
echo "Next: hand-write/update states.json referencing the produced PNGs, then re-run"
echo "site Task 2 (copy states + manifest onto the gh-pages worktree)."
