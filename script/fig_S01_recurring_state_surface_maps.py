#!/usr/bin/env python3
"""Renders Figure S1: recurring-state cortical/subcortical surface maps.

Produces docs/supplementary/figures/S01_recurring_state_surface_maps.png by
montaging per-subject PNGs (6 subjects × top-5 states, 2×3 grid) at
720×612 px cells with bold subject labels.

PREREQUISITE — per-subject renders
-----------------------------------
Each per-subject PNG (sub-0X_top5_recurring_states.png) must be produced
BEFORE running this script by calling 05b_visualize_recurring_states.py in
an environment that has yabplot + pyvista (OSMesa) installed, because the
public venv lacks the yabplot atlas data needed for cortical surface rendering.

Use the private repo's venv for the per-subject renders:

    PRIVATE=/orcd/home/002/yibei/brain-states-friends
    PUB_05B=/orcd/home/002/yibei/brain-states-friends-public/script/05b_visualize_recurring_states.py
    cp $PUB_05B $PRIVATE/script/05b_visualize_recurring_states.py   # copy font-bumped version

    for SUB in sub-01 sub-02 sub-03 sub-04 sub-05 sub-06; do
        xvfb-run -a $PRIVATE/.venv/bin/python $PRIVATE/script/05b_visualize_recurring_states.py \\
            --sub_id $SUB --parcellation atlas-4S156Parcels --n_states 5 --vt 0.95 \\
            --output_dir /tmp/05b_rerender/$SUB
    done

    git -C $PRIVATE checkout -- script/05b_visualize_recurring_states.py  # restore private

Then point RENDER_DIR (below) at /tmp/05b_rerender or wherever you placed the outputs.

Montage parameters (matching committed S01; see .superpowers/sdd/task-fb2b-report.md):
  - Cell size: 720×612 px
  - Grid: 3 columns × 2 rows (sub-01..sub-06, left→right, top→bottom)
  - Gap: 12 px between cells
  - Final composite: ~2181×1236 px, ≤700 KB

Usage:
    uv run python script/fig_S01_recurring_state_surface_maps.py [--render_dir DIR]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from dotenv import load_dotenv
load_dotenv()

from PIL import Image, ImageDraw, ImageFont

SCRATCH_DIR = Path(os.environ["SCRATCH_DIR"])

SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"]
PARC = "atlas-4S156Parcels"
VT = "vt0.95"

# Default: look for font-bumped per-subject renders in scratch.
# For re-rendering, override with --render_dir pointing at /tmp/05b_rerender.
DEFAULT_RENDER_DIR = SCRATCH_DIR / "output" / "05b_recurring_states_visualization" / PARC


def load_and_crop(path: Path) -> Image.Image:
    """Load RGBA PNG and auto-crop whitespace/transparency."""
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)

    # Mask: non-white and non-transparent rows/columns
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    # A pixel is "background" if near-white OR nearly transparent
    is_bg = ((rgb > 245).all(axis=2) | (alpha < 10))
    row_mask = ~is_bg.all(axis=1)
    col_mask = ~is_bg.all(axis=0)

    if not row_mask.any() or not col_mask.any():
        return img

    r0, r1 = np.where(row_mask)[0][[0, -1]]
    c0, c1 = np.where(col_mask)[0][[0, -1]]

    # Small padding (20px)
    pad = 20
    r0 = max(0, r0 - pad)
    r1 = min(arr.shape[0] - 1, r1 + pad)
    c0 = max(0, c0 - pad)
    c1 = min(arr.shape[1] - 1, c1 + pad)

    return img.crop((c0, r0, c1 + 1, r1 + 1))


def add_subject_label(img: Image.Image, label: str, font_size: int = 44) -> Image.Image:
    """Draw a bold subject label in the top-left corner."""
    draw = ImageDraw.Draw(img)

    for candidate in [
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            font = ImageFont.truetype(candidate, font_size)
            break
        except Exception:
            font = ImageFont.load_default()

    x, y = 30, 20
    bbox = draw.textbbox((x, y), label, font=font)
    margin = 8
    draw.rectangle(
        [bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin],
        fill=(255, 255, 255, 220),
    )
    draw.text((x, y), label, fill=(20, 20, 20, 255), font=font)
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render_dir", type=str, default=None,
        help=(
            "Directory containing per-subject font-bumped 05b renders. "
            "Expected layout: <render_dir>/<sub_id>/vt0.95/<sub_id>_top5_recurring_states.png "
            "OR <render_dir>/<sub_id>/<sub_id>_top5_recurring_states.png "
            "(the latter for /tmp/05b_rerender/<sub_id>/ style). "
            "Defaults to $SCRATCH_DIR/output/05b_recurring_states_visualization/atlas-4S156Parcels."
        ),
    )
    args = parser.parse_args()

    # Cell dimensions — must match FB2b montage (see task-fb2b-report.md)
    TARGET_CELL_W = 720
    TARGET_CELL_H = 612

    render_dir = Path(args.render_dir) if args.render_dir else DEFAULT_RENDER_DIR

    cells = []
    for sub in SUBJECTS:
        # Try scratch layout first (sub/vt0.95/sub_top5...png), then flat layout
        candidates = [
            render_dir / sub / VT / f"{sub}_top5_recurring_states.png",
            render_dir / sub / f"{sub}_top5_recurring_states.png",
        ]
        path = next((c for c in candidates if c.exists()), None)

        if path is None:
            print(f"WARNING: {sub} render not found under {render_dir} — using blank placeholder")
            cells.append(None)
            continue

        img = load_and_crop(path)
        scale = min(TARGET_CELL_W / img.width, TARGET_CELL_H / img.height)
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        # Font size ~44px at cell scale (tuned for 720×612 target)
        img = add_subject_label(img.copy(), sub, font_size=max(18, new_h // 14))
        cells.append(img)
        print(f"{sub}: {path.name} -> cell {img.size}")

    widths = [c.width for c in cells if c is not None]
    heights = [c.height for c in cells if c is not None]
    if not widths:
        print("No images found — check render_dir.")
        return

    cell_w = max(widths)
    cell_h = max(heights)

    n_cols, n_rows = 3, 2
    gap = 12
    total_w = n_cols * cell_w + (n_cols - 1) * gap
    total_h = n_rows * cell_h + (n_rows - 1) * gap

    composite = Image.new("RGBA", (total_w, total_h), (255, 255, 255, 255))

    for idx, (sub, cell) in enumerate(zip(SUBJECTS, cells)):
        row = idx // n_cols
        col = idx % n_cols
        x = col * (cell_w + gap)
        y = row * (cell_h + gap)

        if cell is None:
            placeholder = Image.new("RGBA", (cell_w, cell_h), (240, 240, 240, 255))
            composite.paste(placeholder, (x, y))
        else:
            x_offset = (cell_w - cell.width) // 2
            y_offset = (cell_h - cell.height) // 2
            composite.paste(cell, (x + x_offset, y + y_offset), cell)

    # Convert to RGB
    rgb = Image.new("RGB", composite.size, (255, 255, 255))
    rgb.paste(composite, mask=composite.split()[3])

    out_path = Path(
        "/orcd/home/002/yibei/brain-states-friends-public/"
        "docs/supplementary/figures/S01_recurring_state_surface_maps.png"
    )

    rgb.save(out_path, "PNG", optimize=True, compress_level=9)
    size_kb = out_path.stat().st_size / 1024
    print(f"S01 composite -> {out_path}  ({size_kb:.0f} KB, {out_path.stat().st_size} bytes)")
    print(f"  dimensions: {rgb.size}")

    if size_kb > 700:
        print(f"WARNING: output is {size_kb:.0f} KB, exceeds ~700 KB target.")


if __name__ == "__main__":
    main()
