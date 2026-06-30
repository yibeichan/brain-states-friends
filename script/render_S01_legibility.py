#!/usr/bin/env python3
"""Re-assemble S01 surface map grid with improved legibility.

Strategy: montage-only improvement. The per-subject PNGs are yabplot
renders (7205x6111 RGBA) with their own small embedded titles/colorbars
that cannot be enlarged without re-running the surface script. What IS
achievable:
  - Crop excess whitespace from each per-subject PNG.
  - Tighten the 2x3 grid (reduce inter-cell gaps).
  - Add a larger, bolder subject label (sub-01 ... sub-06) in the montage
    layer.

In-panel fonts (state titles, colorbar labels embedded in the yabplot
renders) are NOT enlarged by this script.

Usage:
    uv run python script/render_S01_legibility.py
"""

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

SOURCE_DIR = SCRATCH_DIR / "output" / "05b_recurring_states_visualization" / PARC


def load_and_crop(path: Path) -> Image.Image:
    """Load RGBA PNG and auto-crop whitespace/transparency."""
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)

    # Mask: non-white and non-transparent rows/columns
    # White threshold: R,G,B all > 245 (near-white) AND alpha > 0
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    # A pixel is "background" if it's near-white OR nearly transparent
    is_bg = ((rgb > 245).all(axis=2) | (alpha < 10))
    # Rows with any foreground pixel
    row_mask = ~is_bg.all(axis=1)
    col_mask = ~is_bg.all(axis=0)

    if not row_mask.any() or not col_mask.any():
        return img

    r0, r1 = np.where(row_mask)[0][[0, -1]]
    c0, c1 = np.where(col_mask)[0][[0, -1]]

    # Add small padding (20px)
    pad = 20
    r0 = max(0, r0 - pad)
    r1 = min(arr.shape[0] - 1, r1 + pad)
    c0 = max(0, c0 - pad)
    c1 = min(arr.shape[1] - 1, c1 + pad)

    cropped = img.crop((c0, r0, c1 + 1, r1 + 1))
    return cropped


def add_subject_label(img: Image.Image, label: str, font_size: int = 80) -> Image.Image:
    """Draw a subject label in the top-left corner of the image."""
    draw = ImageDraw.Draw(img)

    # Try to use a system font; fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
                                  font_size)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                                      font_size)
        except Exception:
            font = ImageFont.load_default()

    x, y = 30, 20
    # White background behind text for legibility
    bbox = draw.textbbox((x, y), label, font=font)
    margin = 8
    draw.rectangle(
        [bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin],
        fill=(255, 255, 255, 220),
    )
    draw.text((x, y), label, fill=(20, 20, 20, 255), font=font)
    return img


def main():
    # Target cell dimensions: sized so that 3x2 composite with gaps
    # lands around 2700x1800 px (comfortable legibility, manageable size).
    # At 750x630 per cell: 3*750+2*20=2290 wide, 2*630+20=1280 tall
    # At 800x680 per cell: 3*800+2*20=2440 wide, 2*680+20=1380 tall
    # We tune downward to hit ~500KB after PNG optimization.
    TARGET_CELL_W = 640
    TARGET_CELL_H = 542

    # Load and crop per-subject PNGs
    cells = []
    for sub in SUBJECTS:
        path = SOURCE_DIR / sub / VT / f"{sub}_top5_recurring_states.png"
        if not path.exists():
            print(f"WARNING: {path} not found — using blank placeholder")
            cells.append(None)
            continue
        img = load_and_crop(path)
        # Downscale to target cell size (preserving aspect ratio, fitting within box)
        scale = min(TARGET_CELL_W / img.width, TARGET_CELL_H / img.height)
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        # Add subject label AFTER downscaling (font_size tuned to output pixel size)
        img = add_subject_label(img.copy(), sub, font_size=max(18, new_h // 14))
        cells.append(img)
        print(f"{sub}: -> {img.size}")

    # Determine cell size: max width, max height
    widths = [c.width for c in cells if c is not None]
    heights = [c.height for c in cells if c is not None]
    if not widths:
        print("No images found")
        return

    cell_w = max(widths)
    cell_h = max(heights)

    # 2 rows x 3 cols, with a small gap between cells
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
            # Draw blank placeholder
            placeholder = Image.new("RGBA", (cell_w, cell_h), (240, 240, 240, 255))
            composite.paste(placeholder, (x, y))
        else:
            # Center the cell within the allocated space (may be smaller than cell_w x cell_h)
            x_offset = (cell_w - cell.width) // 2
            y_offset = (cell_h - cell.height) // 2
            composite.paste(cell, (x + x_offset, y + y_offset), cell)

    # Convert to RGB and save
    rgb = Image.new("RGB", composite.size, (255, 255, 255))
    rgb.paste(composite, mask=composite.split()[3])

    out_path = Path(
        "/orcd/home/002/yibei/brain-states-friends-public/"
        "docs/supplementary/figures/S01_recurring_state_surface_maps.png"
    )

    # Save with max compression
    rgb.save(out_path, "PNG", optimize=True, compress_level=9)
    size_kb = out_path.stat().st_size / 1024
    print(f"S01 composite -> {out_path}  ({size_kb:.0f} KB, {out_path.stat().st_size} bytes)")
    print(f"  dimensions: {rgb.size}")


if __name__ == "__main__":
    main()
