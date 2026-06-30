#!/usr/bin/env python3
"""Re-render S03 D1-net montage with improved legibility.

One shared title, per-subject labels in larger font, tighter spacing,
readable colorbar. 5 subjects (sub-06 excluded per caption).

Usage:
    uv run python script/render_S03_legibility.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from dotenv import load_dotenv
load_dotenv()

from utils.plot_style import apply_publication_style

SCRATCH_DIR = Path(os.environ["SCRATCH_DIR"])

# sub-06 excluded (too few states)
SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05"]
PARC = "atlas-4S156Parcels"
MODEL = "friends_dinov2-large"
DATA_DIR = SCRATCH_DIR / "output" / "08d_transformer_depth" / PARC

# Canonical network × polarity ordering (from 08d_transformer_depth.py)
def _d1net_group_order():
    """Return canonical (network, polarity) ordering for D1-net rows."""
    import sys
    sys.path.insert(0, str(SCRIPT_DIR))
    # Reconstruct from NETWORK_ORDER + polarities
    from utils.plot_style import NETWORK_ORDER
    return [f"{net}_{pol}" for net in NETWORK_ORDER for pol in ("pos", "neg")]


def load_d1net(sub):
    path = DATA_DIR / sub / MODEL / "D1_net.json"
    with open(path) as f:
        d = json.load(f)
    return d["groups"]


def make_matrix(groups, row_order, all_layers):
    """Build balanced-accuracy matrix for one subject (all rows including NaN)."""
    rows = [g for g in row_order if g in groups]
    if not rows:
        return None, []
    mat = np.full((len(rows), len(all_layers)), np.nan)
    for i, g in enumerate(rows):
        per_layer = groups[g].get("per_layer")
        if not per_layer:
            continue
        for j, lyr in enumerate(all_layers):
            entry = per_layer.get(str(lyr))
            if entry is not None and isinstance(entry, dict):
                mat[i, j] = entry["balanced_accuracy"]
    return mat, rows


def main():
    apply_publication_style()

    row_order = _d1net_group_order()

    # Load all subjects
    all_groups = {}
    for sub in SUBJECTS:
        try:
            all_groups[sub] = load_d1net(sub)
        except FileNotFoundError:
            print(f"WARNING: D1_net.json not found for {sub}, skipping")

    # Use all groups that appear in any subject's JSON (including skipped ones)
    global_rows_seen = []
    seen_set = set()
    for g in row_order:
        for sub in SUBJECTS:
            if sub in all_groups and g in all_groups[sub]:
                if g not in seen_set:
                    global_rows_seen.append(g)
                    seen_set.add(g)
                break

    all_layers = sorted({
        int(lyr)
        for sub in SUBJECTS
        if sub in all_groups
        for g in all_groups[sub]
        if all_groups[sub][g].get("per_layer")
        for lyr in (all_groups[sub][g]["per_layer"] or {})
    })

    # Global color range (chance is typically ~1/N_states)
    all_vals = []
    for sub in SUBJECTS:
        if sub not in all_groups:
            continue
        for g, v in all_groups[sub].items():
            if v.get("per_layer"):
                for lyr_entry in v["per_layer"].values():
                    if isinstance(lyr_entry, dict):
                        ba = lyr_entry.get("balanced_accuracy")
                    else:
                        ba = lyr_entry
                    if ba is not None:
                        all_vals.append(ba)

    vmin = float(np.nanpercentile(all_vals, 2)) if all_vals else 0.0
    vmax = float(np.nanpercentile(all_vals, 98)) if all_vals else 1.0

    n_subjects = len([s for s in SUBJECTS if s in all_groups])
    n_rows_max = len(global_rows_seen)
    n_cols = len(all_layers)

    if n_rows_max == 0 or n_cols == 0:
        print("No data to plot")
        return

    # Layout: 1 row per subject side-by-side heatmaps, colorbar at right
    # Each panel: (n_rows_max, n_cols) heatmap
    # Use a figure with tight gridspec

    fig_w = 2.5 + n_subjects * max(2.2, 0.22 * n_cols)
    fig_h = max(5.0, 0.28 * n_rows_max + 2.0)

    fig = plt.figure(figsize=(fig_w, fig_h))

    # GridSpec: n_subjects columns + 1 for colorbar
    gs = gridspec.GridSpec(
        1, n_subjects + 1,
        figure=fig,
        width_ratios=[1.0] * n_subjects + [0.05],
        wspace=0.06,
        left=0.12, right=0.97,
        top=0.90, bottom=0.12,
    )

    axes = [fig.add_subplot(gs[0, i]) for i in range(n_subjects)]
    cax = fig.add_subplot(gs[0, n_subjects])

    im_ref = None
    for ax_idx, sub in enumerate([s for s in SUBJECTS if s in all_groups]):
        mat, rows = make_matrix(all_groups[sub], global_rows_seen, all_layers)
        ax = axes[ax_idx]

        if mat is None or mat.size == 0:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9)
            ax.set_title(sub, fontsize=12, fontweight="bold", pad=5)
            continue

        im = ax.imshow(
            mat, aspect="auto", cmap="viridis",
            vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        im_ref = im

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(all_layers, fontsize=7, rotation=0)
        ax.set_title(sub, fontsize=12, fontweight="bold", pad=5)

        if ax_idx == 0:
            ax.set_yticks(range(len(global_rows_seen)))
            ax.set_yticklabels(global_rows_seen, fontsize=8)
            ax.set_ylabel("Network (polarity)", fontsize=10)
        else:
            ax.set_yticks([])

        ax.set_xlabel("Layer", fontsize=9)

    # Colorbar
    if im_ref is not None:
        cb = fig.colorbar(im_ref, cax=cax)
        cb.set_label("Balanced accuracy", fontsize=10)
        cb.ax.tick_params(labelsize=8)

    # Shared title
    fig.suptitle(
        "D1-net: balanced accuracy by (network, polarity) — DINOv2-large",
        fontsize=13, fontweight="bold", y=0.97,
    )

    out_path = Path(
        "/orcd/home/002/yibei/brain-states-friends-public/"
        "docs/supplementary/figures/S03_video_peak_depth.png"
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    size_kb = out_path.stat().st_size / 1024
    print(f"S03 -> {out_path}  ({size_kb:.0f} KB, {out_path.stat().st_size} bytes)")

    if size_kb > 500:
        from PIL import Image
        img = Image.open(out_path)
        img.save(out_path, "PNG", optimize=True, compress_level=9)
        size_kb = out_path.stat().st_size / 1024
        print(f"  recompressed -> {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
