#!/usr/bin/env python3
"""Renders Figure S2: PCA loadings diagnostic panels.

Writes three separate panel files for sub-01 straight into
docs/supplementary/figures/:

    S02_pca_loadings_A.png   loadings heatmap, top 5 PCs
    S02_pca_loadings_B.png   residual variance, per-parcel + per-network
    S02_pca_loadings_C.png   network variance contribution per PC

One file per panel, no in-image panel letters and no titles: the panel letter
lives in the filename and the descriptive text lives in the SI caption, so the
figure can be re-lettered or re-laid-out without editing this script. The
quantities the old in-image titles carried (PC count, k, variance threshold)
are stated in the Figure S2 caption in docs/supplementary/README.md.

Usage:
    uv run python script/fig_S02_pca_loadings.py
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# --- load project modules ---
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from dotenv import load_dotenv
load_dotenv()

SCRATCH_DIR = Path(os.environ["SCRATCH_DIR"])

# Import per-panel functions from 03b
from utils.plot_style import apply_publication_style

# We re-use 03b's data loaders and renderers, but override rcParams
# before calling them.
import importlib.util

spec = importlib.util.spec_from_file_location(
    "pca_03b", SCRIPT_DIR / "03b_pca_loadings.py"
)
pca_03b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pca_03b)

# ── Global legibility rcParams ────────────────────────────────────────────────
# These override whatever apply_publication_style set.
TITLE_SIZE = 14
LABEL_SIZE = 11
TICK_SIZE = 9
LEGEND_SIZE = 8
SMALL_TICK = 8   # for dense tick labels (e.g. parcel names)

def _set_legibility_rc():
    plt.rcParams.update({
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "figure.titlesize": TITLE_SIZE + 2,
    })


# Repo root derived from this file, so the script writes into the clone it was
# run from rather than a hardcoded one (home is dual-pathed, and worktrees would
# otherwise silently overwrite the primary clone's SI figures).
REPO = SCRIPT_DIR.parent
SI_FIG_DIR = REPO / "docs" / "supplementary" / "figures"


def _save_panel(fig, stem):
    """Save one panel as PNG at the project's publication settings.

    PNG only: the supplement ships PNG for every figure, so this writes the
    same format as the other 26 rather than adding a vector file for three
    panels alone.
    """
    SI_FIG_DIR.mkdir(parents=True, exist_ok=True)
    _p = SI_FIG_DIR / f"{stem}.png"
    fig.savefig(_p, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved {stem}.png -> {SI_FIG_DIR}")
    return _p


# ── Re-render A1 ─────────────────────────────────────────────────────────────

def render_A1(components, var_ratio, labels, groups, n_top_pcs):
    """A1 loadings heatmap — larger fonts, tight margins."""
    _set_legibility_rc()
    n_pcs = min(n_top_pcs, components.shape[0])

    reorder = []
    boundary_positions = []
    boundary_labels = []
    pos = 0
    for net_name, indices in groups:
        reorder.extend(indices)
        boundary_positions.append(pos + len(indices) / 2)
        boundary_labels.append(net_name)
        pos += len(indices)

    data = components[:n_pcs, :][:, reorder]
    vmax = np.max(np.abs(data))

    fig_height = max(3.5, 0.5 * n_pcs + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    cb = plt.colorbar(im, ax=ax, label="Loading weight", shrink=0.8)
    cb.ax.tick_params(labelsize=SMALL_TICK)
    cb.set_label("Loading weight", fontsize=LABEL_SIZE)

    pos2 = 0
    for net_name, indices in groups:
        pos2 += len(indices)
        if pos2 < len(reorder):
            ax.axvline(x=pos2 - 0.5, color="black", linewidth=0.5, alpha=0.5)

    ax.set_xticks(boundary_positions)
    ax.set_xticklabels(boundary_labels, rotation=40, ha="right",
                       fontsize=SMALL_TICK)
    ax.set_yticks(range(n_pcs))
    ax.set_yticklabels(
        [f"PC {i+1} ({var_ratio[i]*100:.1f}%)" for i in range(n_pcs)],
        fontsize=TICK_SIZE,
    )
    ax.set_xlabel("Parcels (grouped by network)", fontsize=LABEL_SIZE)

    return _save_panel(fig, "S02_pca_loadings_A")


# ── Re-render A3 ─────────────────────────────────────────────────────────────

def render_A3(components, explained_variance, labels, groups,
              network_per_parcel, k, variance_threshold):
    """A3 residual variance — larger fonts, tight margins."""
    _set_legibility_rc()

    signal_var, residual_var, residual_frac = pca_03b.compute_residual_variance(
        components, explained_variance, k
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))

    sort_idx = np.argsort(residual_frac)[::-1]
    sorted_frac = residual_frac[sort_idx]
    sorted_colors = [
        pca_03b.NETWORK_COLORS.get(network_per_parcel[i], "#888888")
        for i in sort_idx
    ]

    ax1.bar(range(len(sorted_frac)), sorted_frac, color=sorted_colors,
            edgecolor="none", width=1.0)
    ax1.set_xlabel("Parcels (sorted by residual fraction)", fontsize=LABEL_SIZE)
    ax1.set_ylabel("Residual variance fraction", fontsize=LABEL_SIZE)
    ax1.set_xlim(-0.5, len(sorted_frac) - 0.5)
    ax1.set_ylim(0, min(1.0, sorted_frac[0] * 1.15))
    med_val = np.median(residual_frac)
    ax1.axhline(y=med_val, color="red", linestyle="--", alpha=0.7,
                linewidth=1.5, label=f"Median = {med_val:.3f}")
    ax1.legend(fontsize=LEGEND_SIZE)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.tick_params(axis="both", labelsize=TICK_SIZE)

    for rank in range(min(5, len(sort_idx))):
        pidx = sort_idx[rank]
        ax1.annotate(
            pca_03b.abbreviate_parcel_label(labels[pidx]),
            xy=(rank, sorted_frac[rank]),
            xytext=(rank + 3, sorted_frac[rank] + 0.01),
            fontsize=7, rotation=30, ha="left",
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
        )

    net_names, net_means, net_stds, net_colors = [], [], [], []
    for net_name, indices in groups:
        fracs = residual_frac[indices]
        net_names.append(net_name)
        net_means.append(np.mean(fracs))
        net_stds.append(np.std(fracs))
        net_colors.append(pca_03b.NETWORK_COLORS.get(net_name, "#888888"))

    order = np.argsort(net_means)[::-1]
    net_names = [net_names[i] for i in order]
    net_means = [net_means[i] for i in order]
    net_stds = [net_stds[i] for i in order]
    net_colors = [net_colors[i] for i in order]

    ax2.barh(range(len(net_names)), net_means, xerr=net_stds,
             color=net_colors, edgecolor="black", linewidth=0.3, alpha=0.85,
             capsize=3)
    ax2.set_yticks(range(len(net_names)))
    ax2.set_yticklabels(net_names, fontsize=TICK_SIZE)
    ax2.set_xlabel("Mean residual variance fraction", fontsize=LABEL_SIZE)
    ax2.grid(True, axis="x", alpha=0.3)
    ax2.invert_yaxis()
    ax2.tick_params(axis="both", labelsize=TICK_SIZE)

    return _save_panel(fig, "S02_pca_loadings_B")


# ── Re-render A4 ─────────────────────────────────────────────────────────────

def render_A4(components, groups, n_pcs_show, motion_flags=None):
    """A4 network variance per PC — larger fonts, tight margins."""
    _set_legibility_rc()

    n_pcs = min(n_pcs_show, components.shape[0])
    net_names = [g[0] for g in groups]
    n_nets = len(net_names)

    fractions = np.zeros((n_nets, n_pcs))
    for g_idx, (net_name, indices) in enumerate(groups):
        for pc_idx in range(n_pcs):
            sq_load = components[pc_idx, indices] ** 2
            fractions[g_idx, pc_idx] = sq_load.sum()

    col_sums = fractions.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    fractions /= col_sums

    fig, ax = plt.subplots(figsize=(max(9, n_pcs * 0.45 + 2), 5.5))

    bottom = np.zeros(n_pcs)
    x = np.arange(n_pcs)
    for g_idx, net_name in enumerate(net_names):
        color = pca_03b.NETWORK_COLORS.get(net_name, "#888888")
        ax.bar(x, fractions[g_idx], bottom=bottom, color=color,
               edgecolor="white", linewidth=0.2, label=net_name, width=0.85)
        bottom += fractions[g_idx]

    ax.set_xlabel("Principal Component", fontsize=LABEL_SIZE)
    ax.set_ylabel("Fraction of squared loadings", fontsize=LABEL_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i+1}" for i in x], fontsize=SMALL_TICK)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.2)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)

    if motion_flags and motion_flags["any_flag"]:
        flagged_pcs = set(motion_flags.get("sommot_flagged", []))
        flagged_pcs.update(motion_flags.get("subcort_flagged", []))
        for pc in flagged_pcs:
            if pc <= n_pcs:
                ax.annotate("FLAG", xy=(pc - 1, 1.02), fontsize=8,
                            color="red", fontweight="bold", ha="center",
                            annotation_clip=False)

    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=LEGEND_SIZE - 1, ncol=1, frameon=True)

    return _save_panel(fig, "S02_pca_loadings_C")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    apply_publication_style()
    _set_legibility_rc()

    sub_id = "sub-01"
    parcellation = "atlas-4S156Parcels"

    # Load data
    components, explained_variance, var_ratio = pca_03b.load_pca_model(
        sub_id, parcellation
    )
    labels = pca_03b.load_parcel_labels(parcellation)
    n_pcs_lookup = pca_03b.load_n_pcs_lookup(sub_id, parcellation)
    groups, network_per_parcel = pca_03b.group_parcels_by_network(labels)
    motion_flags = pca_03b.compute_motion_artifact_flags(components, groups)

    vt_str = "0.95"
    k = n_pcs_lookup.get(vt_str, 75)
    vt_float = float(vt_str)

    # Render panels, one file per panel — no composite.
    render_A1(components, var_ratio, labels, groups, 5)
    render_A3(components, explained_variance, labels, groups,
              network_per_parcel, k, vt_float)
    render_A4(components, groups, 20, motion_flags=motion_flags)

    print(f"Done. k={k} PCs at vt={vt_float:.2f} for {sub_id}; "
          f"3 panels written to {SI_FIG_DIR}")


if __name__ == "__main__":
    main()
