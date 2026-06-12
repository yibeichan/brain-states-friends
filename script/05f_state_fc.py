#!/usr/bin/env python3
"""
05f_state_fc.py - Empirical within-state functional connectivity.

Computes state-specific parcel-space correlation matrices from the actual BOLD
timeseries, using decoded state assignments from the combined HDP-HMM as labels.
For each state k, all TRs assigned to that state are pooled across runs and a
Ledoit-Wolf shrinkage covariance is estimated, then converted to correlation.

This is genuine state-conditioned FC - parcel-pair correlations computed from
data, not implied by PCA loading structure.

Adapted from mario-rSLDS/scripts/05d_state_fc.py.

Prerequisites:
    - 04_combined_hdphmm.py (mode: select) completed → decoded_states.pkl
    - 02_extract_parcel_ts.py completed → parcel_avg .npy files
    - 05e_temporal_trend_a4.py completed → state_flags.csv (optional; for heatmap sorting)

Outputs:
    {SCRATCH_DIR}/output/05f_state_fc/{parcellation}/{sub_id}/
        state_empirical_corr.npy   # (K, n_parcels, n_parcels)
        state_delta_fc.npy          # (K, n_parcels, n_parcels) R_k - R_grand
        grand_mean_corr.npy         # (n_parcels, n_parcels)
        fc_similarity_corr_rv.npy   # (K, K) RV coefficient on empirical correlations
        network_delta_fc.npy        # (n_active, n_nets, n_nets)
        top_pairs_per_state.json
        metadata.json
        figures/
"""

import os
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from utils.common import normalize_parcellation_name
from utils.stats import compute_rv_coefficient
from utils.state_fc import (
    load_matched_data,
    compute_empirical_state_fc,
    compute_delta_correlation,
)
from utils.state_flags_io import (
    load_state_flags,
    CATEGORY_PRIORITY,
    CATEGORY_COLORS,
    CATEGORY_DISPLAY_NAMES,
)

load_dotenv()
SCRATCH_DIR = os.getenv("SCRATCH_DIR")
if SCRATCH_DIR is None:
    raise ValueError("SCRATCH_DIR must be set in the .env file")

ATLAS_BASE = os.getenv("ATLAS_DIR")
if ATLAS_BASE is None:
    raise ValueError("ATLAS_DIR must be set in the .env file")

from utils.plot_style import NETWORK_ORDER, NETWORK_COLORS, assign_network

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Empirical within-state functional connectivity.")
    p.add_argument("--sub_id", type=str, required=True)
    p.add_argument("--parcellation", type=str, default="atlas-4S156Parcels")
    p.add_argument("--vt", type=str, default=None)
    p.add_argument("--min_trs", type=int, default=30,
                   help="Min TRs per state for reliable FC estimation")
    p.add_argument("--top_n_pairs", type=int, default=20)
    return p.parse_args()


# ── Atlas metadata ────────────────────────────────────────────────────────────


def load_atlas_metadata(parcellation: str) -> pd.DataFrame:
    """Load atlas TSV with network labels."""
    tsv_path = os.path.join(ATLAS_BASE, parcellation, f"{parcellation}_dseg.tsv")
    df = pd.read_csv(tsv_path, sep="\t")
    # Cortical parcels: use network_label; subcortical: parse from label
    df["network"] = df["network_label"]
    subcort_mask = df["network_label"].isna() | (df["network_label"] == "n/a")
    df.loc[subcort_mask, "network"] = df.loc[subcort_mask, "label"].apply(assign_network)
    return df


def compute_network_delta_fc(delta_R, atlas_df, active_states):
    """Aggregate parcel-level delta_R into network-level blocks."""
    net_map = dict(zip(atlas_df["index"] - 1, atlas_df["network"]))  # 0-based
    net_indices = {net: [p for p, n in net_map.items() if n == net]
                   for net in NETWORK_ORDER}

    n_nets = len(NETWORK_ORDER)
    net_delta_fc = np.zeros((len(active_states), n_nets, n_nets))

    for si, k in enumerate(active_states):
        dR_k = delta_R[k]
        for i, net_i in enumerate(NETWORK_ORDER):
            idx_i = net_indices[net_i]
            if not idx_i:
                continue
            for j, net_j in enumerate(NETWORK_ORDER):
                idx_j = net_indices[net_j]
                if not idx_j:
                    continue
                block = dR_k[np.ix_(idx_i, idx_j)]
                if i == j:
                    triu = np.triu_indices(len(idx_i), k=1)
                    if len(triu[0]) > 0:
                        net_delta_fc[si, i, j] = block[triu].mean()
                else:
                    net_delta_fc[si, i, j] = block.mean()

    return net_delta_fc, list(NETWORK_ORDER)


def get_top_pairs(delta_R_k, labels, top_n=20):
    """Find top parcel pairs by |delta_R| for one state."""
    n = delta_R_k.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    vals = delta_R_k[triu_idx]
    ranked = np.argsort(-np.abs(vals))[:top_n]
    pairs = []
    for r in ranked:
        i, j = triu_idx[0][r], triu_idx[1][r]
        pairs.append({
            "parcel_i": int(i), "parcel_j": int(j),
            "label_i": str(labels[i]), "label_j": str(labels[j]),
            "delta_R": float(vals[r]),
        })
    return pairs


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_network_delta_fc(net_delta_fc, network_names, active_states, output_dir):
    """Per-state network-level delta-FC heatmaps."""
    n_nets = len(network_names)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    for si, k in enumerate(active_states):
        mat = net_delta_fc[si]
        vmax = max(np.percentile(np.abs(mat), 98), 0.01)

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(n_nets))
        ax.set_xticklabels(network_names, rotation=45, ha="right")
        ax.set_yticks(range(n_nets))
        ax.set_yticklabels(network_names)

        for tick in ax.get_xticklabels():
            net = tick.get_text()
            if net in NETWORK_COLORS:
                tick.set_color(NETWORK_COLORS[net])
                tick.set_fontweight("bold")
        for tick in ax.get_yticklabels():
            net = tick.get_text()
            if net in NETWORK_COLORS:
                tick.set_color(NETWORK_COLORS[net])
                tick.set_fontweight("bold")

        for i in range(n_nets):
            for j in range(n_nets):
                val = mat[i, j]
                color = "white" if abs(val) > vmax * 0.3 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=5.5, color=color)

        ax.set_title(f"State k={k}: network-level \u0394r")
        plt.colorbar(im, ax=ax, label="\u0394r (Pearson)", shrink=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"fig_network_delta_fc_k{k}.png"), dpi=150)
        plt.close(fig)

    logger.info("Saved %d network delta-FC heatmaps", len(active_states))


def plot_fc_similarity_heatmap(rv_mat, active_states, state_flags, output_dir):
    """Heatmap of FC similarity (RV coefficient), sorted by 05e_a4 category.

    States are sorted by summary_category (CATEGORY_PRIORITY order), then by
    state index within each category.  A colour bar on the left/top annotates
    the category of each state.  ``unused`` and ``low_confidence`` states are
    excluded (insufficient data for meaningful FC).

    Args:
        rv_mat:        (K, K) RV coefficient matrix.
        active_states: array of state indices with >= min_trs.
        state_flags:   DataFrame from load_state_flags (or None for fallback).
        output_dir:    base output directory.
    """
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Build category map for active states
    if state_flags is not None:
        cat_map = dict(zip(state_flags["state"], state_flags["summary_category"]))
    else:
        cat_map = {}
    categories = [cat_map.get(int(s), "unknown") for s in active_states]

    # Filter out unused / low_confidence
    exclude = {"unused", "low_confidence"}
    keep_mask = [c not in exclude for c in categories]
    plot_states = active_states[keep_mask]
    plot_cats = [c for c, keep in zip(categories, keep_mask) if keep]

    if len(plot_states) == 0:
        logger.warning("No states remain after filtering - skipping FC heatmap")
        return

    # Sort by category priority, then state index
    cat_rank = {c: i for i, c in enumerate(CATEGORY_PRIORITY)}
    sort_order = sorted(
        range(len(plot_states)),
        key=lambda i: (cat_rank.get(plot_cats[i], len(CATEGORY_PRIORITY)), plot_states[i]),
    )
    plot_states = plot_states[sort_order]
    plot_cats = [plot_cats[i] for i in sort_order]

    # Subset RV matrix and compute data-driven color range
    sub = rv_mat[np.ix_(plot_states, plot_states)]
    n = len(plot_states)
    off_diag = sub[np.triu_indices(n, k=1)]
    vmin = np.floor(np.nanmin(off_diag) * 20) / 20  # round down to nearest 0.05
    vmax = 1.0

    # Category boundaries for grid lines
    boundary_positions = []
    for i in range(1, n):
        if plot_cats[i] != plot_cats[i - 1]:
            boundary_positions.append(i - 0.5)

    # Build category colour strip
    cat_colors_arr = [CATEGORY_COLORS.get(c, "#999999") for c in plot_cats]

    # ── Figure: heatmap with attached category colour bar ───────────────
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, ax_main = plt.subplots(figsize=(9, 7.5))

    # Main heatmap
    im = ax_main.imshow(sub, vmin=vmin, vmax=vmax, cmap="inferno", aspect="equal",
                        interpolation="nearest")
    ax_main.set_xticks(range(n))
    ax_main.set_xticklabels([f"S{s}" for s in plot_states], rotation=45,
                            ha="right", fontsize=6)
    ax_main.set_yticks([])
    for pos in boundary_positions:
        ax_main.axhline(pos, color="white", linewidth=0.8, alpha=0.7)
        ax_main.axvline(pos, color="white", linewidth=0.8, alpha=0.7)

    plt.colorbar(im, ax=ax_main, label="RV coefficient", shrink=0.8)

    # Category colour bar - attached to left of heatmap for pixel-perfect alignment
    divider = make_axes_locatable(ax_main)
    ax_cat = divider.append_axes("left", size="3%", pad=0.02)
    cat_img = np.arange(n).reshape(-1, 1)
    cmap_cat = ListedColormap(cat_colors_arr)
    norm_cat = BoundaryNorm(np.arange(n + 1) - 0.5, n)
    ax_cat.imshow(cat_img, cmap=cmap_cat, norm=norm_cat, aspect="auto",
                  interpolation="nearest")
    ax_cat.set_xticks([])
    ax_cat.set_yticks(range(n))
    ax_cat.set_yticklabels([f"S{s}" for s in plot_states], fontsize=6)
    for pos in boundary_positions:
        ax_cat.axhline(pos, color="white", linewidth=1.5)

    # Legend for categories
    seen = dict.fromkeys(plot_cats)  # preserves order, unique
    handles = [
        Patch(facecolor=CATEGORY_COLORS.get(c, "#999999"),
              label=CATEGORY_DISPLAY_NAMES.get(c, c))
        for c in seen
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(seen), fontsize=7,
               title="Category", title_fontsize=8, frameon=False)

    fig.suptitle("Empirical FC Similarity (RV) by State Category", fontsize=11)
    fig.subplots_adjust(top=0.93, bottom=0.18)
    fig.savefig(os.path.join(fig_dir, "fig_fc_similarity_heatmap_by_category.png"), dpi=150)
    plt.close(fig)
    logger.info(
        "FC similarity heatmap: %d states (%s)",
        n, ", ".join(f"{c}: {sum(1 for x in plot_cats if x == c)}" for c in seen),
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    sub_id = args.sub_id
    parc = normalize_parcellation_name(args.parcellation)

    logger.info("=" * 70)
    logger.info("05f Empirical State FC: %s", sub_id)
    logger.info("=" * 70)

    # ── Paths ─────────────────────────────────────────────────────────────
    vt_subdir = f"vt{args.vt}" if args.vt else ""
    ds_path = os.path.join(
        SCRATCH_DIR, "output", "04_combined_hdphmm", parc, sub_id,
        "final", vt_subdir, "decoded_states.pkl",
    )
    results_path = os.path.join(
        SCRATCH_DIR, "output", "04_combined_hdphmm", parc, sub_id,
        "final", vt_subdir, "final_results.json",
    )
    output_dir = os.path.join(SCRATCH_DIR, "output", "05f_state_fc", parc, sub_id)
    if args.vt:
        output_dir = os.path.join(output_dir, f"vt{args.vt}")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load inputs ───────────────────────────────────────────────────────
    with open(ds_path, "rb") as f:
        decoded_states = pickle.load(f)

    with open(results_path) as f:
        final_results = json.load(f)
    K = final_results["model_info"]["n_states"]

    # ── State flags from 05e_a4 (optional) ────────────────────────────────
    state_flags = load_state_flags(sub_id, parc, SCRATCH_DIR, vt=args.vt)

    # ── Atlas metadata (load once for column stripping + network aggregation) ──
    atlas_df = load_atlas_metadata(parc)
    n_atlas_parcels = len(atlas_df)

    # ── Load and match parcel timeseries ──────────────────────────────────
    parcel_ts, viterbi, n_runs = load_matched_data(
        sub_id, parc, decoded_states, n_atlas_parcels,
        scratch_dir=SCRATCH_DIR,
    )

    # ── Compute empirical FC ──────────────────────────────────────────────
    corr_parcel, n_trs_per_state, reliable, shrinkage_alpha = compute_empirical_state_fc(
        parcel_ts, viterbi, K, min_trs=args.min_trs,
    )

    # ── Active states and occupancies ─────────────────────────────────────
    occupancies = n_trs_per_state.astype(float) / max(n_trs_per_state.sum(), 1)
    active_states = np.where(n_trs_per_state >= args.min_trs)[0]
    logger.info("Active states: %d of %d", len(active_states), K)

    # ── Delta correlation ─────────────────────────────────────────────────
    delta_R, R_grand = compute_delta_correlation(corr_parcel, occupancies)
    net_delta_fc, network_names = compute_network_delta_fc(
        delta_R, atlas_df, active_states,
    )

    # ── FC similarity (RV coefficient on empirical correlations) ──────────
    # Compute only on active states to avoid identity-matrix inflation from
    # inactive states (n_k < 2 → eye(p)), then embed back into K×K with NaN.
    rv_active = compute_rv_coefficient(corr_parcel[active_states])
    rv_mat = np.full((K, K), np.nan)
    rv_mat[np.ix_(active_states, active_states)] = rv_active

    # ── Top parcel pairs ──────────────────────────────────────────────────
    labels = atlas_df.sort_values("index")["label"].values
    top_pairs = {}
    for k in active_states:
        top_pairs[f"state_{k}"] = get_top_pairs(
            delta_R[k], labels, top_n=args.top_n_pairs,
        )

    # ── Save ──────────────────────────────────────────────────────────────
    np.save(os.path.join(output_dir, "state_empirical_corr.npy"), corr_parcel)
    np.save(os.path.join(output_dir, "state_delta_fc.npy"), delta_R)
    np.save(os.path.join(output_dir, "grand_mean_corr.npy"), R_grand)
    np.save(os.path.join(output_dir, "fc_similarity_corr_rv.npy"), rv_mat)
    np.save(os.path.join(output_dir, "network_delta_fc.npy"), net_delta_fc)

    with open(os.path.join(output_dir, "top_pairs_per_state.json"), "w") as f:
        json.dump(top_pairs, f, indent=2)

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump({
            "method": "empirical_ledoit_wolf",
            "note": (
                "Empirical within-state FC: parcel correlations computed from "
                "BOLD timeseries at TRs assigned to each state by Viterbi "
                "decoding. Covariance estimated with Ledoit-Wolf shrinkage."
            ),
            "K": K,
            "n_parcels": int(corr_parcel.shape[1]),
            "n_networks": len(network_names),
            "network_names": network_names,
            "n_parcels_per_bin": [
                int((atlas_df["network"] == net).sum()) for net in network_names
            ],
            "active_states": active_states.tolist(),
            "n_trs_per_state": n_trs_per_state.tolist(),
            "reliable_states": np.where(reliable)[0].tolist(),
            "unreliable_states": np.where(~reliable)[0].tolist(),
            "shrinkage_alpha_per_state": {
                str(k): round(float(shrinkage_alpha[k]), 4)
                for k in range(K) if np.isfinite(shrinkage_alpha[k])
            },
            "min_trs": args.min_trs,
            "n_runs": n_runs,
            "total_trs": int(parcel_ts.shape[0]),
        }, f, indent=2)

    logger.info("Saved arrays and metadata to %s", output_dir)

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_network_delta_fc(net_delta_fc, network_names, active_states, output_dir)
    plot_fc_similarity_heatmap(rv_mat, active_states, state_flags, output_dir)

    logger.info("=" * 70)
    logger.info("Done. Output: %s", output_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
