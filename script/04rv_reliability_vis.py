#!/usr/bin/env python3
"""
04rv_reliability_vis.py - Reliability figures for LOSO and split-half analyses.

Produces 7 figures summarising the structural reliability of HMM solutions:

  Fig 1  State Spatial Reproducibility         (manuscript, multi-subject 2x3)
  Fig 2  Scalar Invariant Stability            (manuscript, 2x2 grid)
  Fig 3  Recurrence Profile Reliability        (manuscript, multi-subject 2x3)
  Fig 4  LOSO Recurrence CDFs                  (supplementary, multi-subject 2x3)
  Fig 5  Network Composition                   (supplementary, multi-subject 2x3)
  Fig 6a Transition Scalar Stability           (supplementary, per-subject)
  Fig 6b Transition Matrix Comparison          (supplementary, per-subject)
  Fig 7  FC Structure Stability                (supplementary, multi-subject 2x3)

Prerequisites:
    - 04ra_loso_struct_comp.py completed (all subjects)
    - 04rb_split_half_reliability.py completed (all subjects)
    - 04_combined_hdphmm.py (mode: select) completed (primary model)

Outputs:
    Multi-subject: {SCRATCH_DIR}/output/04rv_reliability_vis/{parcellation}/
    Per-subject:   {SCRATCH_DIR}/output/04rv_reliability_vis/{parcellation}/{sub_id}/
"""

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import theilslopes

sys.path.insert(0, str(Path(__file__).parent))
from utils.common import normalize_parcellation_name
from utils.plot_style import (
    apply_publication_style,
    NETWORK_ORDER,
    NETWORK_COLORS,
    recurrence_color,
    make_recurrence_colorbar,
    RECURRENCE_CMAP,
)

from dotenv import load_dotenv

load_dotenv()
SCRATCH_DIR = os.getenv("SCRATCH_DIR")
if SCRATCH_DIR is None:
    raise ValueError("SCRATCH_DIR must be set in the .env file")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ALL_SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"]

# Fold colours (tab10, one per season)
FOLD_COLORS = [cm.tab10(i) for i in range(6)]

# Metric display names
METRIC_DISPLAY = {
    "k_active": r"$K_{\mathrm{active}}$",
    "transition_entropy": "Transition entropy",
    "self_transition_prob": "Self-transition prob.",
    "dwell_median_tr": "Median dwell (TR)",
}


# =============================================================================
# Data loaders
# =============================================================================

def load_loso_data(parcellation, sub_id):
    """Load all LOSO reliability data for a single subject.

    Returns dict with keys:
        fold_invariants, cross_fold_consistency, hungarian_matching, noise_floor
    """
    base = os.path.join(
        SCRATCH_DIR, "output", "04ra_loso_struct_comp", parcellation, sub_id
    )
    data = {}
    for name in [
        "fold_invariants",
        "cross_fold_consistency",
        "hungarian_matching",
        "noise_floor",
    ]:
        path = os.path.join(base, f"{name}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing LOSO file: {path}")
        with open(path) as f:
            data[name] = json.load(f)
    data["sub_id"] = sub_id
    return data


def load_split_half_data(parcellation, sub_id):
    """Load all split-half reliability data for a single subject.

    Returns dict with keys:
        half_invariants, split_half_reliability, hungarian_matching
    """
    base = os.path.join(
        SCRATCH_DIR, "output", "04rb_split_half", parcellation, sub_id
    )
    data = {}
    for name in [
        "half_invariants",
        "split_half_reliability",
        "hungarian_matching",
    ]:
        path = os.path.join(base, f"{name}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing split-half file: {path}")
        with open(path) as f:
            data[name] = json.load(f)
    data["sub_id"] = sub_id
    return data


def load_primary_data(parcellation, sub_id, vt="0.95"):
    """Load primary model data (best_model.pkl + final_results.json).

    Returns dict with keys: model, final_results, n_states
    """
    base = os.path.join(
        SCRATCH_DIR, "output", "04_combined_hdphmm",
        parcellation, sub_id, "final", f"vt{vt}",
    )
    data = {"sub_id": sub_id}

    results_path = os.path.join(base, "final_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            data["final_results"] = json.load(f)
        data["n_states"] = data["final_results"]["model_info"]["n_states"]

    model_path = os.path.join(base, "best_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            data["model"] = pickle.load(f)

    return data


def load_loso_fold_model(parcellation, sub_id, season, vt="0.95"):
    """Load a single LOSO fold's best_model.pkl."""
    path = os.path.join(
        SCRATCH_DIR, "output", "04_combined_hdphmm",
        parcellation, sub_id, "loso", f"season_{season}", "best_model.pkl",
    )
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_split_half_model(parcellation, sub_id, half):
    """Load a split-half model."""
    path = os.path.join(
        SCRATCH_DIR, "output", "04_combined_hdphmm",
        parcellation, sub_id, "split_half", half, "best_model.pkl",
    )
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# =============================================================================
# Save helper
# =============================================================================

def save_fig(fig, out_dir, name):
    """Save figure as both PNG and PDF at 300 DPI."""
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("Saved %s", path)
    plt.close(fig)


# =============================================================================
# Fig 1 - State Spatial Reproducibility
# =============================================================================

def fig1_spatial_reproducibility(all_loso, all_sh, out_dir):
    """Violin + strip of Hungarian-matched Pearson r, LOSO vs split-half.

    Multi-subject 2x3 layout.
    """
    subjects = [d["sub_id"] for d in all_loso]
    n_sub = len(subjects)
    n_cols = 3
    n_rows = 2 if n_sub > 3 else 1

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.5 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for i, (loso, sh) in enumerate(zip(all_loso, all_sh)):
        ax = axes_flat[i]
        sub_id = loso["sub_id"]

        # LOSO: pool matched_correlations across all folds
        loso_corrs = []
        per_fold = loso["hungarian_matching"]["per_fold"]
        for season_key in per_fold:
            loso_corrs.extend(per_fold[season_key]["matched_correlations"])
        loso_corrs = np.array(loso_corrs)

        # Split-half: per-pair correlation
        sh_corrs = np.array([
            p["correlation"] for p in sh["hungarian_matching"]["matching"]["pairs"]
        ])

        # Combine for violin
        all_data = [loso_corrs, sh_corrs]
        labels = ["LOSO", "Split-half"]
        positions = [0, 1]

        parts = ax.violinplot(
            all_data, positions=positions, showextrema=False, showmedians=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("lightsteelblue")
            pc.set_edgecolor("grey")
            pc.set_alpha(0.5)

        # Strip (jittered points)
        for j, (vals, pos) in enumerate(zip(all_data, positions)):
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(
                np.full(len(vals), pos) + jitter, vals,
                s=6, alpha=0.4, color="steelblue", zorder=3,
            )

        # Mean annotations
        for j, (vals, pos) in enumerate(zip(all_data, positions)):
            mean_r = np.mean(vals)
            ax.plot(pos, mean_r, "D", color="firebrick", markersize=5, zorder=5)
            ax.annotate(
                f"r={mean_r:.2f}", (pos, mean_r),
                textcoords="offset points", xytext=(12, 0),
                fontsize=7, color="firebrick", ha="left", va="center",
            )

        # Reference line
        ax.axhline(0.3, ls="--", color="grey", lw=0.8, zorder=1)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Matched Pearson r")
        ax.set_title(sub_id)
        ax.set_ylim(-0.1, 1.05)

    # Hide unused axes
    for j in range(n_sub, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("State Spatial Reproducibility", fontsize=12, y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig1_spatial_reproducibility")


# =============================================================================
# Fig 2 - Scalar Invariant Stability
# =============================================================================

def fig2_scalar_stability(all_loso, all_sh, out_dir):
    """2x2 grid showing K_active, transition_entropy, self_transition_prob, median_dwell.

    Per-subject x-axis. Each subject shows:
      - Grey vertical bar for seed min-max (noise floor)
      - Colored dots for LOSO fold values
      - Triangle markers for split-half A/B
    """
    metrics = ["k_active", "transition_entropy", "self_transition_prob", "dwell_median_tr"]
    # Keys in fold_invariants vs half_invariants
    loso_keys = {
        "k_active": "k_active",
        "transition_entropy": "transition_entropy",
        "self_transition_prob": "self_transition_prob",
        "dwell_median_tr": "dwell_median_tr",
    }
    sh_keys = {
        "k_active": "K_active",
        "transition_entropy": "transition_entropy",
        "self_transition_prob": "self_transition_prob",
        "dwell_median_tr": "median_dwell",
    }
    noise_keys = {
        "k_active": "k_active",
        "transition_entropy": "transition_entropy",
        "self_transition_prob": "self_transition_prob",
        "dwell_median_tr": None,  # not available
    }

    subjects = [d["sub_id"] for d in all_loso]
    n_sub = len(subjects)
    x_positions = np.arange(n_sub)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes_flat = axes.flatten()

    for mi, metric in enumerate(metrics):
        ax = axes_flat[mi]

        for si, (loso, sh) in enumerate(zip(all_loso, all_sh)):
            fold_inv = loso["fold_invariants"]
            half_inv = sh["half_invariants"]
            noise = loso["noise_floor"]
            x = x_positions[si]

            # Noise floor (seed range) - grey bar
            nk = noise_keys[metric]
            if nk is not None and nk in noise:
                seed_range = noise[nk]["range"]
                ax.fill_between(
                    [x - 0.3, x + 0.3],
                    seed_range[0], seed_range[1],
                    color="lightgrey", alpha=0.6, zorder=1,
                )

            # LOSO fold values - colored dots
            fold_vals = []
            seasons = sorted(fold_inv.keys(), key=int)
            for si_fold, season_key in enumerate(seasons):
                val = fold_inv[season_key].get(loso_keys[metric])
                if val is not None:
                    fold_vals.append(val)
                    color_idx = int(season_key) - 1
                    ax.scatter(
                        x, val, s=30, color=FOLD_COLORS[color_idx],
                        edgecolors="black", linewidths=0.4, zorder=3,
                    )

            # Split-half A/B - triangle markers
            for half_label, marker_offset in [("A", -0.12), ("B", 0.12)]:
                val = half_inv[half_label].get(sh_keys[metric])
                if val is not None:
                    ax.scatter(
                        x + marker_offset, val, s=50,
                        marker="^", color="coral", edgecolors="black",
                        linewidths=0.4, zorder=4,
                    )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([s.replace("sub-0", "S") for s in subjects], fontsize=8)
        ax.set_ylabel(METRIC_DISPLAY[metric])
        ax.set_title(METRIC_DISPLAY[metric])

        # Dwell time hemodynamic annotation
        if metric == "dwell_median_tr":
            ax.text(
                0.02, 0.98, "~3 TR \u2248 4.5 s\n(hemodynamic timescale)",
                transform=ax.transAxes, fontsize=6,
                va="top", ha="left", alpha=0.6,
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
            )

    fig.suptitle("Scalar Invariant Stability", fontsize=12, y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig2_scalar_stability")


# =============================================================================
# Fig 3 - Recurrence Profile Reliability
# =============================================================================

def fig3_recurrence_reliability(all_sh, out_dir):
    """Scatter of matched half-A vs half-B recurrence scores.

    Multi-subject 2x3. Only above-threshold pairs from Hungarian matching.
    """
    subjects = [d["sub_id"] for d in all_sh]
    n_sub = len(subjects)
    n_cols = 3
    n_rows = 2 if n_sub > 3 else 1

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.5 * n_cols, 3.3 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for i, sh in enumerate(all_sh):
        ax = axes_flat[i]
        sub_id = sh["sub_id"]
        half_inv = sh["half_invariants"]
        pairs = sh["hungarian_matching"]["matching"]["pairs"]
        rec_a = half_inv["A"]["recurrence_scores"]
        rec_b = half_inv["B"]["recurrence_scores"]

        # Collect above-threshold matched pairs
        xs, ys, mean_recs = [], [], []
        for pair in pairs:
            if not pair.get("above_threshold", False):
                continue
            sa = pair["state_A"]
            sb = pair["state_B"]
            ra = rec_a[sa]
            rb = rec_b[sb]
            xs.append(ra)
            ys.append(rb)
            mean_recs.append(pair.get("mean_recurrence", (ra + rb) / 2))

        xs = np.array(xs)
        ys = np.array(ys)
        mean_recs = np.array(mean_recs)

        if len(xs) == 0:
            ax.text(0.5, 0.5, "No matched pairs", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(sub_id)
            continue

        # Scatter colored by mean recurrence
        norm = mcolors.Normalize(vmin=0, vmax=1)
        sc = ax.scatter(
            xs, ys, c=mean_recs, cmap=RECURRENCE_CMAP, norm=norm,
            s=25, edgecolors="black", linewidths=0.3, zorder=3,
        )

        # Identity line
        lim = [0, max(xs.max(), ys.max()) * 1.05]
        lim[1] = max(lim[1], 0.1)
        ax.plot(lim, lim, "--", color="black", lw=0.8, alpha=0.5, zorder=1)

        # Theil-Sen regression
        if len(xs) >= 3:
            slope, intercept, lo, hi = theilslopes(ys, xs)
            if np.isfinite(slope) and np.isfinite(intercept):
                x_fit = np.linspace(xs.min(), xs.max(), 50)
                y_fit = slope * x_fit + intercept
                ax.plot(x_fit, y_fit, "-", color="firebrick", lw=1.5, zorder=2)

        # Annotate Spearman rho from reliability JSON
        rec_corr = sh["split_half_reliability"].get("matched_recurrence_correlation", {})
        rho = rec_corr.get("raw_spearman", None)
        pval = rec_corr.get("raw_pvalue", None)
        if rho is not None and pval is not None:
            p_str = f"{pval:.1e}" if pval < 0.001 else f"{pval:.3f}"
            ax.text(
                0.05, 0.95,
                f"$\\rho$={rho:.3f}\np={p_str}\nn={len(xs)}",
                transform=ax.transAxes, fontsize=7,
                va="top", ha="left",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

        ax.set_xlabel("Half-A recurrence")
        ax.set_ylabel("Half-B recurrence")
        ax.set_title(sub_id)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    # Shared colorbar on last visible axis
    cbar_ax = axes_flat[n_sub - 1]
    make_recurrence_colorbar(cbar_ax, vmin=0, vmax=1, label="Mean recurrence")

    for j in range(n_sub, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Recurrence Profile Reliability", fontsize=12, y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig3_recurrence_reliability")


# =============================================================================
# Fig 4 - LOSO Recurrence CDFs
# =============================================================================

def fig4_loso_recurrence_cdfs(all_loso, out_dir):
    """Overlaid empirical CDFs of recurrence scores, one per LOSO fold.

    Multi-subject 2x3.
    """
    subjects = [d["sub_id"] for d in all_loso]
    n_sub = len(subjects)
    n_cols = 3
    n_rows = 2 if n_sub > 3 else 1

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.5 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for i, loso in enumerate(all_loso):
        ax = axes_flat[i]
        sub_id = loso["sub_id"]
        fold_inv = loso["fold_invariants"]
        seasons = sorted(fold_inv.keys(), key=int)

        for si, season_key in enumerate(seasons):
            rec_scores = np.array(fold_inv[season_key]["recurrence_scores_active"])
            rec_sorted = np.sort(rec_scores)
            cdf = np.arange(1, len(rec_sorted) + 1) / len(rec_sorted)

            color_idx = int(season_key) - 1
            ax.step(
                rec_sorted, cdf, where="post",
                color=FOLD_COLORS[color_idx], lw=1.2,
                label=f"S{season_key}",
            )

        # Mean KS statistic annotation
        ks_data = loso["cross_fold_consistency"].get("recurrence_ks_tests", {})
        mean_ks = ks_data.get("mean_ks_statistic", None)
        if mean_ks is not None:
            ax.text(
                0.95, 0.05,
                f"Mean KS = {mean_ks:.3f}",
                transform=ax.transAxes, fontsize=7,
                va="bottom", ha="right",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

        ax.set_xlabel("Recurrence score")
        ax.set_ylabel("Cumulative fraction")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_title(sub_id)
        ax.legend(fontsize=6, loc="upper left", ncol=2, framealpha=0.7)

    for j in range(n_sub, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("LOSO Recurrence CDFs", fontsize=12, y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig4_loso_recurrence_cdfs")


# =============================================================================
# Fig 5 - Network Composition
# =============================================================================

def fig5_network_composition(all_loso, all_sh, out_dir):
    """Grouped bars of network composition: LOSO mean +/- range and split-half A/B.

    Multi-subject 2x3.
    """
    subjects = [d["sub_id"] for d in all_loso]
    n_sub = len(subjects)
    n_cols = 3
    n_rows = 2 if n_sub > 3 else 1

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    n_nets = len(NETWORK_ORDER)
    bar_width = 0.25
    x_base = np.arange(n_nets)

    for i, (loso, sh) in enumerate(zip(all_loso, all_sh)):
        ax = axes_flat[i]
        sub_id = loso["sub_id"]
        fold_inv = loso["fold_invariants"]
        half_inv = sh["half_invariants"]
        seasons = sorted(fold_inv.keys(), key=int)

        # LOSO: collect per-network counts across folds
        loso_per_net = {net: [] for net in NETWORK_ORDER}
        for season_key in seasons:
            nc = fold_inv[season_key]["network_composition"]
            for net in NETWORK_ORDER:
                loso_per_net[net].append(nc.get(net, 0))

        loso_means = np.array([np.mean(loso_per_net[net]) for net in NETWORK_ORDER])
        loso_mins = np.array([np.min(loso_per_net[net]) for net in NETWORK_ORDER])
        loso_maxs = np.array([np.max(loso_per_net[net]) for net in NETWORK_ORDER])
        loso_err_lo = loso_means - loso_mins
        loso_err_hi = loso_maxs - loso_means

        # Split-half A and B
        sh_a_counts = np.array([
            half_inv["A"]["network_composition"].get(net, 0) for net in NETWORK_ORDER
        ])
        sh_b_counts = np.array([
            half_inv["B"]["network_composition"].get(net, 0) for net in NETWORK_ORDER
        ])

        # Bars
        ax.bar(
            x_base - bar_width, loso_means, bar_width,
            yerr=[loso_err_lo, loso_err_hi],
            color=[NETWORK_COLORS.get(net, "grey") for net in NETWORK_ORDER],
            edgecolor="black", linewidth=0.4, alpha=0.7,
            capsize=2, label="LOSO mean",
        )
        ax.bar(
            x_base, sh_a_counts, bar_width,
            color=[NETWORK_COLORS.get(net, "grey") for net in NETWORK_ORDER],
            edgecolor="black", linewidth=0.4, alpha=0.5,
            hatch="//", label="Half-A",
        )
        ax.bar(
            x_base + bar_width, sh_b_counts, bar_width,
            color=[NETWORK_COLORS.get(net, "grey") for net in NETWORK_ORDER],
            edgecolor="black", linewidth=0.4, alpha=0.5,
            hatch="\\\\", label="Half-B",
        )

        ax.set_xticks(x_base)
        ax.set_xticklabels(NETWORK_ORDER, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("State count")
        ax.set_title(sub_id)
        if i == 0:
            ax.legend(fontsize=6, loc="upper right", framealpha=0.7)

    for j in range(n_sub, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Network Composition Across Reliability Folds", fontsize=12, y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig5_network_composition")


# =============================================================================
# Fig 6a - Transition Scalar Stability (per-subject)
# =============================================================================

def fig6a_transition_scalars(loso, sh, sub_id, out_dir):
    """Two-row strip: transition entropy (top) and self-transition prob (bottom).

    Per-subject figure with LOSO fold values, split-half A/B, and seed range.
    """
    fold_inv = loso["fold_invariants"]
    half_inv = sh["half_invariants"]
    noise = loso["noise_floor"]
    seasons = sorted(fold_inv.keys(), key=int)

    metrics = [
        ("transition_entropy", "Transition entropy", "transition_entropy"),
        ("self_transition_prob", "Self-transition prob.", "self_transition_prob"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(4, 4.5))

    for mi, (metric_key, display_name, noise_key) in enumerate(metrics):
        ax = axes[mi]

        # Seed range (grey span)
        if noise_key in noise:
            seed_range = noise[noise_key]["range"]
            ax.axhspan(
                seed_range[0], seed_range[1],
                color="lightgrey", alpha=0.5, zorder=1,
                label="Seed range",
            )

        # LOSO fold values - strip/swarm
        fold_vals = []
        for season_key in seasons:
            val = fold_inv[season_key].get(metric_key)
            if val is not None:
                fold_vals.append(val)
                color_idx = int(season_key) - 1
                jitter = np.random.default_rng(int(season_key)).uniform(-0.15, 0.15)
                ax.scatter(
                    jitter, val, s=40,
                    color=FOLD_COLORS[color_idx],
                    edgecolors="black", linewidths=0.4,
                    zorder=3, label=f"S{season_key}" if mi == 0 else None,
                )

        # Split-half A/B - triangles
        for half_label, x_offset in [("A", 0.6), ("B", 0.85)]:
            sh_key = metric_key
            val = half_inv[half_label].get(sh_key)
            if val is not None:
                ax.scatter(
                    x_offset, val, s=60,
                    marker="^", color="coral", edgecolors="black",
                    linewidths=0.4, zorder=4,
                    label=f"Half-{half_label}" if mi == 0 else None,
                )

        ax.set_ylabel(display_name)
        ax.set_xlim(-0.5, 1.2)
        ax.set_xticks([0.0, 0.725])
        ax.set_xticklabels(["LOSO folds", "Split-half"], fontsize=8)

        if mi == 0:
            ax.legend(
                fontsize=6, loc="upper right", ncol=3, framealpha=0.7,
            )

    fig.suptitle(f"Transition Scalar Stability - {sub_id}", fontsize=11)
    fig.tight_layout()
    sub_out = os.path.join(out_dir, sub_id)
    save_fig(fig, sub_out, f"fig6a_transition_scalars_{sub_id}")


# =============================================================================
# Fig 6b - Transition Matrix Comparison (per-subject)
# =============================================================================

def fig6b_transition_matrices(loso, sh, primary, sub_id, parcellation, out_dir):
    """1x3 heatmaps: primary | best LOSO fold | split-half A.

    Active states only. Log-scale normalization.
    """
    # Primary model transmat
    primary_model = primary.get("model")
    if primary_model is None:
        logger.warning("Skipping fig6b for %s: no primary model loaded", sub_id)
        return

    if not hasattr(primary_model, "transmat_"):
        logger.warning("Skipping fig6b for %s: model has no transmat_", sub_id)
        return

    primary_transmat = primary_model.transmat_

    n_total = primary_transmat.shape[0]

    def _get_active_transmat(transmat, threshold=0.001):
        """Extract active-state submatrix from a transition matrix."""
        diag = np.diag(transmat)
        active = np.where(diag > threshold)[0]
        if len(active) == 0:
            return None, 0
        return transmat[np.ix_(active, active)], len(active)

    # Best LOSO fold (highest mean correlation)
    per_fold = loso["hungarian_matching"]["per_fold"]
    best_season = max(per_fold, key=lambda s: per_fold[s]["correlation_mean"])
    loso_model = load_loso_fold_model(parcellation, sub_id, best_season)

    # Split-half A model
    sh_model = load_split_half_model(parcellation, sub_id, "A")

    # Collect matrices - each model uses its own active states
    panels = []
    titles = []

    # Primary
    mat, n_act = _get_active_transmat(primary_transmat)
    if mat is not None:
        panels.append(mat)
        titles.append(f"Primary (K={n_act})")

    # Best LOSO fold
    if loso_model is not None and hasattr(loso_model, "transmat_"):
        mat, n_act = _get_active_transmat(loso_model.transmat_)
        if mat is not None:
            panels.append(mat)
            titles.append(f"LOSO S{best_season} (K={n_act})")
    else:
        logger.warning("LOSO model not found for %s S%s", sub_id, best_season)

    # Split-half A
    if sh_model is not None and hasattr(sh_model, "transmat_"):
        mat, n_act = _get_active_transmat(sh_model.transmat_)
        if mat is not None:
            panels.append(mat)
            titles.append(f"Half-A (K={n_act})")
    else:
        logger.warning("Split-half A model not found for %s", sub_id)

    n_panels = len(panels)
    if n_panels == 0:
        logger.warning("Skipping fig6b for %s: no transition matrices available", sub_id)
        return

    fig, axes = plt.subplots(
        1, n_panels, figsize=(4 * n_panels, 3.5),
        squeeze=False,
    )

    for pi, (mat, title) in enumerate(zip(panels, titles)):
        ax = axes[0, pi]
        # Log-scale: add small epsilon to avoid log(0)
        mat_log = np.log10(mat + 1e-10)
        vmin = -4
        vmax = 0
        im = ax.imshow(
            mat_log, cmap="viridis", aspect="equal",
            vmin=vmin, vmax=vmax,
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("To state")
        ax.set_ylabel("From state")
        fig.colorbar(im, ax=ax, label="log10(P)", shrink=0.8)

    fig.suptitle(
        f"Transition Matrices (active states) - {sub_id}",
        fontsize=11,
    )
    fig.tight_layout()
    sub_out = os.path.join(out_dir, sub_id)
    save_fig(fig, sub_out, f"fig6b_transition_matrices_{sub_id}")


# =============================================================================
# Fig 7 - FC Structure Stability
# =============================================================================

def fig7a_fc_split_half(all_sh, parcellation, out_dir):
    """Violin/box of per-matched-pair FC similarity from state_empirical_corr.npy.

    Multi-subject 2x3. Skips gracefully if FC data is unavailable.
    """
    from utils.state_fc import compute_fc_similarity_pairs

    subjects = [d["sub_id"] for d in all_sh]
    n_sub = len(subjects)
    n_cols = 3
    n_rows = 2 if n_sub > 3 else 1

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.5 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    any_plotted = False
    for i, sh in enumerate(all_sh):
        ax = axes_flat[i]
        sub_id = sh["sub_id"]

        # Try loading FC data for both halves
        fc_a_path = os.path.join(
            SCRATCH_DIR, "output", "04_combined_hdphmm",
            parcellation, sub_id, "split_half", "A", "state_empirical_corr.npy",
        )
        fc_b_path = os.path.join(
            SCRATCH_DIR, "output", "04_combined_hdphmm",
            parcellation, sub_id, "split_half", "B", "state_empirical_corr.npy",
        )

        # Also try under 05f_state_fc split_half paths
        if not os.path.exists(fc_a_path):
            fc_a_path = os.path.join(
                SCRATCH_DIR, "output", "05f_state_fc",
                parcellation, sub_id, "split_half_A", "state_empirical_corr.npy",
            )
        if not os.path.exists(fc_b_path):
            fc_b_path = os.path.join(
                SCRATCH_DIR, "output", "05f_state_fc",
                parcellation, sub_id, "split_half_B", "state_empirical_corr.npy",
            )

        if not os.path.exists(fc_a_path) or not os.path.exists(fc_b_path):
            ax.text(
                0.5, 0.5, "FC data\nnot available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="grey",
            )
            ax.set_title(sub_id)
            continue

        corr_a = np.load(fc_a_path)
        corr_b = np.load(fc_b_path)
        pairs = sh["hungarian_matching"]["matching"]["pairs"]

        fc_sims = compute_fc_similarity_pairs(corr_a, corr_b, pairs)
        fc_sims = np.array(fc_sims)
        fc_sims = fc_sims[np.isfinite(fc_sims)]  # Filter NaN/Inf

        if len(fc_sims) == 0:
            ax.text(0.5, 0.5, "No valid FC pairs", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(sub_id)
            any_plotted = True
            continue

        # Violin + strip
        parts = ax.violinplot(
            fc_sims, positions=[0], showextrema=False, showmedians=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("lightsalmon")
            pc.set_edgecolor("grey")
            pc.set_alpha(0.5)

        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, size=len(fc_sims))
        ax.scatter(
            jitter, fc_sims,
            s=10, alpha=0.4, color="coral", zorder=3,
        )

        mean_fc = np.nanmean(fc_sims)
        ax.plot(0, mean_fc, "D", color="firebrick", markersize=5, zorder=5)
        ax.annotate(
            f"RV={mean_fc:.2f}", (0, mean_fc),
            textcoords="offset points", xytext=(15, 0),
            fontsize=7, color="firebrick", ha="left", va="center",
        )

        ax.axhline(0.3, ls="--", color="grey", lw=0.8, zorder=1)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_ylabel("FC similarity (RV)")
        ax.set_title(sub_id)
        any_plotted = True

    for j in range(n_sub, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if not any_plotted:
        logger.warning("Fig 7: No FC data available for any subject. Skipping save.")
        plt.close(fig)
        return

    fig.suptitle("FC Structure Stability (Split-Half)", fontsize=12, y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig7a_fc_split_half")


# =============================================================================
# Fig 7b - LOSO FC Similarity
# =============================================================================

def _load_primary_fc(parcellation, sub_id, vt="0.95"):
    """Load primary model FC from 05f output."""
    path = os.path.join(
        SCRATCH_DIR, "output", "05f_state_fc",
        parcellation, sub_id, f"vt{vt}", "state_empirical_corr.npy",
    )
    if not os.path.exists(path):
        return None
    return np.load(path)


def _remap_loso_pairs(matches):
    """Remap LOSO match dicts to state_A/state_B format for compute_fc_similarity_pairs."""
    return [
        {"state_A": m["primary_state"], "state_B": m["fold_state"]}
        for m in matches if m.get("well_matched", False)
    ]


def fig7b_fc_loso(all_loso, parcellation, out_dir):
    """LOSO FC similarity: RV between primary and fold FC for matched states.

    Multi-subject 2x3. Pools across all folds per subject.
    """
    from utils.state_fc import compute_fc_similarity_pairs

    subjects = [d["sub_id"] for d in all_loso]
    n_sub = len(subjects)
    n_cols = 3
    n_rows = 2 if n_sub > 3 else 1

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.5 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    any_plotted = False
    for i, loso in enumerate(all_loso):
        ax = axes_flat[i]
        sub_id = loso["sub_id"]

        primary_fc = _load_primary_fc(parcellation, sub_id)
        if primary_fc is None:
            ax.text(0.5, 0.5, "Primary FC\nnot available",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="grey")
            ax.set_title(sub_id)
            continue

        per_fold = loso["hungarian_matching"]["per_fold"]
        all_rv = []

        for season_key in sorted(per_fold.keys(), key=int):
            fold_fc_path = os.path.join(
                SCRATCH_DIR, "output", "04_combined_hdphmm",
                parcellation, sub_id, "loso", f"season_{season_key}",
                "state_empirical_corr.npy",
            )
            if not os.path.exists(fold_fc_path):
                continue

            fold_fc = np.load(fold_fc_path)
            pairs = _remap_loso_pairs(per_fold[season_key]["matches"])
            if not pairs:
                continue

            rv_vals = compute_fc_similarity_pairs(primary_fc, fold_fc, pairs)
            all_rv.extend(rv_vals)

        all_rv = np.array(all_rv)
        all_rv = all_rv[np.isfinite(all_rv)]

        if len(all_rv) == 0:
            ax.text(0.5, 0.5, "No valid FC pairs", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(sub_id)
            continue

        # Violin + strip
        parts = ax.violinplot(all_rv, positions=[0], showextrema=False, showmedians=False)
        for pc in parts["bodies"]:
            pc.set_facecolor("lightblue")
            pc.set_edgecolor("grey")
            pc.set_alpha(0.5)

        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, size=len(all_rv))
        ax.scatter(jitter, all_rv, s=10, alpha=0.4, color="steelblue", zorder=3)

        mean_rv = np.nanmean(all_rv)
        ax.plot(0, mean_rv, "D", color="darkblue", markersize=5, zorder=5)
        ax.annotate(
            f"RV={mean_rv:.2f}\nn={len(all_rv)}", (0, mean_rv),
            textcoords="offset points", xytext=(15, 0),
            fontsize=7, color="darkblue", ha="left", va="center",
        )

        ax.axhline(0.3, ls="--", color="grey", lw=0.8, zorder=1)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_ylabel("FC similarity (RV)")
        ax.set_title(sub_id)
        any_plotted = True

    for j in range(n_sub, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if not any_plotted:
        plt.close(fig)
        return

    fig.suptitle("FC Structure Stability (LOSO)", fontsize=12, y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig7b_fc_loso")


# =============================================================================
# Fig 7c - FC Similarity: LOSO vs Split-Half (combined)
# =============================================================================

def fig7c_fc_combined(all_loso, all_sh, parcellation, out_dir):
    """Side-by-side violins of LOSO vs split-half FC similarity.

    Multi-subject 2x3. Parallels Fig 1 (which compares mean correlations).
    """
    from utils.state_fc import compute_fc_similarity_pairs

    subjects = [d["sub_id"] for d in all_loso]
    n_sub = len(subjects)
    n_cols = 3
    n_rows = 2 if n_sub > 3 else 1

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.0 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for i, (loso, sh) in enumerate(zip(all_loso, all_sh)):
        ax = axes_flat[i]
        sub_id = loso["sub_id"]

        # --- LOSO FC similarities ---
        primary_fc = _load_primary_fc(parcellation, sub_id)
        loso_rv = []
        if primary_fc is not None:
            per_fold = loso["hungarian_matching"]["per_fold"]
            for season_key in sorted(per_fold.keys(), key=int):
                fold_fc_path = os.path.join(
                    SCRATCH_DIR, "output", "04_combined_hdphmm",
                    parcellation, sub_id, "loso", f"season_{season_key}",
                    "state_empirical_corr.npy",
                )
                if not os.path.exists(fold_fc_path):
                    continue
                fold_fc = np.load(fold_fc_path)
                pairs = _remap_loso_pairs(per_fold[season_key]["matches"])
                if pairs:
                    loso_rv.extend(compute_fc_similarity_pairs(primary_fc, fold_fc, pairs))

        loso_rv = np.array(loso_rv)
        loso_rv = loso_rv[np.isfinite(loso_rv)] if len(loso_rv) > 0 else loso_rv

        # --- Split-half FC similarities ---
        sh_rv = []
        fc_a_path = os.path.join(
            SCRATCH_DIR, "output", "04_combined_hdphmm",
            parcellation, sub_id, "split_half", "A", "state_empirical_corr.npy",
        )
        fc_b_path = os.path.join(
            SCRATCH_DIR, "output", "04_combined_hdphmm",
            parcellation, sub_id, "split_half", "B", "state_empirical_corr.npy",
        )
        if os.path.exists(fc_a_path) and os.path.exists(fc_b_path):
            corr_a = np.load(fc_a_path)
            corr_b = np.load(fc_b_path)
            pairs = sh["hungarian_matching"]["matching"]["pairs"]
            sh_rv = compute_fc_similarity_pairs(corr_a, corr_b, pairs)

        sh_rv = np.array(sh_rv)
        sh_rv = sh_rv[np.isfinite(sh_rv)] if len(sh_rv) > 0 else sh_rv

        # --- Plot side-by-side ---
        data_groups = []
        positions = []
        colors = []
        labels = []

        if len(loso_rv) > 0:
            data_groups.append(loso_rv)
            positions.append(0)
            colors.append("steelblue")
            labels.append("LOSO")
        if len(sh_rv) > 0:
            data_groups.append(sh_rv)
            positions.append(1)
            colors.append("coral")
            labels.append("Split-half")

        for di, (data, pos, col) in enumerate(zip(data_groups, positions, colors)):
            parts = ax.violinplot(data, positions=[pos], showextrema=False, showmedians=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(col)
                pc.set_edgecolor("grey")
                pc.set_alpha(0.4)

            jitter = np.random.default_rng(di).uniform(-0.1, 0.1, size=len(data))
            ax.scatter(pos + jitter, data, s=8, alpha=0.3, color=col, zorder=3)

            mean_val = np.nanmean(data)
            ax.plot(pos, mean_val, "D", color="black", markersize=4, zorder=5)
            ax.annotate(
                f"{mean_val:.2f}", (pos, mean_val),
                textcoords="offset points", xytext=(12, 0),
                fontsize=7, ha="left", va="center",
            )

        ax.axhline(0.3, ls="--", color="grey", lw=0.8, zorder=1)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("FC similarity (RV)")
        ax.set_title(sub_id)

    for j in range(n_sub, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("FC Reliability: LOSO vs Split-Half", fontsize=12, y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig7c_fc_combined")


# =============================================================================
# Fig 7d - FC Similarity vs Mean Correlation
# =============================================================================

def fig7d_fc_vs_mean(all_sh, parcellation, out_dir):
    """Scatter: Hungarian mean correlation (x) vs FC RV (y) for split-half pairs.

    Multi-subject 2x3. Tests whether mean-only matching captures FC structure.
    """
    from utils.state_fc import compute_fc_similarity_pairs
    from scipy.stats import spearmanr

    subjects = [d["sub_id"] for d in all_sh]
    n_sub = len(subjects)
    n_cols = 3
    n_rows = 2 if n_sub > 3 else 1

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.5 * n_cols, 3.3 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for i, sh in enumerate(all_sh):
        ax = axes_flat[i]
        sub_id = sh["sub_id"]

        # Load FC
        fc_a_path = os.path.join(
            SCRATCH_DIR, "output", "04_combined_hdphmm",
            parcellation, sub_id, "split_half", "A", "state_empirical_corr.npy",
        )
        fc_b_path = os.path.join(
            SCRATCH_DIR, "output", "04_combined_hdphmm",
            parcellation, sub_id, "split_half", "B", "state_empirical_corr.npy",
        )

        if not os.path.exists(fc_a_path) or not os.path.exists(fc_b_path):
            ax.text(0.5, 0.5, "FC data\nnot available",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="grey")
            ax.set_title(sub_id)
            continue

        corr_a = np.load(fc_a_path)
        corr_b = np.load(fc_b_path)
        pairs = sh["hungarian_matching"]["matching"]["pairs"]

        fc_rvs = compute_fc_similarity_pairs(corr_a, corr_b, pairs)

        mean_corrs = []
        mean_recs = []
        fc_vals = []
        for pair, rv in zip(pairs, fc_rvs):
            if not np.isfinite(rv):
                continue
            mean_corrs.append(pair["correlation"])
            mean_recs.append(pair.get("mean_recurrence", 0.5))
            fc_vals.append(rv)

        mean_corrs = np.array(mean_corrs)
        fc_vals = np.array(fc_vals)
        mean_recs = np.array(mean_recs)

        if len(mean_corrs) < 3:
            ax.text(0.5, 0.5, "Too few pairs", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(sub_id)
            continue

        # Scatter colored by recurrence
        norm = mcolors.Normalize(vmin=0, vmax=1)
        ax.scatter(
            mean_corrs, fc_vals, c=mean_recs, cmap=RECURRENCE_CMAP, norm=norm,
            s=25, edgecolors="black", linewidths=0.3, zorder=3,
        )

        # Spearman annotation
        rho, pval = spearmanr(mean_corrs, fc_vals)
        p_str = f"{pval:.1e}" if pval < 0.001 else f"{pval:.3f}"
        ax.text(
            0.05, 0.95,
            f"$\\rho$={rho:.3f}\np={p_str}\nn={len(fc_vals)}",
            transform=ax.transAxes, fontsize=7,
            va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        ax.set_xlabel("Mean correlation (parcel space)")
        ax.set_ylabel("FC similarity (RV)")
        ax.set_title(sub_id)

    for j in range(n_sub, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("FC Similarity vs Mean Correlation (Split-Half)", fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig7d_fc_vs_mean")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate reliability visualisations for LOSO and split-half analyses.",
    )
    parser.add_argument(
        "--sub_id", type=str, default=None,
        help="Subject ID (default: all subjects). For per-subject figs only.",
    )
    parser.add_argument(
        "--parcellation", type=str, default="atlas-4S156Parcels",
        help="Parcellation name (default: atlas-4S156Parcels)",
    )
    parser.add_argument(
        "--skip_fc", action="store_true",
        help="Skip Fig 7 (FC stability) if FC not computed yet.",
    )
    args = parser.parse_args()

    parcellation = normalize_parcellation_name(args.parcellation)
    apply_publication_style()

    # Determine subjects
    if args.sub_id:
        subjects = [args.sub_id]
    else:
        subjects = ALL_SUBJECTS

    # Load data for all requested subjects
    all_loso = []
    all_sh = []
    all_primary = []
    for sub_id in subjects:
        try:
            loso = load_loso_data(parcellation, sub_id)
            all_loso.append(loso)
        except FileNotFoundError as e:
            logger.warning("Skipping %s LOSO: %s", sub_id, e)
            continue

        try:
            sh = load_split_half_data(parcellation, sub_id)
            all_sh.append(sh)
        except FileNotFoundError as e:
            logger.warning("Skipping %s split-half: %s", sub_id, e)
            # Remove the LOSO entry so indices stay aligned
            all_loso.pop()
            continue

        try:
            primary = load_primary_data(parcellation, sub_id)
            all_primary.append(primary)
        except Exception as e:
            logger.warning("Could not load primary model for %s: %s", sub_id, e)
            all_primary.append({"sub_id": sub_id})

    if not all_loso:
        logger.error("No subjects with complete data. Exiting.")
        sys.exit(1)

    logger.info(
        "Loaded data for %d subjects: %s",
        len(all_loso),
        [d["sub_id"] for d in all_loso],
    )

    # Output directories
    multi_out = os.path.join(
        SCRATCH_DIR, "output", "04rv_reliability_vis", parcellation,
    )
    per_sub_out = multi_out  # per-subject figs go into {multi_out}/{sub_id}/

    # ── Multi-subject figures ─────────────────────────────────────────────
    logger.info("--- Fig 1: State Spatial Reproducibility ---")
    fig1_spatial_reproducibility(all_loso, all_sh, multi_out)

    logger.info("--- Fig 2: Scalar Invariant Stability ---")
    fig2_scalar_stability(all_loso, all_sh, multi_out)

    logger.info("--- Fig 3: Recurrence Profile Reliability ---")
    fig3_recurrence_reliability(all_sh, multi_out)

    logger.info("--- Fig 4: LOSO Recurrence CDFs ---")
    fig4_loso_recurrence_cdfs(all_loso, multi_out)

    logger.info("--- Fig 5: Network Composition ---")
    fig5_network_composition(all_loso, all_sh, multi_out)

    # ── Per-subject figures ───────────────────────────────────────────────
    for loso, sh, primary in zip(all_loso, all_sh, all_primary):
        sub_id = loso["sub_id"]
        logger.info("--- Fig 6a: Transition Scalars - %s ---", sub_id)
        fig6a_transition_scalars(loso, sh, sub_id, per_sub_out)

        logger.info("--- Fig 6b: Transition Matrices - %s ---", sub_id)
        fig6b_transition_matrices(loso, sh, primary, sub_id, parcellation, per_sub_out)

    # ── Fig 7: FC stability (optional) ───────────────────────────────────
    if args.skip_fc:
        logger.info("--- Fig 7a-7d: Skipped (--skip_fc) ---")
    else:
        logger.info("--- Fig 7a: Split-Half FC Similarity ---")
        fig7a_fc_split_half(all_sh, parcellation, multi_out)

        logger.info("--- Fig 7b: LOSO FC Similarity ---")
        fig7b_fc_loso(all_loso, parcellation, multi_out)

        logger.info("--- Fig 7c: FC Combined (LOSO vs Split-Half) ---")
        fig7c_fc_combined(all_loso, all_sh, parcellation, multi_out)

        logger.info("--- Fig 7d: FC vs Mean Correlation ---")
        fig7d_fc_vs_mean(all_sh, parcellation, multi_out)

    logger.info("All figures complete.")


if __name__ == "__main__":
    main()
