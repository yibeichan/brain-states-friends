#!/usr/bin/env python3
"""
07b_physio_state_correspondence.py - Test physio-state correspondence.

Post-hoc analysis testing whether brain states have distinct autonomic signatures.
Physio was not used to define states, so any association is independent validation.

Supports multiple stimuli via --stimulus flag:
    friends   → uses 04 decoded states + 07a physio
    movie10   → uses m10_04 decoded states + m10_07a physio
    harrypotter → uses hp_04 decoded states + hp_07a physio

Recurrence scores (continuous) always come from the Friends recurrence analysis
(05a), since all stimuli use the Friends-trained HMM.

Five analyses:
    1. Per-state physio profiles (epoch-level, Kruskal-Wallis + pairwise MWU, FDR)
    2. Multi-lag analysis (-3 to +4 TRs)
    3. Transition-triggered averages (TTAs)
    4. Cross-episode consistency (Spearman: recurrence vs SD)
    5. Episode-level arousal vs state diversity

Prerequisites:
    - 07a physio features completed for the target stimulus
    - Decoded states (04 for friends, m10_04/hp_04 for others)
    - 05a recurrence_summary.json (Friends — always)

Outputs:
    {SCRATCH_DIR}/output/{output_dir}/{parcellation}/{sub_id}/
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
from scipy import stats
from scipy.stats import rankdata, pearsonr

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from utils.stats import benjamini_hochberg, partial_spearman
from utils.physio_io import FEATURE_COLUMNS
from utils.state_blocks import extract_state_block_records, load_eligible_states
from utils.state_flags_io import (
    load_state_flags, annotate_dataframe,
    CATEGORY_COLORS, CATEGORY_DISPLAY_NAMES, CATEGORY_MARKERS,
)
from utils.physio_qc import (
    load_run_percentage_valid, classify_channel_confidence,
    compute_qc_report, run_eda_mnar_diagnostic,
)

from dotenv import load_dotenv

load_dotenv()
SCRATCH_DIR = os.getenv("SCRATCH_DIR")
if SCRATCH_DIR is None:
    raise ValueError("SCRATCH_DIR must be set in the .env file")
DATA_DIR = os.getenv("DATA_DIR")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Stimulus configuration ─────────────────────────────────────────────────

STIMULUS_CONFIG = {
    "friends": {
        "decoded_dir": "04_combined_hdphmm",
        "decoded_subpath": ["final"],
        "physio_dir": "07a_physio_features",
        "output_dir": "07b_physio_state_correspondence",
    },
    "movie10": {
        "decoded_dir": "m10_04_decoded",
        "decoded_subpath": [],
        "physio_dir": "m10_07a_physio_features",
        "output_dir": "m10_07b_physio_state_correspondence",
    },
    "harrypotter": {
        "decoded_dir": "hp_04_decoded",
        "decoded_subpath": [],
        "physio_dir": "hp_07a_physio_features",
        "output_dir": "hp_07b_physio_state_correspondence",
    },
}


def _make_safe_filename(run_key):
    """Extract short filename from BIDS-style run key (for m10/hp matching)."""
    import re

    m = re.search(r"task-(\w+?)(?:_run-(\d+))?(?:_space|$)", run_key)
    if m:
        task = m.group(1)
        run = m.group(2)
        return f"{task}_run-{run.zfill(2)}" if run else task
    return run_key.replace("/", "_").replace(" ", "_")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Physio-state correspondence analysis.",
    )
    parser.add_argument("--sub_id", type=str, required=True)
    parser.add_argument("--parcellation", type=str, default="atlas-4S156Parcels")
    parser.add_argument("--vt", type=str, default=None)
    parser.add_argument(
        "--stimulus", type=str, default="friends",
        choices=["friends", "movie10", "harrypotter"],
        help="Stimulus dataset (default: friends). Recurrence scores always "
             "come from Friends 05a recurrence analysis.",
    )
    parser.add_argument("--exclude_sub_hrf", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Exclude sub-HRF states from physio analyses "
                             "(default: False — all states included, annotated by category).")
    parser.add_argument("--sensitivity_analysis", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Run sensitivity analyses: EDA-excluded + high-confidence "
                             "(default: False).")
    parser.add_argument("--pv_threshold", type=float, default=0.80,
                        help="PercentageValid threshold for per-channel confidence "
                             "(default: 0.80, physprep's validated boundary).")
    return parser.parse_args()


# ── Data Loading ──────────────────────────────────────────────────────────


def _load_physio_from_dir(physio_dir, decoded_states, suffix="_physio_features"):
    """Load physio .npy files matching decoded_states keys.

    Tries exact filename match first, then safe filename (for m10/hp BIDS keys).
    """
    features = {}
    for run_id in decoded_states:
        path = os.path.join(physio_dir, f"{run_id}{suffix}.npy")
        if os.path.exists(path):
            features[run_id] = np.load(path)
            continue
        safe = _make_safe_filename(run_id)
        path2 = os.path.join(physio_dir, f"{safe}{suffix}.npy")
        if os.path.exists(path2):
            features[run_id] = np.load(path2)
            continue
        if suffix == "_physio_features":
            logger.warning("No physio features for run %s", run_id)
    return features


def load_inputs(sub_id, parc, vt, stimulus="friends"):
    """Load decoded states, physio features, and recurrence summary."""
    config = STIMULUS_CONFIG[stimulus]
    vt_subdir = f"vt{vt}" if vt else ""

    # Decoded states (stimulus-specific)
    ds_path = os.path.join(
        SCRATCH_DIR, "output", config["decoded_dir"], parc, sub_id,
        *config["decoded_subpath"], vt_subdir, "decoded_states.pkl",
    ).replace("//", "/")
    with open(ds_path, "rb") as f:
        decoded_states = pickle.load(f)

    # Recurrence summary (always from Friends 05a)
    rec_path = os.path.join(
        SCRATCH_DIR, "output", "05a_recurrence_analysis", parc, sub_id,
        vt_subdir, "recurrence_summary.json",
    ).replace("//", "/")
    with open(rec_path) as f:
        rec_summary = json.load(f)

    # Physio features (stimulus-specific)
    physio_dir = os.path.join(
        SCRATCH_DIR, "output", config["physio_dir"], sub_id
    )
    physio_features = _load_physio_from_dir(physio_dir, decoded_states)
    physio_features_raw = _load_physio_from_dir(
        physio_dir, decoded_states, suffix="_physio_features_raw"
    )

    logger.info(
        "Loaded (%s): %d decoded runs, %d with physio, %d with raw features",
        stimulus, len(decoded_states),
        len(physio_features), len(physio_features_raw),
    )
    return decoded_states, physio_features, physio_features_raw, rec_summary


def load_recurrence_scores(rec_summary):
    """Extract recurrence_scores array from recurrence summary."""
    return np.array(rec_summary["recurrence_scores"], dtype=float)


# ── Analysis 1: Per-state physio profiles ─────────────────────────────────


def analysis_1_state_profiles(
    decoded_states, physio_features, recurrence_scores, out_dir,
    state_flags=None, feature_cols=None,
):
    """Per-state physio profiles using epoch-level aggregation.

    Kruskal-Wallis omnibus per feature, pairwise Mann-Whitney U with FDR.
    All active states (recurrence > 0) are included.

    Design choice — equal-weight epochs:
        Each contiguous state block (epoch) contributes one observation
        regardless of its duration (number of TRs).  This treats each epoch
        as one realization of a brain state rather than weighting by
        persistence.  Duration-weighting would conflate state persistence
        with state identity, biasing results toward long-lived (typically
        more recurrent) states.  This is consistent with standard HMM
        state-analysis conventions (Vidaurre et al., 2017).

    Parameters
    ----------
    feature_cols : list[str] | None
        Subset of FEATURE_COLUMNS to analyse (for sensitivity runs).
        None → all 7 features.
    """
    feat_cols = feature_cols or FEATURE_COLUMNS
    feat_indices = [FEATURE_COLUMNS.index(f) for f in feat_cols]
    logger.info("Analysis 1: Per-state physio profiles")
    from utils.plot_style import recurrence_color, make_recurrence_colorbar

    # Extract epoch-level means
    block_records = extract_state_block_records(decoded_states, recurrence_scores)

    epoch_data = []
    for rec in block_records:
        run_id = rec["run_id"]
        if run_id not in physio_features:
            continue
        feats = physio_features[run_id]
        start = rec["start_tr"]
        end = rec["end_tr"]
        if end <= start or start >= len(feats):
            continue
        n_trs_epoch = end - start
        epoch_mean = np.nanmean(feats[start:end], axis=0)
        epoch_data.append({
            "state": rec["state"],
            "recurrence_score": rec["recurrence_score"],
            "run_id": run_id,
            "n_trs": n_trs_epoch,
            **{FEATURE_COLUMNS[i]: epoch_mean[i] for i in feat_indices},
        })

    if not epoch_data:
        logger.warning("No epoch data — skipping Analysis 1")
        return

    df = pd.DataFrame(epoch_data)
    df = annotate_dataframe(df, state_flags)
    df.to_csv(os.path.join(out_dir, "state_physio_profiles.csv"), index=False)

    # Kruskal-Wallis omnibus per feature (across states)
    active_states = sorted(df["state"].unique())
    n_states = len(active_states)
    if n_states < 2:
        logger.warning("Fewer than 2 active states — skipping KW tests")
        return

    # Per-state sample sizes
    state_n = df.groupby("state").agg(
        n_epochs=("n_trs", "count"), total_trs=("n_trs", "sum"),
    ).to_dict("index")

    kw_results = {"_sample_sizes": {str(k): v for k, v in state_n.items()}}
    for feat in feat_cols:
        groups = [
            df.loc[df["state"] == s, feat].dropna().values
            for s in active_states
        ]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            continue
        h_stat, p_val = stats.kruskal(*groups)
        kw_results[feat] = {"H": float(h_stat), "p": float(p_val), "n_groups": len(groups)}

    # FDR on omnibus p-values
    feat_names = [k for k in kw_results if not k.startswith("_")]
    if feat_names:
        raw_p = np.array([kw_results[f]["p"] for f in feat_names])
        fdr_p = benjamini_hochberg(raw_p)
        for i, f in enumerate(feat_names):
            kw_results[f]["p_fdr"] = float(fdr_p[i])

    with open(os.path.join(out_dir, "kruskal_wallis_results.json"), "w") as f:
        json.dump(kw_results, f, indent=2)

    # ── Pairwise Mann-Whitney U where omnibus is significant ──────────
    sig_features = [f for f in feat_names if kw_results[f].get("p_fdr", 1.0) < 0.05]
    pairwise_results = {}
    if sig_features:
        from itertools import combinations

        all_pairs_p = []  # collect for FDR across all pairs × features
        pair_records = []
        for feat in sig_features:
            for s1, s2 in combinations(active_states, 2):
                g1 = df.loc[df["state"] == s1, feat].dropna().values
                g2 = df.loc[df["state"] == s2, feat].dropna().values
                if len(g1) < 2 or len(g2) < 2:
                    continue
                u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                pair_records.append({
                    "feature": feat,
                    "state_1": int(s1),
                    "state_2": int(s2),
                    "U": float(u_stat),
                    "p": float(p_val),
                    "n_1": len(g1),
                    "n_2": len(g2),
                })
                all_pairs_p.append(p_val)

        if pair_records:
            fdr_p = benjamini_hochberg(np.array(all_pairs_p))
            for i, rec in enumerate(pair_records):
                rec["p_fdr"] = float(fdr_p[i])
            pairwise_results = pair_records

    with open(os.path.join(out_dir, "pairwise_mwu_results.json"), "w") as f:
        json.dump(pairwise_results, f, indent=2)
    logger.info("Pairwise MWU: %d significant features, %d pair tests", len(sig_features), len(pairwise_results))

    # Plot: per-state mean physio — 2-row grid with category strip + colorbar at bottom
    from matplotlib.patches import Patch
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    import matplotlib.gridspec as gridspec

    n_feat = len(feat_cols)
    n_cols = min(4, n_feat)
    n_rows = (n_feat + n_cols - 1) // n_cols  # ceil division

    state_means = df.groupby("state")[feat_cols].mean()
    state_cats = df.groupby("state")["summary_category"].first()
    sorted_states = sorted(state_means.index)
    x_positions = np.arange(len(sorted_states))

    # Layout: n_rows of (bar + strip) pairs, then a footer row for colorbar + legend
    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows + 1.2))
    outer = gridspec.GridSpec(
        n_rows + 1, 1, figure=fig,
        height_ratios=[1] * n_rows + [0.15],
        hspace=0.45,
    )

    bar_axes = []
    for row_idx in range(n_rows):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, n_cols, subplot_spec=outer[row_idx],
            height_ratios=[15, 1], hspace=0.0,
        )
        for col_idx in range(n_cols):
            feat_idx = row_idx * n_cols + col_idx
            if feat_idx >= n_feat:
                break
            feat = feat_cols[feat_idx]
            ax = fig.add_subplot(inner[0, col_idx])
            bar_axes.append(ax)
            for j, state in enumerate(sorted_states):
                val = state_means.loc[state, feat]
                if np.isfinite(val):
                    ax.bar(
                        j, val, width=0.8,
                        color=recurrence_color(recurrence_scores[state]),
                        edgecolor="white", linewidth=0.3,
                    )
            ax.set_xticks([])
            ax.set_ylabel(feat, fontsize=9)
            # Category strip directly below
            ax_strip = fig.add_subplot(inner[1, col_idx])
            for j, state in enumerate(sorted_states):
                cat = state_cats.get(state, "unknown")
                ax_strip.barh(0, 1, left=j - 0.5, height=1,
                              color=CATEGORY_COLORS.get(cat, "#999999"))
            ax_strip.set_xlim(-0.5, len(sorted_states) - 0.5)
            ax_strip.set_xticks(x_positions[::5])
            ax_strip.set_xticklabels(
                [str(sorted_states[k]) for k in range(0, len(sorted_states), 5)],
                fontsize=6,
            )
            ax_strip.set_yticks([])
            ax_strip.set_xlabel("State", fontsize=8)

    # Footer: horizontal colorbar + category legend
    footer = outer[n_rows].subgridspec(1, 2, width_ratios=[1, 1])
    ax_cbar = fig.add_subplot(footer[0, 0])
    sm = ScalarMappable(cmap="viridis", norm=Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cbar, orientation="horizontal")
    cb.set_label("Recurrence score", fontsize=9)

    ax_leg = fig.add_subplot(footer[0, 1])
    ax_leg.axis("off")
    legend_handles = [
        Patch(facecolor=CATEGORY_COLORS[c], edgecolor="black", linewidth=0.5,
              label=CATEGORY_DISPLAY_NAMES[c])
        for c in CATEGORY_COLORS if c in state_cats.values
    ]
    if legend_handles:
        ax_leg.legend(
            handles=legend_handles, loc="center",
            ncol=min(len(legend_handles), 3), fontsize=9, frameon=False,
        )

    fig.suptitle("Per-state physio profiles (colored by recurrence score)", fontsize=12, y=0.98)
    for fmt in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"state_physio_profiles.{fmt}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info("Analysis 1 complete: %d states, %d features tested", n_states, len(kw_results))


# ── Analysis 2: Multi-lag analysis ────────────────────────────────────────


def analysis_2_multilag(decoded_states, physio_features, recurrence_scores, active_states, out_dir, state_flags=None):
    """Shift physio relative to states by lags -3 to +4 TRs."""
    logger.info("Analysis 2: Multi-lag analysis")
    from utils.plot_style import recurrence_color

    lags = list(range(-3, 5))  # -3 to +4 TRs

    lag_results = []

    for lag in lags:
        for run_id, state_seq in decoded_states.items():
            if run_id not in physio_features:
                continue
            feats = physio_features[run_id]
            n_trs = min(len(state_seq), len(feats))

            for state in active_states:
                # State mask (where state is active)
                mask = np.array(state_seq[:n_trs]) == state
                # Shift physio by lag (negative lag = physio precedes BOLD)
                shifted_mask = np.zeros(n_trs, dtype=bool)
                if lag >= 0:
                    if lag < n_trs:
                        shifted_mask[lag:] = mask[: n_trs - lag]
                else:
                    if -lag < n_trs:
                        shifted_mask[: n_trs + lag] = mask[-lag:]

                if np.sum(shifted_mask) < 2:
                    continue

                mean_feats = np.nanmean(feats[:n_trs][shifted_mask], axis=0)
                lag_results.append({
                    "lag": lag,
                    "state": state,
                    "recurrence_score": float(recurrence_scores[state]),
                    "run_id": run_id,
                    "n_trs": int(np.sum(shifted_mask)),
                    **{FEATURE_COLUMNS[i]: mean_feats[i] for i in range(7)},
                })

    if not lag_results:
        logger.warning("No lag data — skipping Analysis 2")
        return

    df = pd.DataFrame(lag_results)
    df = annotate_dataframe(df, state_flags)
    df.to_csv(os.path.join(out_dir, "lag_analysis.csv"), index=False)

    # Plot: lag profiles for ALL active states, sorted by recurrence, colored by category
    all_sorted = sorted(active_states, key=lambda s: recurrence_scores[s], reverse=True)
    # Only include states that have lag data
    plot_states = [s for s in all_sorted if s in df["state"].values]
    if plot_states:
        n_plot = len(plot_states)
        fig, axes = plt.subplots(n_plot, 2, figsize=(10, 2.5 * n_plot))
        if n_plot == 1:
            axes = axes[np.newaxis, :]
        # Build category lookup
        cat_lookup = {}
        if state_flags is not None:
            cat_lookup = dict(zip(state_flags["state"], state_flags["summary_category"]))
        for idx, state in enumerate(plot_states):
            sub = df[df["state"] == state]
            cat = cat_lookup.get(state, "unknown")
            color = CATEGORY_COLORS.get(cat, "#999999")
            cat_label = CATEGORY_DISPLAY_NAMES.get(cat, cat)
            for col_idx, feat in enumerate(["HR_bpm", "RVT"]):
                ax = axes[idx, col_idx]
                lag_means = sub.groupby("lag")[feat].mean()
                lag_sems = sub.groupby("lag")[feat].sem()
                ax.errorbar(lag_means.index, lag_means.values, yerr=lag_sems.values,
                            marker="o", capsize=3, color=color, markersize=4)
                ax.set_xlabel("Lag (TRs)")
                ax.set_ylabel(feat)
                ax.set_title(
                    f"S{state} (rec={recurrence_scores[state]:.2f}, {cat_label})",
                    fontsize=9,
                )
                ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        # Category legend at top
        from matplotlib.patches import Patch
        present_cats = sorted(set(cat_lookup.get(s, "unknown") for s in plot_states))
        legend_handles = [
            Patch(facecolor=CATEGORY_COLORS.get(c, "#999999"), edgecolor="black",
                  linewidth=0.5, label=CATEGORY_DISPLAY_NAMES.get(c, c))
            for c in present_cats
        ]
        if legend_handles:
            fig.legend(
                handles=legend_handles, loc="upper center",
                bbox_to_anchor=(0.5, 1.0), ncol=len(legend_handles),
                fontsize=8, frameon=False,
            )
        fig.suptitle("Multi-lag profiles (all active states, sorted by recurrence)", y=1.02)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        for fmt in ("pdf", "png"):
            fig.savefig(os.path.join(out_dir, f"lag_profiles.{fmt}"),
                        dpi=200, bbox_inches="tight")
        plt.close(fig)

    logger.info("Analysis 2 complete: %d lag-state-run records", len(lag_results))


# ── Analysis 3: Transition-triggered averages ─────────────────────────────


def analysis_3_tta(decoded_states, physio_features, recurrence_scores, active_states, out_dir, state_flags=None):
    """Physio time courses around state transitions (all active states)."""
    logger.info("Analysis 3: Transition-triggered averages")
    from utils.plot_style import recurrence_color

    window = 10  # TRs before and after
    active_set = set(active_states)

    tta_into = {s: [] for s in active_states}
    tta_outof = {s: [] for s in active_states}

    for run_id, state_seq in decoded_states.items():
        if run_id not in physio_features:
            continue
        feats = physio_features[run_id]
        n_trs = min(len(state_seq), len(feats))
        state_seq = np.asarray(state_seq[:n_trs])

        # Find transition points
        transitions = np.flatnonzero(state_seq[1:] != state_seq[:-1]) + 1

        for t in transitions:
            if t - window < 0 or t + window >= n_trs:
                continue
            snippet = feats[t - window : t + window]  # shape (2*window, 7)

            state_after = int(state_seq[t])
            state_before = int(state_seq[t - 1])

            if state_after in active_set:
                tta_into[state_after].append(snippet)
            if state_before in active_set:
                tta_outof[state_before].append(snippet)

    # Compute mean TTAs
    tta_results = []
    for direction, tta_dict, label in [
        ("into", tta_into, "into"),
        ("outof", tta_outof, "outof"),
    ]:
        for state, snippets in tta_dict.items():
            if len(snippets) < 5:
                continue
            arr = np.array(snippets)  # (n_transitions, 2*window, 7)
            mean_tta = np.nanmean(arr, axis=0)
            for t_idx in range(2 * window):
                row = {
                    "state": state,
                    "recurrence_score": float(recurrence_scores[state]),
                    "direction": label,
                    "relative_tr": t_idx - window,
                    "n_transitions": len(snippets),
                }
                for i, feat in enumerate(FEATURE_COLUMNS):
                    row[feat] = float(mean_tta[t_idx, i])
                tta_results.append(row)

    if tta_results:
        df = pd.DataFrame(tta_results)
        df = annotate_dataframe(df, state_flags)
        df.to_csv(os.path.join(out_dir, "transition_triggered_averages.csv"), index=False)

        # Plot TTA for HR — 2 rows (into/outof), colored by category, SEM shading
        cat_lookup = {}
        if state_flags is not None:
            cat_lookup = dict(zip(state_flags["state"], state_flags["summary_category"]))

        plot_states = sorted(
            [s for s in active_states if len(tta_into.get(s, [])) >= 5],
            key=lambda s: recurrence_scores[s], reverse=True,
        )
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        legend_handles = []
        for ax, (direction, tta_dict_plot) in zip(axes, [("into", tta_into), ("outof", tta_outof)]):
            for state in plot_states:
                snippets = tta_dict_plot.get(state, [])
                if len(snippets) < 5:
                    continue
                arr = np.array(snippets)  # (n_transitions, 2*window, 7)
                hr_col = arr[:, :, 0]  # HR is column 0
                mean_hr = np.nanmean(hr_col, axis=0)
                sem_hr = np.nanstd(hr_col, axis=0) / np.sqrt(len(snippets))
                x = np.arange(len(mean_hr)) - window

                cat = cat_lookup.get(state, "unknown")
                color = CATEGORY_COLORS.get(cat, "#999999")
                cat_label = CATEGORY_DISPLAY_NAMES.get(cat, cat)
                line, = ax.plot(x, mean_hr, color=color, alpha=0.7, linewidth=1.2)
                ax.fill_between(x, mean_hr - sem_hr, mean_hr + sem_hr,
                                color=color, alpha=0.1)
                if direction == "into":  # collect legend entries once
                    legend_handles.append((line, f"S{state} ({cat_label})"))
            ax.axvline(0, color="gray", linestyle="--")
            ax.set_xlabel("TR relative to transition")
            ax.set_ylabel("HR (z-scored)")
            ax.set_title(f"Transitions {direction}")
        # Legend at bottom, outside plots
        if legend_handles:
            fig.legend(
                [h for h, _ in legend_handles],
                [l for _, l in legend_handles],
                loc="lower center", bbox_to_anchor=(0.5, -0.02),
                ncol=min(5, len(legend_handles)), fontsize=6, frameon=False,
            )
        fig.suptitle(f"Transition-triggered averages (HR, {len(plot_states)} states)")
        fig.tight_layout(rect=[0, 0.05, 1, 0.97])
        for fmt in ("pdf", "png"):
            fig.savefig(os.path.join(out_dir, f"tta_plots.{fmt}"),
                        dpi=200, bbox_inches="tight")
        plt.close(fig)

    logger.info("Analysis 3 complete: %d TTA records", len(tta_results))


# ── Analysis 4 helpers: n_episodes corrections ───────────────────────────


def _partial_spearman(x, y, covariate):
    """Partial Spearman correlation controlling for a covariate.

    Delegates to utils.stats.partial_spearman (shared implementation).
    """
    return partial_spearman(x, y, covariate)


def _matched_n_correlation(df, recurrence_scores, n_target,
                           n_resamples=1000, seed=42):
    """Matched-n subsampling: equalize episode counts, recompute SD, correlate.

    For each resample, subsample n_target runs per state (without replacement),
    recompute mean SD across 6 physio features, and Spearman-correlate with
    recurrence scores.
    """
    state_ep_counts = df.groupby("state")["run_id"].nunique()
    eligible_states = sorted(state_ep_counts[state_ep_counts >= n_target].index)

    if len(eligible_states) < 10:
        return {
            "median_rho": None, "ci95": None,
            "n_target": n_target, "n_states": len(eligible_states),
            "n_resamples": n_resamples,
            "note": f"Only {len(eligible_states)} states with >= {n_target} episodes",
        }

    rng = np.random.default_rng(seed)
    rhos = []

    for _ in range(n_resamples):
        mean_sds = []
        rec_scores = []
        for state in eligible_states:
            sub = df[df["state"] == state]
            runs = sub["run_id"].unique()
            chosen = rng.choice(runs, size=n_target, replace=False)
            sub_sampled = sub[sub["run_id"].isin(chosen)]
            sds = []
            for feat in FEATURE_COLUMNS[:6]:
                vals = sub_sampled[feat].dropna()
                if len(vals) >= 5:
                    sds.append(vals.std())
            if sds:
                mean_sds.append(np.mean(sds))
                rec_scores.append(float(recurrence_scores[state]))

        if len(mean_sds) >= 5:
            rho, _ = stats.spearmanr(rec_scores, mean_sds)
            rhos.append(rho)

    if not rhos:
        return {
            "median_rho": None, "ci95": None,
            "n_target": n_target, "n_states": len(eligible_states),
            "n_resamples": n_resamples, "note": "All resamples failed",
        }

    rhos = np.array(rhos)
    return {
        "median_rho": float(np.median(rhos)),
        "ci95": [float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))],
        "n_target": n_target,
        "n_states": len(eligible_states),
        "n_resamples": n_resamples,
    }


def _simulation_null(state_n_episodes, recurrence_scores, pooled_values,
                     n_features=6, n_resamples=1000, seed=43):
    """Simulation null: draw from pooled physio distribution, compute expected rho.

    For each resample, for each state draw n_episodes[state] rows from the
    pooled empirical distribution, compute SD per feature, average, and
    correlate with recurrence. Returns null distribution statistics.
    """
    rng = np.random.default_rng(seed)

    states_sorted = sorted(state_n_episodes.keys())
    n_eps = np.array([int(round(state_n_episodes[s])) for s in states_sorted])
    rec = np.array([float(recurrence_scores[s]) for s in states_sorted])
    n_pool = len(pooled_values)

    null_rhos = []
    for _ in range(n_resamples):
        mean_sds = []
        for n_ep in n_eps:
            n_ep = max(n_ep, 5)  # floor at 5 for meaningful SD
            idx = rng.integers(0, n_pool, size=(n_ep, n_features))
            sds = []
            for f in range(n_features):
                col_vals = pooled_values[idx[:, f], f] if pooled_values.ndim > 1 else pooled_values[idx[:, f]]
                sds.append(np.std(col_vals, ddof=1))
            mean_sds.append(np.mean(sds))

        rho, _ = stats.spearmanr(rec, mean_sds)
        null_rhos.append(rho)

    null_rhos = np.array(null_rhos)

    # Observed percentile computed by caller (needs raw rho from main analysis)
    return {
        "null_median_rho": float(np.median(null_rhos)),
        "null_95ci": [float(np.percentile(null_rhos, 2.5)),
                      float(np.percentile(null_rhos, 97.5))],
        "null_rhos": null_rhos,  # pass back for percentile calc in caller
        "n_resamples": n_resamples,
    }


# ── Analysis 4: Cross-episode consistency ─────────────────────────────────


def analysis_4_consistency(
    decoded_states, physio_features, recurrence_scores, active_states, out_dir,
    state_flags=None,
):
    """Cross-episode physio consistency: Spearman recurrence vs SD."""
    logger.info("Analysis 4: Cross-episode consistency")

    active_set = set(active_states)

    # Per-run, per-state mean physio
    run_state_means = []
    for run_id, state_seq in decoded_states.items():
        if run_id not in physio_features:
            continue
        feats = physio_features[run_id]
        n_trs = min(len(state_seq), len(feats))
        state_seq = np.asarray(state_seq[:n_trs])
        feats = feats[:n_trs]

        for state in np.unique(state_seq):
            state = int(state)
            if state not in active_set:
                continue
            mask = state_seq == state
            if np.sum(mask) < 2:
                continue
            mean_feats = np.nanmean(feats[mask], axis=0)
            run_state_means.append({
                "run_id": run_id,
                "state": state,
                "recurrence_score": float(recurrence_scores[state]),
                **{FEATURE_COLUMNS[i]: mean_feats[i] for i in range(7)},
            })

    if not run_state_means:
        logger.warning("No data for consistency analysis")
        return

    df = pd.DataFrame(run_state_means)
    df = annotate_dataframe(df, state_flags)
    df.to_csv(os.path.join(out_dir, "cross_episode_consistency.csv"), index=False)

    # Per-state SD across episodes (SD, not CV — CV is unstable on z-scored data)
    consistency = []
    for state in sorted(df["state"].unique()):
        sub = df[df["state"] == state]
        if len(sub) < 5:
            continue
        for feat in FEATURE_COLUMNS[:6]:  # skip SCR_binary
            vals = sub[feat].dropna()
            if len(vals) < 5:
                continue
            sd = float(vals.std())
            consistency.append({
                "state": int(state),
                "recurrence_score": float(recurrence_scores[state]),
                "feature": feat,
                "sd": sd,
                "n_episodes": len(vals),
            })

    if consistency:
        sd_df = pd.DataFrame(consistency)
        sd_df = annotate_dataframe(sd_df, state_flags)

        # Spearman: recurrence score vs mean SD per state
        state_mean_sd = sd_df.groupby("state").agg(
            mean_sd=("sd", "mean"),
            recurrence_score=("recurrence_score", "first"),
            mean_n_episodes=("n_episodes", "mean"),
        ).reset_index()
        state_mean_sd = annotate_dataframe(state_mean_sd, state_flags)

        corr_result = {}
        if len(state_mean_sd) >= 5:
            rho, p = stats.spearmanr(state_mean_sd["recurrence_score"],
                                     state_mean_sd["mean_sd"])
            corr_result = {"rho": float(rho), "p": float(p), "n": len(state_mean_sd)}

        # ── n_episodes corrections ──────────────────────────────────
        if len(state_mean_sd) >= 5 and state_mean_sd["mean_n_episodes"].nunique() > 1:
            # Collinearity diagnostic
            rec_n_rho, _ = stats.spearmanr(
                state_mean_sd["recurrence_score"],
                state_mean_sd["mean_n_episodes"],
            )
            corr_result["recurrence_n_episodes_rho"] = float(rec_n_rho)
            logger.info("  Collinearity recurrence~n_episodes: rho=%.3f", rec_n_rho)

            # 1. Partial Spearman controlling for n_episodes
            p_rho, p_p = _partial_spearman(
                state_mean_sd["recurrence_score"].values,
                state_mean_sd["mean_sd"].values,
                state_mean_sd["mean_n_episodes"].values,
            )
            corr_result["partial_rho"] = float(p_rho) if not np.isnan(p_rho) else None
            corr_result["partial_p"] = float(p_p) if not np.isnan(p_p) else None
            logger.info("  Partial Spearman: rho=%.3f, p=%.4f",
                        p_rho if not np.isnan(p_rho) else 0.0,
                        p_p if not np.isnan(p_p) else 1.0)

            # 2. Matched-n subsampling
            ep_counts = df.groupby("state")["run_id"].nunique()
            n_target = max(10, int(np.percentile(ep_counts, 25)))
            matched = _matched_n_correlation(df, recurrence_scores, n_target)
            corr_result["matched_n"] = matched
            logger.info("  Matched-n (n=%d): median_rho=%s",
                        n_target, matched.get("median_rho"))

            # 3. Simulation null
            pooled_values = df[FEATURE_COLUMNS[:6]].dropna().values
            state_n_ep_dict = dict(zip(
                state_mean_sd["state"].astype(int),
                state_mean_sd["mean_n_episodes"],
            ))
            null_result = _simulation_null(
                state_n_ep_dict, recurrence_scores, pooled_values,
            )
            # Compute observed percentile from raw rho
            null_rhos = null_result.pop("null_rhos")
            observed_rho = corr_result.get("rho", 0.0)
            null_result["observed_percentile"] = float(
                np.mean(null_rhos <= observed_rho)
            )
            null_result["p_sim"] = float(
                (np.sum(null_rhos <= observed_rho) + 1) / (len(null_rhos) + 1)
            )
            corr_result["simulation_null"] = null_result
            logger.info("  Simulation null: median=%.3f, observed_pctile=%.3f",
                        null_result["null_median_rho"],
                        null_result["observed_percentile"])

        with open(os.path.join(out_dir, "consistency_recurrence_correlation.json"), "w") as f:
            json.dump(corr_result, f, indent=2)

        # Scatter: recurrence vs mean SD — two-panel (raw + corrected)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left panel: raw correlation (existing visualization)
        for cat in CATEGORY_MARKERS:
            mask = state_mean_sd["summary_category"] == cat
            if not mask.any():
                continue
            sub = state_mean_sd[mask]
            ax1.scatter(
                sub["recurrence_score"], sub["mean_sd"],
                c=CATEGORY_COLORS.get(cat, "#999999"),
                marker=CATEGORY_MARKERS[cat],
                alpha=0.8, edgecolors="black", linewidths=0.5, s=60,
                label=CATEGORY_DISPLAY_NAMES.get(cat, cat),
            )
        ax1.set_xlabel("Recurrence score")
        ax1.set_ylabel("Mean SD (cross-episode)")
        ax1.legend(fontsize=6, title="Category", title_fontsize=6)
        if corr_result.get("rho") is not None:
            ax1.set_title(f"Raw: rho={corr_result['rho']:.3f}, p={corr_result['p']:.2e}")
        else:
            ax1.set_title("Raw: insufficient data")

        # Right panel: same scatter with n_episodes sizing + correction annotations
        n_eps = state_mean_sd["mean_n_episodes"].values
        sizes = 15 + 1.5 * n_eps  # scale point size by n_episodes
        ax2.scatter(
            state_mean_sd["recurrence_score"], state_mean_sd["mean_sd"],
            s=sizes, c=n_eps, cmap="viridis", alpha=0.8,
            edgecolors="black", linewidths=0.5,
        )
        cbar = fig.colorbar(ax2.collections[0], ax=ax2, shrink=0.8)
        cbar.set_label("n_episodes", fontsize=8)
        ax2.set_xlabel("Recurrence score")
        ax2.set_ylabel("Mean SD (cross-episode)")

        # Title: partial rho
        partial_rho = corr_result.get("partial_rho")
        partial_p = corr_result.get("partial_p")
        if partial_rho is not None:
            ax2.set_title(f"Partial (ctrl n_eps): rho={partial_rho:.3f}, p={partial_p:.2e}")
        else:
            ax2.set_title("Partial: n/a")

        # Annotation box with matched-n and simulation null
        annot_lines = []
        matched_n = corr_result.get("matched_n", {})
        if matched_n.get("median_rho") is not None:
            annot_lines.append(
                f"Matched-n (n={matched_n['n_target']}): "
                f"rho={matched_n['median_rho']:.3f} "
                f"[{matched_n['ci95'][0]:.3f}, {matched_n['ci95'][1]:.3f}]"
            )
        sim_null = corr_result.get("simulation_null", {})
        if sim_null.get("observed_percentile") is not None:
            annot_lines.append(
                f"Sim null: pctile={sim_null['observed_percentile']:.3f}, "
                f"null median={sim_null['null_median_rho']:.3f}"
            )
        if annot_lines:
            ax2.text(
                0.03, 0.97, "\n".join(annot_lines),
                transform=ax2.transAxes, fontsize=7,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
            )

        fig.tight_layout()
        for fmt in ("pdf", "png"):
            fig.savefig(os.path.join(out_dir, f"consistency_scatter.{fmt}"),
                        dpi=200, bbox_inches="tight")
        plt.close(fig)

    logger.info("Analysis 4 complete: %d consistency records", len(consistency))


# ── Analysis 5: Arousal vs state diversity ────────────────────────────────


def analysis_5_arousal_diversity(
    decoded_states, physio_features_raw, out_dir
):
    """Episode-level arousal (HR, EDA tonic) vs state diversity.

    Uses RAW (pre-z-score) physio features for arousal proxies.
    Z-scored features have ~0 mean per run by construction, so run-level
    means would be uninterpretable.
    """
    logger.info("Analysis 5: Arousal vs state diversity")

    episode_data = []
    for run_id, state_seq in decoded_states.items():
        if run_id not in physio_features_raw:
            continue
        feats = physio_features_raw[run_id]
        n_trs = min(len(state_seq), len(feats))
        state_seq = np.asarray(state_seq[:n_trs])

        n_unique = len(np.unique(state_seq))
        transitions = np.sum(state_seq[1:] != state_seq[:-1])
        switch_rate = float(transitions / max(n_trs - 1, 1))
        mean_hr = float(np.nanmean(feats[:n_trs, 0]))
        mean_eda = float(np.nanmean(feats[:n_trs, 4]))

        # Extract season from run_id (e.g., "s01e03a" → season 1)
        import re
        season_match = re.search(r"s(\d{2})", run_id)
        season = int(season_match.group(1)) if season_match else 0

        episode_data.append({
            "run_id": run_id,
            "season": season,
            "n_trs": n_trs,
            "n_unique_states": n_unique,
            "switch_rate": switch_rate,
            "mean_HR": mean_hr,
            "mean_EDA_tonic": mean_eda,
        })

    if not episode_data:
        logger.warning("No episode data for arousal-diversity analysis")
        return

    df = pd.DataFrame(episode_data)
    df.to_csv(os.path.join(out_dir, "arousal_diversity_correlation.csv"), index=False)

    # Spearman correlations (uncorrected + partial controlling for run length)
    corr_results = {}
    for arousal_feat in ["mean_HR", "mean_EDA_tonic"]:
        for diversity_feat in ["n_unique_states", "switch_rate"]:
            valid = df[[arousal_feat, diversity_feat, "n_trs"]].dropna()
            if len(valid) < 5:
                continue
            rho, p = stats.spearmanr(valid[arousal_feat], valid[diversity_feat])
            rho_partial, p_partial = partial_spearman(
                valid[arousal_feat].values,
                valid[diversity_feat].values,
                valid["n_trs"].values,
            )
            corr_results[f"{arousal_feat}_vs_{diversity_feat}"] = {
                "rho": float(rho),
                "p": float(p),
                "rho_partial_n_trs": float(rho_partial),
                "p_partial_n_trs": float(p_partial),
                "n": len(valid),
            }

    with open(os.path.join(out_dir, "arousal_diversity_results.json"), "w") as f:
        json.dump(corr_results, f, indent=2)

    # Scatter plot — color-coded by season
    season_colors = {1: "#E63946", 2: "#F4A261", 3: "#2A9D8F",
                     4: "#264653", 5: "#7209B7", 6: "#E76F51"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i, arousal in enumerate(["mean_HR", "mean_EDA_tonic"]):
        for j, diversity in enumerate(["n_unique_states", "switch_rate"]):
            ax = axes[i, j]
            for season in sorted(df["season"].unique()):
                sub = df[df["season"] == season].dropna(subset=[arousal, diversity])
                if len(sub) == 0:
                    continue
                ax.scatter(
                    sub[arousal], sub[diversity],
                    c=season_colors.get(season, "#888888"),
                    alpha=0.4, s=15, label=f"S{season}",
                )
            ax.set_xlabel(arousal)
            ax.set_ylabel(diversity)
            key = f"{arousal}_vs_{diversity}"
            if key in corr_results:
                r = corr_results[key]
                rp = r.get("rho_partial_n_trs")
                if rp is not None and np.isfinite(rp):
                    ax.set_title(f"rho={r['rho']:.2f} (partial={rp:.2f}), p={r['p']:.3f}")
                else:
                    ax.set_title(f"rho={r['rho']:.2f}, p={r['p']:.3f}")
            if i == 0 and j == 1:
                ax.legend(fontsize=7, title="Season", title_fontsize=7, markerscale=1.5)
    fig.suptitle(f"Episode-level arousal vs state diversity (each point = 1 episode, N={len(df)})")
    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"arousal_diversity_scatter.{fmt}"), dpi=200)
    plt.close(fig)

    logger.info("Analysis 5 complete: %d episodes", len(episode_data))


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    sub_id = args.sub_id
    parc = args.parcellation
    stimulus = args.stimulus
    config = STIMULUS_CONFIG[stimulus]

    out_dir = os.path.join(
        SCRATCH_DIR, "output", config["output_dir"], parc, sub_id
    )
    if args.exclude_sub_hrf:
        out_dir = os.path.join(out_dir, "sub_hrf_excluded")
    os.makedirs(out_dir, exist_ok=True)

    decoded_states, physio_features, physio_features_raw, rec_summary = load_inputs(
        sub_id, parc, args.vt, stimulus
    )
    recurrence_scores = load_recurrence_scores(rec_summary)

    # Build active states set (recurrence > 0), with sub-HRF filtering
    active_states = sorted(
        int(i) for i in range(len(recurrence_scores)) if recurrence_scores[i] > 0
    )
    if args.exclude_sub_hrf:
        rec_base = os.path.join(
            SCRATCH_DIR, "output", "05a_recurrence_analysis", parc, sub_id,
        )
        if args.vt is not None:
            rec_base = os.path.join(rec_base, f"vt{args.vt}")
        try:
            _, excluded_ids, _ = load_eligible_states(rec_base)
            excluded_set = set(int(s) for s in excluded_ids)
            active_states = [s for s in active_states if s not in excluded_set]
            logger.info(
                "Sub-HRF exclusion ON: removed %d states",
                len(excluded_set),
            )
        except FileNotFoundError:
            logger.warning(
                "eligible_states.json not found; sub-HRF filtering skipped. "
                "Re-run 05a to generate it."
            )

    logger.info(
        "Active states: %d (recurrence > 0)",
        len(active_states),
    )

    # ── State flags (optional, from 05e_a4) ──────────────────────────────
    state_flags = load_state_flags(sub_id, parc, SCRATCH_DIR, vt=args.vt)

    # ── Physio QC ────────────────────────────────────────────────────────
    pv_df = None
    if DATA_DIR is not None:
        physprep_dir = os.path.join(
            DATA_DIR, "all_about_cneuromod", f"{stimulus}.physprep",
        )
        if os.path.isdir(physprep_dir):
            pv_df = load_run_percentage_valid(
                sub_id, physprep_dir, list(decoded_states.keys()),
                stimulus=stimulus,
            )
            qc_report = compute_qc_report(pv_df, threshold=args.pv_threshold)
            with open(os.path.join(out_dir, "physio_qc_report.json"), "w") as f:
                json.dump(qc_report, f, indent=2)
            logger.info("Physio QC report saved (%d runs)", qc_report["n_runs_total"])

            # EDA MNAR diagnostic (must use raw physio — z-scored HR is ~0 per run)
            mnar = run_eda_mnar_diagnostic(physio_features_raw, pv_df)
            with open(os.path.join(out_dir, "eda_mnar_diagnostic.json"), "w") as f:
                json.dump(mnar, f, indent=2)
        else:
            logger.warning("Physprep dir not found: %s — QC skipped", physprep_dir)
    else:
        logger.warning("DATA_DIR not set — physio QC skipped")

    # ── Primary analyses ─────────────────────────────────────────────────
    analysis_1_state_profiles(
        decoded_states, physio_features, recurrence_scores, out_dir,
        state_flags=state_flags,
    )
    analysis_2_multilag(
        decoded_states, physio_features, recurrence_scores, active_states,
        out_dir, state_flags=state_flags,
    )
    analysis_3_tta(
        decoded_states, physio_features, recurrence_scores, active_states,
        out_dir, state_flags=state_flags,
    )
    analysis_4_consistency(
        decoded_states, physio_features, recurrence_scores, active_states,
        out_dir, state_flags=state_flags,
    )
    analysis_5_arousal_diversity(decoded_states, physio_features_raw, out_dir)

    # ── Sensitivity analyses (optional) ──────────────────────────────────
    if args.sensitivity_analysis:
        sens_dir = os.path.join(out_dir, "sensitivity")

        # (a) EDA-excluded: cardiac + respiratory only (cols 0-3)
        cardiac_resp_cols = FEATURE_COLUMNS[:4]  # HR, HRV, breathing, RVT
        eda_excl_dir = os.path.join(sens_dir, "eda_excluded")
        os.makedirs(eda_excl_dir, exist_ok=True)
        logger.info("Sensitivity: EDA-excluded analysis (cols 0-3)")
        analysis_1_state_profiles(
            decoded_states, physio_features, recurrence_scores, eda_excl_dir,
            state_flags=state_flags, feature_cols=cardiac_resp_cols,
        )
        analysis_4_consistency(
            decoded_states, physio_features, recurrence_scores, active_states,
            eda_excl_dir, state_flags=state_flags,
        )

        # (b) High-confidence: mask per-channel below PV threshold
        if pv_df is not None:
            conf = classify_channel_confidence(pv_df, threshold=args.pv_threshold)
            # Build filtered physio: NaN out channels below threshold per run
            channel_feat_map = {
                "ECG": [0, 1], "RSP": [2, 3], "EDA": [4, 5, 6], "PPG": [],
            }
            physio_hc = {}
            for run_id, feat_arr in physio_features.items():
                run_idx = pv_df.index[pv_df["run_id"] == run_id]
                if len(run_idx) == 0:
                    physio_hc[run_id] = feat_arr
                    continue
                arr = feat_arr.copy()
                for ch, cols in channel_feat_map.items():
                    if not conf.loc[run_idx[0], f"{ch}_ok"]:
                        for c in cols:
                            arr[:, c] = np.nan
                physio_hc[run_id] = arr

            n_hc_runs = sum(
                conf.loc[i].all()
                for i in conf.index
                if pv_df.loc[i, "run_id"] in physio_features
            )
            if n_hc_runs >= 10:
                hc_dir = os.path.join(sens_dir, "high_confidence")
                os.makedirs(hc_dir, exist_ok=True)
                logger.info(
                    "Sensitivity: high-confidence (PV>=%.2f), %d/%d runs fully valid",
                    args.pv_threshold, n_hc_runs, len(physio_features),
                )
                analysis_1_state_profiles(
                    decoded_states, physio_hc, recurrence_scores, hc_dir,
                    state_flags=state_flags,
                )
                analysis_4_consistency(
                    decoded_states, physio_hc, recurrence_scores, active_states,
                    hc_dir, state_flags=state_flags,
                )
            else:
                logger.warning(
                    "Too few high-confidence runs (%d) — skipping sensitivity",
                    n_hc_runs,
                )

    logger.info("All analyses complete. Output: %s", out_dir)


if __name__ == "__main__":
    main()
