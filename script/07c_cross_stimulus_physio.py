#!/usr/bin/env python3
"""
07c_cross_stimulus_physio.py - Cross-stimulus physio correspondence.

Tests whether brain states maintain consistent autonomic signatures
across Friends, Movie10, and (optionally) Harry Potter (unimodal reading),
using continuous recurrence scores.

Four analyses:
    C1: Cross-stimulus physio signature stability (PRIMARY)
        Friends vs Movie10 + Friends vs HP (if available)
    C2: Genre-specific physio profiles (all active states)
        Includes HP as reading-only genre with modality separator
    C3: Arousal modulation of state dynamics
    C4: Cross-stimulus transition-triggered averages (all active states)

Prerequisites:
    - 07a completed for Friends and Movie10 (+ hp_07a for HP)
    - 04 decoded_states.pkl (Friends)
    - m10_04 decoded_states.pkl (Movie10)
    - hp_04 decoded_states.pkl (HP, optional - absent for some subjects)
    - 05a recurrence_summary.json

Outputs:
    {SCRATCH_DIR}/output/07c_cross_stimulus_physio/{parcellation}/{sub_id}/
"""

import os
import sys
import json
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from utils.physio_io import FEATURE_COLUMNS
from utils.stats import benjamini_hochberg, partial_spearman
from utils.state_blocks import load_eligible_states
from utils.state_flags_io import load_state_flags, annotate_dataframe
from utils.physio_qc import (
    load_run_percentage_valid,
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

# Movie type lookup (from run_id)
MOVIE_PREFIXES = ["bourne", "wolf", "figures", "life"]


def get_movie_type(run_id):
    """Extract movie type from run_id."""
    for prefix in MOVIE_PREFIXES:
        if f"task-{prefix}" in run_id or run_id.startswith(prefix):
            return prefix
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-stimulus physio correspondence analysis.",
    )
    parser.add_argument("--sub_id", type=str, required=True)
    parser.add_argument("--parcellation", type=str, default="atlas-4S156Parcels")
    parser.add_argument("--vt", type=str, default=None)
    parser.add_argument("--exclude_sub_hrf", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Exclude sub-HRF states from analyses "
                             "(default: False - all states included, annotated by category).")
    parser.add_argument("--sensitivity_analysis", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Run sensitivity analyses: EDA-excluded + high-confidence "
                             "(default: False).")
    parser.add_argument("--pv_threshold", type=float, default=0.80,
                        help="PercentageValid threshold for per-channel confidence "
                             "(default: 0.80, physprep's validated boundary).")
    return parser.parse_args()


def load_recurrence_scores(rec_summary):
    """Extract recurrence_scores array from recurrence summary."""
    return np.array(rec_summary["recurrence_scores"], dtype=float)


def load_physio_features_from_dir(physio_dir, decoded_states, suffix="_physio_features"):
    """Load physio features matching decoded_states keys.

    Args:
        physio_dir: Directory containing .npy files.
        decoded_states: dict with run_id keys.
        suffix: File suffix before .npy (default: '_physio_features').
            Use '_physio_features_raw' for pre-normalization features.
    """
    features = {}
    for run_id in decoded_states:
        # Try exact name
        path = os.path.join(physio_dir, f"{run_id}{suffix}.npy")
        if os.path.exists(path):
            features[run_id] = np.load(path)
            continue
        # Try safe filename (movie10 keys get shortened)
        safe = _make_safe_filename(run_id)
        path2 = os.path.join(physio_dir, f"{safe}{suffix}.npy")
        if os.path.exists(path2):
            features[run_id] = np.load(path2)
            continue
        if suffix == "_physio_features":
            logger.warning("No physio features for %s", run_id)
    return features


def _make_safe_filename(run_key):
    """Same logic as 07a."""
    import re

    m = re.search(r"task-(\w+?)(?:_run-(\d+))?(?:_space|$)", run_key)
    if m:
        task = m.group(1)
        run = m.group(2)
        return f"{task}_run-{run.zfill(2)}" if run else task
    return run_key.replace("/", "_").replace(" ", "_")


def load_all_inputs(sub_id, parc, vt):
    """Load all data needed for cross-stimulus analysis."""
    vt_subdir = f"vt{vt}" if vt else ""

    # Friends decoded states
    friends_ds_path = os.path.join(
        SCRATCH_DIR, "output", "04_combined_hdphmm", parc, sub_id,
        "final", vt_subdir, "decoded_states.pkl",
    ).replace("//", "/")
    with open(friends_ds_path, "rb") as f:
        friends_ds = pickle.load(f)

    # Movie10 decoded states
    m10_ds_path = os.path.join(
        SCRATCH_DIR, "output", "m10_04_decoded", parc, sub_id,
        vt_subdir, "decoded_states.pkl",
    ).replace("//", "/")
    with open(m10_ds_path, "rb") as f:
        m10_ds = pickle.load(f)

    # Recurrence summary
    rec_path = os.path.join(
        SCRATCH_DIR, "output", "05a_recurrence_analysis", parc, sub_id,
        vt_subdir, "recurrence_summary.json",
    ).replace("//", "/")
    with open(rec_path) as f:
        rec_summary = json.load(f)

    # Physio features
    friends_physio_dir = os.path.join(
        SCRATCH_DIR, "output", "07a_physio_features", sub_id
    )
    m10_physio_dir = os.path.join(
        SCRATCH_DIR, "output", "m10_07a_physio_features", sub_id
    )

    friends_physio = load_physio_features_from_dir(friends_physio_dir, friends_ds)
    m10_physio = load_physio_features_from_dir(m10_physio_dir, m10_ds)

    # Raw (pre-z-score) features for arousal proxies (C3)
    friends_physio_raw = load_physio_features_from_dir(
        friends_physio_dir, friends_ds, suffix="_physio_features_raw"
    )
    m10_physio_raw = load_physio_features_from_dir(
        m10_physio_dir, m10_ds, suffix="_physio_features_raw"
    )

    # Harry Potter decoded states + physio (optional - absent for some subjects)
    hp_ds = None
    hp_physio = None
    hp_physio_raw = None
    try:
        hp_ds_path = os.path.join(
            SCRATCH_DIR, "output", "hp_04_decoded", parc, sub_id,
            vt_subdir, "decoded_states.pkl",
        ).replace("//", "/")
        with open(hp_ds_path, "rb") as f:
            hp_ds = pickle.load(f)

        hp_physio_dir = os.path.join(
            SCRATCH_DIR, "output", "hp_07a_physio_features", sub_id
        )
        hp_physio = load_physio_features_from_dir(hp_physio_dir, hp_ds)
        hp_physio_raw = load_physio_features_from_dir(
            hp_physio_dir, hp_ds, suffix="_physio_features_raw"
        )
        logger.info(
            "HP loaded: %d/%d runs with physio",
            len(hp_physio), len(hp_ds),
        )
    except FileNotFoundError:
        logger.info("HP data not available for %s - HP analyses will be skipped", sub_id)
        hp_ds = None
        hp_physio = None
        hp_physio_raw = None

    logger.info(
        "Loaded: Friends %d/%d runs with physio, Movie10 %d/%d runs with physio",
        len(friends_physio), len(friends_ds),
        len(m10_physio), len(m10_ds),
    )

    return (
        friends_ds, m10_ds,
        friends_physio, m10_physio,
        friends_physio_raw, m10_physio_raw,
        rec_summary,
        hp_ds, hp_physio, hp_physio_raw,
    )


# ── C1: Cross-stimulus physio signature stability ─────────────────────────


def _compute_state_profiles(decoded_states, physio_features):
    """Epoch-level mean physio per state."""
    profiles = {}  # state → list of epoch mean vectors
    for run_id, state_seq in decoded_states.items():
        if run_id not in physio_features:
            continue
        feats = physio_features[run_id]
        n_trs = min(len(state_seq), len(feats))
        state_seq = np.asarray(state_seq[:n_trs])

        # Find contiguous blocks
        changes = np.flatnonzero(state_seq[1:] != state_seq[:-1]) + 1
        starts = np.concatenate(([0], changes))
        ends = np.concatenate((changes, [n_trs]))

        for s, e in zip(starts, ends):
            state = int(state_seq[s])
            if e <= s:
                continue
            epoch_mean = np.nanmean(feats[s:e], axis=0)
            profiles.setdefault(state, []).append(epoch_mean)

    # Average across epochs → one vector per state
    state_means = {}
    for state, epochs in profiles.items():
        if len(epochs) >= 3:  # minimum epochs
            state_means[state] = np.nanmean(epochs, axis=0)
    return state_means


def _c1_pairwise_correlation(friends_profiles, other_profiles, active_states):
    """Compute per-feature and overall profile correlations for one stimulus pair."""
    active_set = set(active_states)
    shared_states = sorted(
        set(friends_profiles.keys()) & set(other_profiles.keys()) & active_set
    )
    if len(shared_states) < 3:
        return {"n_shared_states": len(shared_states), "status": "insufficient_states"}, shared_states

    feature_correlations = {}
    for i, feat in enumerate(FEATURE_COLUMNS[:6]):
        f_vals = np.array([friends_profiles[s][i] for s in shared_states])
        o_vals = np.array([other_profiles[s][i] for s in shared_states])
        valid = np.isfinite(f_vals) & np.isfinite(o_vals)
        if np.sum(valid) < 3:
            continue
        r, p = stats.pearsonr(f_vals[valid], o_vals[valid])
        feature_correlations[feat] = {"r": float(r), "p": float(p), "n": int(np.sum(valid))}

    # FDR correction across per-feature p-values
    feat_names = list(feature_correlations.keys())
    if feat_names:
        p_vals = np.array([feature_correlations[f]["p"] for f in feat_names])
        q_vals = benjamini_hochberg(p_vals)
        for f_name, q in zip(feat_names, q_vals):
            feature_correlations[f_name]["p_fdr"] = float(q)

    # Fisher-z averaged per-feature correlations (replaces flattened Pearson r)
    valid_rs = [feature_correlations[f]["r"] for f in feat_names
                if abs(feature_correlations[f]["r"]) < 1.0]  # exclude r=+/-1 (arctanh undefined)
    if valid_rs:
        z_vals = [np.arctanh(r) for r in valid_rs]
        r_fisher_z = float(np.tanh(np.mean(z_vals)))
    else:
        r_fisher_z = np.nan

    # Keep flattened Pearson r for backward compatibility
    f_all = np.array([friends_profiles[s] for s in shared_states])
    o_all = np.array([other_profiles[s] for s in shared_states])
    valid_mask = np.isfinite(f_all) & np.isfinite(o_all)
    if np.sum(valid_mask) > 3:
        overall_r_flattened, overall_p_flattened = stats.pearsonr(
            f_all[valid_mask].ravel(), o_all[valid_mask].ravel()
        )
    else:
        overall_r_flattened, overall_p_flattened = np.nan, np.nan

    return {
        "n_shared_states": len(shared_states),
        "overall_profile_correlation": {"r": float(r_fisher_z), "method": "fisher_z_average"},
        "overall_profile_correlation_flattened": {
            "r": float(overall_r_flattened), "p": float(overall_p_flattened),
            "method": "flattened_pearson", "note": "deprecated - inflates df",
        },
        "per_feature_correlations": feature_correlations,
        "fdr_corrected": True,
        "shared_states": shared_states,
    }, shared_states


def _c1_scatter_figure(
    friends_profiles, other_profiles, shared_states,
    recurrence_scores, feature_correlations, label_other, n_shared,
):
    """Create a 2×3 scatter figure for one stimulus pair."""
    from utils.plot_style import recurrence_color, make_recurrence_colorbar

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    for i, feat in enumerate(FEATURE_COLUMNS[:6]):
        ax = axes[i]
        f_vals = np.array([friends_profiles[s][i] for s in shared_states])
        o_vals = np.array([other_profiles[s][i] for s in shared_states])
        colors = [recurrence_color(recurrence_scores[s]) for s in shared_states]
        ax.scatter(f_vals, o_vals, c=colors, alpha=0.7, edgecolors="white", linewidths=0.3)
        ax.set_xlabel(f"Friends {feat}")
        ax.set_ylabel(f"{label_other} {feat}")
        if feat in feature_correlations:
            fc = feature_correlations[feat]
            ax.set_title(f"{feat}: r={fc['r']:.2f}, p={fc['p']:.3f}")
        lim = [
            min(np.nanmin(f_vals), np.nanmin(o_vals)),
            max(np.nanmax(f_vals), np.nanmax(o_vals)),
        ]
        ax.plot(lim, lim, "k--", alpha=0.3)
    make_recurrence_colorbar(axes[-1])
    fig.suptitle(
        f"C1: Friends vs {label_other} physio signature stability (n={n_shared} states)"
    )
    fig.tight_layout()
    return fig


def analysis_c1_signature_stability(
    friends_ds, m10_ds, friends_physio, m10_physio,
    recurrence_scores, active_states, out_dir, state_flags=None,
    hp_ds=None, hp_physio=None,
):
    """Compare per-state physio profiles across stimuli.

    Primary test: profile correlation across all shared active states.
    Produces Friends-vs-Movie10 figure; if HP available, also Friends-vs-HP.
    """
    logger.info("C1: Cross-stimulus physio signature stability")

    friends_profiles = _compute_state_profiles(friends_ds, friends_physio)
    m10_profiles = _compute_state_profiles(m10_ds, m10_physio)

    # Friends vs Movie10
    fm_results, fm_shared = _c1_pairwise_correlation(
        friends_profiles, m10_profiles, active_states,
    )
    results = {"friends_vs_movie10": fm_results}

    if fm_results.get("status") != "insufficient_states":
        fig = _c1_scatter_figure(
            friends_profiles, m10_profiles, fm_shared,
            recurrence_scores, fm_results["per_feature_correlations"],
            "Movie10", fm_results["n_shared_states"],
        )
        for fmt in ("pdf", "png"):
            fig.savefig(os.path.join(out_dir, f"profile_correlation_by_recurrence.{fmt}"), dpi=200)
        plt.close(fig)

    # Friends vs HP (if available)
    if hp_ds is not None and hp_physio is not None:
        hp_profiles = _compute_state_profiles(hp_ds, hp_physio)
        fh_results, fh_shared = _c1_pairwise_correlation(
            friends_profiles, hp_profiles, active_states,
        )
        results["friends_vs_hp"] = fh_results

        if fh_results.get("status") != "insufficient_states":
            fig = _c1_scatter_figure(
                friends_profiles, hp_profiles, fh_shared,
                recurrence_scores, fh_results["per_feature_correlations"],
                "HP (reading)", fh_results["n_shared_states"],
            )
            for fmt in ("pdf", "png"):
                fig.savefig(os.path.join(out_dir, f"profile_correlation_by_recurrence_hp.{fmt}"), dpi=200)
            plt.close(fig)

    with open(os.path.join(out_dir, "cross_stimulus_signature_stability.json"), "w") as f:
        json.dump(results, f, indent=2)

    fm_r = fm_results.get("overall_profile_correlation", {}).get("r", float("nan"))
    logger.info(
        "C1 complete: Friends-M10 %d shared states (r=%.3f)%s",
        fm_results.get("n_shared_states", 0),
        fm_r if fm_r == fm_r else float("nan"),
        f", Friends-HP {results.get('friends_vs_hp', {}).get('n_shared_states', 0)} shared states"
        if "friends_vs_hp" in results else "",
    )


# ── C2: Genre-specific physio profiles ────────────────────────────────────


def analysis_c2_genre_profiles(
    friends_ds, m10_ds, friends_physio, m10_physio,
    recurrence_scores, active_states, out_dir, state_flags=None,
    hp_ds=None, hp_physio=None,
):
    """Per-genre mean physio profiles for all active states."""
    logger.info("C2: Genre-specific physio profiles")

    active_set = set(active_states)

    # Compute per-genre, per-state mean physio for movie10
    genre_state_means = {}  # (genre, state) → [mean vectors]
    for run_id, state_seq in m10_ds.items():
        genre = get_movie_type(run_id)
        if genre is None or run_id not in m10_physio:
            continue
        feats = m10_physio[run_id]
        n_trs = min(len(state_seq), len(feats))
        state_seq = np.asarray(state_seq[:n_trs])
        feats = feats[:n_trs]

        for state in active_set:
            mask = state_seq == state
            if np.sum(mask) < 2:
                continue
            mean_vec = np.nanmean(feats[mask], axis=0)
            key = (genre, state)
            genre_state_means.setdefault(key, []).append(mean_vec)

    # Aggregate Movie10 rows
    rows = []
    for (genre, state), epochs in genre_state_means.items():
        grand_mean = np.nanmean(epochs, axis=0)
        rows.append({
            "genre": genre,
            "state": state,
            "recurrence_score": float(recurrence_scores[state]),
            "stimulus": "movie10",
            "n_runs": len(epochs),
            **{FEATURE_COLUMNS[i]: float(grand_mean[i]) for i in range(7)},
        })

    # Add Friends profiles
    for run_id, state_seq in friends_ds.items():
        if run_id not in friends_physio:
            continue
        feats = friends_physio[run_id]
        n_trs = min(len(state_seq), len(feats))
        state_seq = np.asarray(state_seq[:n_trs])
        feats = feats[:n_trs]
        for state in active_set:
            mask = state_seq == state
            if np.sum(mask) < 2:
                continue
            mean_vec = np.nanmean(feats[mask], axis=0)
            key = ("friends", state)
            genre_state_means.setdefault(key, []).append(mean_vec)

    for (genre, state), epochs in genre_state_means.items():
        if genre != "friends":
            continue
        grand_mean = np.nanmean(epochs, axis=0)
        rows.append({
            "genre": "friends",
            "state": state,
            "recurrence_score": float(recurrence_scores[state]),
            "stimulus": "friends",
            "n_runs": len(epochs),
            **{FEATURE_COLUMNS[i]: float(grand_mean[i]) for i in range(7)},
        })

    # Add HP profiles (reading-only)
    if hp_ds is not None and hp_physio is not None:
        for run_id, state_seq in hp_ds.items():
            if run_id not in hp_physio:
                continue
            feats = hp_physio[run_id]
            n_trs = min(len(state_seq), len(feats))
            state_seq = np.asarray(state_seq[:n_trs])
            feats = feats[:n_trs]
            for state in active_set:
                mask = state_seq == state
                if np.sum(mask) < 2:
                    continue
                mean_vec = np.nanmean(feats[mask], axis=0)
                key = ("harrypotter", state)
                genre_state_means.setdefault(key, []).append(mean_vec)

        for (genre, state), epochs in genre_state_means.items():
            if genre != "harrypotter":
                continue
            grand_mean = np.nanmean(epochs, axis=0)
            rows.append({
                "genre": "harrypotter",
                "state": state,
                "recurrence_score": float(recurrence_scores[state]),
                "stimulus": "harrypotter",
                "n_runs": len(epochs),
                **{FEATURE_COLUMNS[i]: float(grand_mean[i]) for i in range(7)},
            })

    if not rows:
        logger.warning("No genre profile data")
        return

    df = pd.DataFrame(rows)
    df = annotate_dataframe(df, state_flags)
    df.to_csv(os.path.join(out_dir, "genre_physio_profiles.csv"), index=False)

    # Heatmap: states × genres for HR
    pivot_states = sorted(df["state"].unique())
    genres = ["friends", "bourne", "wolf", "figures", "life", "harrypotter"]
    available_genres = [g for g in genres if g in df["genre"].values]
    # Display labels: mark HP as reading-only
    genre_labels = {g: g for g in genres}
    genre_labels["harrypotter"] = "HP (reading)"

    if len(pivot_states) > 0 and len(available_genres) > 1:
        hr_matrix = np.full((len(pivot_states), len(available_genres)), np.nan)
        for i, state in enumerate(pivot_states):
            for j, genre in enumerate(available_genres):
                sub = df[(df["state"] == state) & (df["genre"] == genre)]
                if len(sub) > 0:
                    hr_matrix[i, j] = sub["HR_bpm"].values[0]

        fig, ax = plt.subplots(figsize=(max(8, len(available_genres) * 1.3), max(4, len(pivot_states) * 0.3)))
        im = ax.imshow(hr_matrix, aspect="auto", cmap="RdBu_r")
        ax.set_xticks(range(len(available_genres)))
        ax.set_xticklabels([genre_labels.get(g, g) for g in available_genres], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot_states)))
        ax.set_yticklabels([f"S{s}" for s in pivot_states])
        ax.set_ylabel("State")
        ax.set_xlabel("Genre / Stimulus")
        ax.set_title("HR (z-scored) by State × Genre")
        plt.colorbar(im, ax=ax, label="HR (z-scored)")
        # Vertical separator before HP (reading-only) if present
        if "harrypotter" in available_genres:
            hp_idx = available_genres.index("harrypotter")
            ax.axvline(hp_idx - 0.5, color="black", linewidth=1.5, linestyle="--")
        fig.tight_layout()
        for fmt in ("pdf", "png"):
            fig.savefig(os.path.join(out_dir, f"genre_heatmap.{fmt}"), dpi=200)
        plt.close(fig)

    # ── Per-state cross-genre SD and similarity to Friends ──────────
    av_genres = ["bourne", "wolf", "figures", "life"]  # audiovisual (Movie10) only
    comparison_genres = list(av_genres)
    if "harrypotter" in df["genre"].values:
        comparison_genres.append("harrypotter")
    all_genres = ["friends"] + comparison_genres
    sd_records = []
    similarity_records = []

    for state in pivot_states:
        # Collect per-genre profile vectors for this state
        genre_profiles = {}
        for genre in all_genres:
            sub = df[(df["state"] == state) & (df["genre"] == genre)]
            if len(sub) > 0:
                vec = np.array([sub[FEATURE_COLUMNS[i]].values[0] for i in range(7)])
                genre_profiles[genre] = vec

        if len(genre_profiles) < 2:
            continue

        # SD across audiovisual (Friends + Movie10) genres only
        for i, feat in enumerate(FEATURE_COLUMNS[:6]):
            av_vals = [genre_profiles[g][i] for g in (["friends"] + av_genres)
                       if g in genre_profiles and np.isfinite(genre_profiles[g][i])]
            if len(av_vals) >= 2:
                sd_records.append({
                    "state": int(state),
                    "feature": feat,
                    "sd_cross_genre": float(np.std(av_vals)),
                    "n_genres": len(av_vals),
                    "includes_reading_only": False,
                })

            # SD across all genres (including HP)
            all_vals = [genre_profiles[g][i] for g in all_genres
                        if g in genre_profiles and np.isfinite(genre_profiles[g][i])]
            if len(all_vals) > len(av_vals) and len(all_vals) >= 2:
                sd_records.append({
                    "state": int(state),
                    "feature": feat,
                    "sd_cross_genre": float(np.std(all_vals)),
                    "n_genres": len(all_vals),
                    "includes_reading_only": True,
                })

        # Similarity to Friends (Pearson r of 7-feature vector)
        if "friends" in genre_profiles:
            f_vec = genre_profiles["friends"]
            for genre in comparison_genres:
                if genre not in genre_profiles:
                    continue
                m_vec = genre_profiles[genre]
                valid = np.isfinite(f_vec) & np.isfinite(m_vec)
                if np.sum(valid) >= 3:
                    r, p = stats.pearsonr(f_vec[valid], m_vec[valid])
                    n_feat = int(np.sum(valid))
                    rec = {
                        "state": int(state),
                        "genre": genre,
                        "modality": "reading_only" if genre == "harrypotter" else "audiovisual",
                        "r_vs_friends": float(r),
                        "p": float(p),
                        "n_features": n_feat,
                    }
                    if n_feat < 10:
                        rec["low_power_warning"] = True
                        logger.warning(
                            "C2 state %d × %s: correlation on %d features (< 10 recommended)",
                            state, genre, n_feat,
                        )
                    similarity_records.append(rec)

    if sd_records:
        pd.DataFrame(sd_records).to_csv(
            os.path.join(out_dir, "genre_cross_genre_sd.csv"), index=False
        )
    if similarity_records:
        with open(os.path.join(out_dir, "genre_similarity_to_friends.json"), "w") as f:
            json.dump(similarity_records, f, indent=2)

    logger.info(
        "C2 complete: %d genre-state profiles, %d SD records, %d similarity records",
        len(rows), len(sd_records), len(similarity_records),
    )


# ── C3: Arousal modulation of state dynamics ──────────────────────────────


def analysis_c3_arousal_modulation(
    friends_ds, m10_ds, friends_physio_raw, m10_physio_raw, out_dir,
    hp_ds=None, hp_physio_raw=None,
):
    """Per-run arousal vs state diversity, across all stimuli.

    Uses RAW (pre-z-score) features for arousal proxies (HR, EDA tonic).
    Z-scored features have ~0 mean per run by construction.
    """
    logger.info("C3: Arousal modulation of state dynamics")

    stimulus_list = [
        ("friends", friends_ds, friends_physio_raw),
        ("movie10", m10_ds, m10_physio_raw),
    ]
    if hp_ds is not None and hp_physio_raw is not None:
        stimulus_list.append(("harrypotter", hp_ds, hp_physio_raw))

    rows = []
    for stimulus, ds, physio in stimulus_list:
        for run_id, state_seq in ds.items():
            if run_id not in physio:
                continue
            feats = physio[run_id]
            n_trs = min(len(state_seq), len(feats))
            state_seq = np.asarray(state_seq[:n_trs])

            n_unique = len(np.unique(state_seq))
            transitions = np.sum(state_seq[1:] != state_seq[:-1])
            switch_rate = float(transitions / max(n_trs - 1, 1))
            mean_hr = float(np.nanmean(feats[:n_trs, 0]))
            mean_eda = float(np.nanmean(feats[:n_trs, 4]))

            if stimulus == "movie10":
                genre = get_movie_type(run_id)
            elif stimulus == "harrypotter":
                genre = "harrypotter"
            else:
                genre = "friends"

            rows.append({
                "stimulus": stimulus,
                "genre": genre,
                "run_id": run_id,
                "n_trs": n_trs,
                "n_unique_states": n_unique,
                "switch_rate": switch_rate,
                "mean_HR": mean_hr,
                "mean_EDA_tonic": mean_eda,
            })

    if not rows:
        logger.warning("No data for C3")
        return

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "arousal_state_dynamics.csv"), index=False)

    # Spearman per stimulus (uncorrected + partial controlling for run length)
    corr_results = {}
    stim_names = ["friends", "movie10"]
    if "harrypotter" in df["stimulus"].values:
        stim_names.append("harrypotter")
    for stim in stim_names:
        sub = df[df["stimulus"] == stim]
        for arousal in ["mean_HR", "mean_EDA_tonic"]:
            for diversity in ["n_unique_states", "switch_rate"]:
                valid = sub[[arousal, diversity, "n_trs"]].dropna()
                if len(valid) < 5:
                    continue
                rho, p = stats.spearmanr(valid[arousal], valid[diversity])
                rho_partial, p_partial = partial_spearman(
                    valid[arousal].values,
                    valid[diversity].values,
                    valid["n_trs"].values,
                )
                corr_results[f"{stim}_{arousal}_vs_{diversity}"] = {
                    "rho": float(rho), "p": float(p),
                    "rho_partial_n_trs": float(rho_partial),
                    "p_partial_n_trs": float(p_partial),
                    "n": len(valid),
                }

    with open(os.path.join(out_dir, "arousal_modulation_results.json"), "w") as f:
        json.dump(corr_results, f, indent=2)

    # Scatter: HR vs state diversity, colored by stimulus
    stim_colors = [("friends", "tab:blue"), ("movie10", "tab:red")]
    if "harrypotter" in df["stimulus"].values:
        stim_colors.append(("harrypotter", "tab:green"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, div_feat in zip(axes, ["n_unique_states", "switch_rate"]):
        for stim, color in stim_colors:
            sub = df[df["stimulus"] == stim]
            label = "HP (reading)" if stim == "harrypotter" else stim
            ax.scatter(sub["mean_HR"], sub[div_feat], c=color, alpha=0.4, s=15, label=label)
        ax.set_xlabel("Mean HR (raw)")
        ax.set_ylabel(div_feat)
        ax.legend()
    fig.suptitle("C3: Arousal modulation of state dynamics")
    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"arousal_modulation_scatter.{fmt}"), dpi=200)
    plt.close(fig)

    logger.info("C3 complete: %d run records", len(rows))


# ── C4: Cross-stimulus TTAs ──────────────────────────────────────────────


def analysis_c4_cross_stimulus_tta(
    friends_ds, m10_ds, friends_physio, m10_physio,
    recurrence_scores, active_states, out_dir, state_flags=None,
    hp_ds=None, hp_physio=None,
):
    """Compare TTA shapes across stimuli for all active states.

    Extracts transitions BOTH into and out of active states.
    Includes bootstrap CIs at the transition level.
    """
    logger.info("C4: Cross-stimulus transition-triggered averages")

    window = 10
    active_set = set(active_states)
    min_transitions = 20
    n_bootstrap = 1000

    def _extract_ttas(decoded_states, physio_features):
        """Extract TTA snippets for transitions into and out of active states."""
        ttas_into = {s: [] for s in active_states}
        ttas_outof = {s: [] for s in active_states}
        for run_id, state_seq in decoded_states.items():
            if run_id not in physio_features:
                continue
            feats = physio_features[run_id]
            n_trs = min(len(state_seq), len(feats))
            state_seq = np.asarray(state_seq[:n_trs])

            transitions = np.flatnonzero(state_seq[1:] != state_seq[:-1]) + 1
            for t in transitions:
                if t - window < 0 or t + window >= n_trs:
                    continue
                state_after = int(state_seq[t])
                state_before = int(state_seq[t - 1])
                snippet = feats[t - window : t + window]
                if state_after in active_set:
                    ttas_into[state_after].append(snippet)
                if state_before in active_set:
                    ttas_outof[state_before].append(snippet)
        return ttas_into, ttas_outof

    def _bootstrap_ci(snippets, n_boot=n_bootstrap):
        """Compute bootstrap 95% CI for mean TTA (resample transitions)."""
        arr = np.array(snippets)  # (n_transitions, 2*window, 7)
        n = len(arr)
        rng = np.random.default_rng(42)
        boot_means = np.empty((n_boot, 2 * window, 7))
        for b in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            boot_means[b] = np.nanmean(arr[idx], axis=0)
        ci_lo = np.nanpercentile(boot_means, 2.5, axis=0)
        ci_hi = np.nanpercentile(boot_means, 97.5, axis=0)
        return ci_lo, ci_hi

    # Extract TTAs for each stimulus
    friends_into, friends_outof = _extract_ttas(friends_ds, friends_physio)
    m10_into, m10_outof = _extract_ttas(m10_ds, m10_physio)

    hp_into, hp_outof = None, None
    if hp_ds is not None and hp_physio is not None:
        hp_into, hp_outof = _extract_ttas(hp_ds, hp_physio)

    # Build list of stimuli TTAs for iteration
    stim_ttas = [
        ("friends", friends_into, friends_outof),
        ("movie10", m10_into, m10_outof),
    ]
    if hp_into is not None:
        stim_ttas.append(("harrypotter", hp_into, hp_outof))

    rows = []
    comparison_results = {}

    for direction in ["into", "outof"]:
        for state in sorted(active_states):
            # Collect snippets per stimulus for this state+direction
            stim_snippets = {}
            for stim_name, into_d, outof_d in stim_ttas:
                ttas_d = into_d if direction == "into" else outof_d
                snippets = ttas_d.get(state, [])
                stim_snippets[stim_name] = snippets

                if len(snippets) < 5:
                    continue
                arr = np.array(snippets)
                mean_tta = np.nanmean(arr, axis=0)

                ci_lo, ci_hi = None, None
                if len(snippets) >= 20:
                    ci_lo, ci_hi = _bootstrap_ci(snippets)

                for t_idx in range(2 * window):
                    row = {
                        "state": state,
                        "direction": direction,
                        "stimulus": stim_name,
                        "relative_tr": t_idx - window,
                        "n_transitions": len(snippets),
                    }
                    for i, feat in enumerate(FEATURE_COLUMNS):
                        row[feat] = float(mean_tta[t_idx, i])
                        if ci_lo is not None:
                            row[f"{feat}_ci_lo"] = float(ci_lo[t_idx, i])
                            row[f"{feat}_ci_hi"] = float(ci_hi[t_idx, i])
                    rows.append(row)

            # Compare TTA shapes (HR, into direction only)
            if direction == "into":
                f_snippets = stim_snippets.get("friends", [])
                if len(f_snippets) < min_transitions:
                    continue
                f_mean = np.nanmean(np.array(f_snippets), axis=0)[:, 0]

                state_result = {}
                for other_stim in ["movie10", "harrypotter"]:
                    o_snippets = stim_snippets.get(other_stim, [])
                    if len(o_snippets) < min_transitions:
                        continue
                    o_mean = np.nanmean(np.array(o_snippets), axis=0)[:, 0]
                    valid = np.isfinite(f_mean) & np.isfinite(o_mean)
                    if np.sum(valid) > 3:
                        r, p = stats.pearsonr(f_mean[valid], o_mean[valid])
                        key = f"friends_vs_{other_stim}"
                        state_result[key] = {
                            "hr_tta_correlation": float(r),
                            "hr_tta_p": float(p),
                            "friends_n": len(f_snippets),
                            f"{other_stim}_n": len(o_snippets),
                        }
                if state_result:
                    comparison_results[int(state)] = state_result

    if rows:
        df = pd.DataFrame(rows)
        df = annotate_dataframe(df, state_flags)
        df.to_csv(os.path.join(out_dir, "cross_stimulus_tta.csv"), index=False)

    # Per-stimulus eligible state counts
    eligible_counts = {}
    for stim_name, into_d, _ in stim_ttas:
        eligible_counts[stim_name] = sum(
            1 for s in active_states if len(into_d.get(s, [])) >= min_transitions
        )

    with open(os.path.join(out_dir, "tta_comparison_results.json"), "w") as f:
        json.dump({
            "per_state": comparison_results,
            "eligible_states_per_stimulus": eligible_counts,
            "min_transitions": min_transitions,
        }, f, indent=2, default=str)

    # Plot: overlay TTAs for top-recurrence states (into direction)
    # Include states that have at least Friends + one other stimulus compared
    states_to_plot = sorted(
        [s for s in active_states if int(s) in comparison_results],
        key=lambda s: recurrence_scores[s], reverse=True,
    )[:6]
    if states_to_plot and rows:
        df = pd.DataFrame(rows)
        stim_plot_config = [
            ("friends", "tab:blue", "-", "Friends"),
            ("movie10", "tab:red", "-", "Movie10"),
        ]
        if hp_into is not None:
            stim_plot_config.append(("harrypotter", "tab:green", "-", "HP (reading)"))

        n_plot = len(states_to_plot)
        fig, axes = plt.subplots(n_plot, 1, figsize=(10, 3 * n_plot))
        if n_plot == 1:
            axes = [axes]
        for ax, state in zip(axes, states_to_plot):
            for stim, color, ls, label_base in stim_plot_config:
                sub = df[
                    (df["state"] == state)
                    & (df["stimulus"] == stim)
                    & (df["direction"] == "into")
                ]
                if len(sub) == 0:
                    continue
                ax.plot(sub["relative_tr"], sub["HR_bpm"], color=color, ls=ls,
                        label=f"{label_base} (n={sub['n_transitions'].iloc[0]})")
                if "HR_bpm_ci_lo" in sub.columns:
                    ci_lo = sub["HR_bpm_ci_lo"].values
                    ci_hi = sub["HR_bpm_ci_hi"].values
                    if np.any(np.isfinite(ci_lo)):
                        ax.fill_between(
                            sub["relative_tr"].values, ci_lo, ci_hi,
                            color=color, alpha=0.15,
                        )
            ax.axvline(0, color="gray", linestyle="--")
            ax.set_xlabel("TR relative to transition")
            ax.set_ylabel("HR (z-scored)")
            cr = comparison_results.get(int(state), {})
            # Show Friends-M10 r in title; append Friends-HP r if available
            parts = []
            fm = cr.get("friends_vs_movie10", {})
            if fm:
                parts.append(f"F-M10 r={fm['hr_tta_correlation']:.2f}")
            fh = cr.get("friends_vs_harrypotter", {})
            if fh:
                parts.append(f"F-HP r={fh['hr_tta_correlation']:.2f}")
            r_str = ", ".join(parts) if parts else "N/A"
            ax.set_title(f"State {state}: {r_str}")
            ax.legend()
        fig.suptitle("C4: Cross-stimulus TTAs (transitions into active states)")
        fig.tight_layout()
        for fmt in ("pdf", "png"):
            fig.savefig(os.path.join(out_dir, f"tta_comparison.{fmt}"), dpi=200)
        plt.close(fig)

    logger.info(
        "C4 complete: %d states with TTA comparison, eligible per stimulus: %s",
        len(comparison_results), eligible_counts,
    )


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    sub_id = args.sub_id
    parc = args.parcellation

    out_dir = os.path.join(
        SCRATCH_DIR, "output", "07c_cross_stimulus_physio", parc, sub_id
    )
    if args.exclude_sub_hrf:
        out_dir = os.path.join(out_dir, "sub_hrf_excluded")
    os.makedirs(out_dir, exist_ok=True)

    (
        friends_ds, m10_ds,
        friends_physio, m10_physio,
        friends_physio_raw, m10_physio_raw,
        rec_summary,
        hp_ds, hp_physio, hp_physio_raw,
    ) = load_all_inputs(sub_id, parc, args.vt)
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

    # ── Physio QC (per stimulus) ─────────────────────────────────────────
    if DATA_DIR is not None:
        qc_stimuli = [
            ("friends", friends_ds, friends_physio_raw),
            ("movie10", m10_ds, m10_physio_raw),
        ]
        if hp_ds is not None and hp_physio_raw is not None:
            qc_stimuli.append(("harrypotter", hp_ds, hp_physio_raw))

        for stim_name, decoded_dict, physio_raw_dict in qc_stimuli:
            physprep_dir = os.path.join(
                DATA_DIR, "all_about_cneuromod", f"{stim_name}.physprep",
            )
            if not os.path.isdir(physprep_dir):
                logger.warning("Physprep dir not found for %s - QC skipped", stim_name)
                continue
            pv_df = load_run_percentage_valid(
                sub_id, physprep_dir, list(decoded_dict.keys()),
                stimulus=stim_name,
            )
            qc_report = compute_qc_report(pv_df, threshold=args.pv_threshold)
            with open(os.path.join(out_dir, f"physio_qc_report_{stim_name}.json"), "w") as f:
                json.dump(qc_report, f, indent=2)
            mnar = run_eda_mnar_diagnostic(physio_raw_dict, pv_df)
            with open(os.path.join(out_dir, f"eda_mnar_diagnostic_{stim_name}.json"), "w") as f:
                json.dump(mnar, f, indent=2)
            logger.info("Physio QC report saved for %s (%d runs)", stim_name, qc_report["n_runs_total"])
    else:
        logger.warning("DATA_DIR not set - physio QC skipped")

    # ── Primary analyses ─────────────────────────────────────────────────
    analysis_c1_signature_stability(
        friends_ds, m10_ds, friends_physio, m10_physio,
        recurrence_scores, active_states, out_dir, state_flags=state_flags,
        hp_ds=hp_ds, hp_physio=hp_physio,
    )
    analysis_c2_genre_profiles(
        friends_ds, m10_ds, friends_physio, m10_physio,
        recurrence_scores, active_states, out_dir, state_flags=state_flags,
        hp_ds=hp_ds, hp_physio=hp_physio,
    )
    analysis_c3_arousal_modulation(
        friends_ds, m10_ds, friends_physio_raw, m10_physio_raw, out_dir,
        hp_ds=hp_ds, hp_physio_raw=hp_physio_raw,
    )
    analysis_c4_cross_stimulus_tta(
        friends_ds, m10_ds, friends_physio, m10_physio,
        recurrence_scores, active_states, out_dir, state_flags=state_flags,
        hp_ds=hp_ds, hp_physio=hp_physio,
    )

    logger.info("All 4 cross-stimulus analyses complete. Output: %s", out_dir)


if __name__ == "__main__":
    main()
