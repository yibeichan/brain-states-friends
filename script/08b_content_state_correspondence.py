#!/usr/bin/env python3
"""
08b_content_state_correspondence.py - Test content-state correspondence.

Post-hoc analysis testing whether brain states carry information about
narrative content. Content was not used to define states, so any association
is independent validation.

Seven analyses (A1 / A3 were redesigned 2026-04-23 — see
the design notes):

    1. **Per-state content signatures** (NEW). For each (state, feature),
       a per-epoch two-sample Mann-Whitney AUC with within-run circular-shift
       null. Two-layer BH-FDR (per-state and matrix-level).
    2. Content decoding from brain states (logistic/linear, leave-one-run-out CV)
       + optional arousal control (HR, EDA from 07a)
    3. **Multi-lag per-state content signatures** (NEW). Same per-state AUC
       framework extended across lags τ ∈ {0..6}; peak lag per (state, feature)
       identifies the HRF peak.
    4. Transition-triggered averages (TTAs) — all active states
    5. Cross-episode consistency (split-half reliability, occupancy-content corr)
    6. Content selectivity (IQR/variance for continuous, Bernoulli entropy for binary)
       + Spearman: recurrence_score vs selectivity
    7. Sensory confound control (network classification of states)

Statistical safeguards (from review):
    - Epoch-level aggregation throughout (no TR-level parametric tests)
    - Within-run circular shift for null models (preserves autocorrelation)
    - BH-FDR at q=0.05 within each analysis
    - Sub-HRF state exclusion via eligible_states.json
    - Season covariate for session-order confound

Prerequisites:
    - 08a_content_features.py completed
    - 04 decoded_states.pkl + state_means_parcel.npy
    - 05a recurrence_summary.json + eligible_states.json

Outputs:
    {SCRATCH_DIR}/output/08b_content_state_correspondence/{parcellation}/{sub_id}/
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

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, average_precision_score
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from utils.content_io import CONTENT_FEATURE_COLUMNS
from utils.state_blocks import extract_state_block_records
from utils.stats import (
    benjamini_hochberg,
    per_state_auc_mann_whitney as _per_state_auc,
    per_state_auc_grid as _compute_auc_grid,
    two_layer_bh_fdr as _apply_two_layer_fdr,
)
from utils.common import (
    _get_season, normalize_parcellation_name, resolve_stage_file,
    check_checkpoint, resolve_n_jobs,
)
from utils.plot_style import NETWORK_ORDER, assign_network
from utils.transformer_analysis import (
    circular_shift_states_by_run as _shared_circular_shift,
    load_content_eligibility,
    build_epoch_run_position_design,
    partial_effect_residualize,
    mask_a_run_opening,
)


# ── Control modes for C4 partial-effect / C1 / C3 negative controls ─────────
# See the design notes §3 and §4.1.
# "raw" is the pre-existing behavior. Other modes append a suffix to output
# filenames so existing artifacts are never clobbered.

CONTROL_MODE_SUFFIX = {
    "raw": "",
    "partial": "_partial",
    "mask33a": "_mask33a",
    "run_onset_anchored": "_run_onset_anchored",
}

MASK33A_TR = 33  # theme-song + opening-credits window on Friends a-runs

from dotenv import load_dotenv

load_dotenv()
SCRATCH_DIR = os.getenv("SCRATCH_DIR")
if SCRATCH_DIR is None:
    raise ValueError("SCRATCH_DIR must be set in the .env file")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Content-state correspondence analysis.",
    )
    parser.add_argument("--sub_id", type=str, required=True)
    parser.add_argument("--parcellation", type=str, default="atlas-4S156Parcels")
    parser.add_argument(
        "--n_permutations", type=int, default=1000,
        help="Number of permutations for null distributions (A2/A4/A5).",
    )
    parser.add_argument(
        "--n_permutations_per_state", type=int, default=500,
        help=(
            "Permutations per (state, feature) for the A1/A3 per-state signature "
            "pipeline (2026-04-23 redesign). Default 500 — min-p ≈ 1/501 ≈ 0.002, "
            "tolerable at K×16 ≈ 500-test matrix-level BH. Increase to 1000 for "
            "borderline q-values."
        ),
    )
    parser.add_argument(
        "--include_physio", action="store_true",
        help="Include HR and EDA as arousal covariates in Analysis 2.",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=-1,
        help="Number of joblib workers for permutation/state-level parallelism (-1 = all assigned CPUs).",
    )
    parser.add_argument(
        "--vt", type=str, default=None,
        help="VT suffix (e.g. '0.95') for locating 05e_a4 state_flags.csv.",
    )
    parser.add_argument(
        "--analyses", nargs="+",
        choices=["A1", "A2", "A3", "A4", "A5", "A6", "A7"],
        default=None,
        help="Run only these analyses (default: all). E.g. --analyses A2 A6",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run analyses even if checkpoint outputs already exist.",
    )
    parser.add_argument(
        "--control_modes", nargs="+",
        choices=["raw", "partial", "run_onset_anchored", "mask33a"],
        default=["raw", "partial", "run_onset_anchored", "mask33a"],
        help=(
            "Negative-control modes to run for A1/A2/A3/A6 (see "
            "the design notes). "
            "'raw' is the primary analysis. Modes emit suffixed output files "
            "(e.g. analysis_1_state_signatures_partial.json). A4/A5/A7 are "
            "not mode-varied and always run once in 'raw'."
        ),
    )
    return parser.parse_args()


def _parallel_map(func, items, n_jobs, job_name):
    """Run a simple map either serially or with joblib."""
    items = list(items)
    if not items:
        return []
    n_jobs = resolve_n_jobs(n_jobs)
    if n_jobs == 1:
        return [func(item) for item in items]
    logger.info("Running %d %s jobs with n_jobs=%s", len(items), job_name, n_jobs)
    # Threads avoid copying large decoded-state/content dictionaries per worker.
    return Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
        delayed(func)(item) for item in items
    )


def build_state_design_matrix(states, state_to_idx, season_dummies=None):
    """Build a one-hot state design matrix with optional season covariates."""
    n_states = len(state_to_idx)
    col_indices = np.array([state_to_idx[s] for s in states])
    X = np.zeros((len(states), n_states))
    X[np.arange(len(states)), col_indices] = 1.0
    if season_dummies is not None:
        X = np.hstack([X, season_dummies])
    return X


def finite_permutation_pvalue(null_values, observed_value, absolute=False):
    """Compute finite-sample permutation p-values with a +1 correction."""
    from utils.stats import permutation_pvalue
    alternative = 'two-sided' if absolute else 'greater'
    return permutation_pvalue(observed_value, null_values, alternative=alternative)


# ── Data Loading ──────────────────────────────────────────────────────────────


def load_inputs(sub_id, parc):
    """Load decoded states, content features, and recurrence scores."""
    ds_base = os.path.join(
        SCRATCH_DIR, "output", "04_combined_hdphmm", parc, sub_id,
        "final",
    ).replace("//", "/")
    ds_path = resolve_stage_file(ds_base, "decoded_states.pkl", "decoded states")
    with open(ds_path, "rb") as f:
        decoded_states = pickle.load(f)

    rec_base = os.path.join(
        SCRATCH_DIR, "output", "05a_recurrence_analysis", parc, sub_id,
    )
    rec_path = resolve_stage_file(
        rec_base, "recurrence_summary.json", "recurrence summary",
    )
    with open(rec_path) as f:
        rec_summary = json.load(f)

    # Recurrence scores as array
    recurrence_scores = np.array(rec_summary["recurrence_scores"])

    # Content features are stimulus-level, with no subject-specific subdirectory.
    content_dir = os.path.join(
        SCRATCH_DIR, "output", "08a_content_features", parc,
    )

    content_features = {}
    content_features_raw = {}
    for run_id in decoded_states:
        raw_path = os.path.join(content_dir, f"{run_id}_content_features.npy")
        norm_path = os.path.join(content_dir, f"{run_id}_content_features_norm.npy")
        if os.path.exists(norm_path):
            content_features[run_id] = np.load(norm_path)
        if os.path.exists(raw_path):
            content_features_raw[run_id] = np.load(raw_path)

    # Align content features to decoded state sequence length per run
    # (content features are stimulus-level; TR counts may differ by 1 across subjects)
    n_truncated = 0
    max_delta = 0
    for run_id in list(content_features.keys()):
        n_states = len(decoded_states[run_id])
        n_feats = len(content_features[run_id])
        if n_feats != n_states:
            delta = abs(n_feats - n_states)
            max_delta = max(max_delta, delta)
            n_truncated += 1
            n_min = min(n_states, n_feats)
            content_features[run_id] = content_features[run_id][:n_min]
            if run_id in content_features_raw:
                content_features_raw[run_id] = content_features_raw[run_id][:n_min]
    if n_truncated > 0:
        if max_delta > 5:
            logger.warning("Aligned content features for %d runs (max delta: %d TRs)", n_truncated, max_delta)
        else:
            logger.info("Aligned content features for %d runs (max delta: %d TRs)", n_truncated, max_delta)

    logger.info(
        "Loaded: %d decoded runs, %d with content features, %d states (%d active)",
        len(decoded_states), len(content_features),
        len(recurrence_scores),
        int(np.sum(recurrence_scores > 0)),
    )
    return decoded_states, content_features, content_features_raw, recurrence_scores


def get_content_eligibility(sub_id, parcellation, vt=None):
    """Return the project-wide content-eligibility dict from 05e_a4.

    Wraps :func:`utils.transformer_analysis.load_content_eligibility`. See
    the ``2026-04-09_08_transformer_refactor_design.md`` design doc §6 for
    the full eligibility-category convention. Returns a dict with keys
    ``content_eligible``, ``run_onset_anchored``, ``season_temporal``,
    ``basic_sub_hrf``, ``eligibility_source``.
    """
    return load_content_eligibility(
        sub_id=sub_id, parcellation=parcellation,
        scratch_dir=SCRATCH_DIR, vt=vt,
    )


def load_physio_features(sub_id, decoded_states):
    """Load per-run physio features from 07a output.

    Returns dict {run_id: np.ndarray (n_trs, 7)} or empty dict if unavailable.
    """
    physio_dir = os.path.join(
        SCRATCH_DIR, "output", "07a_physio_features", sub_id,
    )
    physio = {}
    if not os.path.isdir(physio_dir):
        logger.info("Physio directory not found: %s", physio_dir)
        return physio

    for run_id in decoded_states:
        p = os.path.join(physio_dir, f"{run_id}_physio_features.npy")
        if os.path.exists(p):
            physio[run_id] = np.load(p)
    logger.info("Loaded physio features for %d / %d runs", len(physio), len(decoded_states))
    return physio


# ── Per-state content signature helpers (2026-04-23 redesign) ────────────────
#
# These helpers implement the per-state × per-feature Mann-Whitney AUC testing
# framework spec'd in the design notes.
# A1 (single-lag) and A3 (multi-lag) share the same core: per permutation,
# circularly shift the TR-level state sequence within each run, re-extract
# epoch blocks, compute epoch-mean features, optionally residualize against
# an epoch-center run-position cubic (C4 partial mode), and score each
# (state, feature) pair via AUC = U / (n_1 * n_0).

PER_STATE_SEED_BASE_A1 = 15_000_000
PER_STATE_SEED_BASE_A3 = 16_000_000


def _build_run_boundaries_ordered(run_ids_order, decoded_states):
    """Return ``(concat_states, run_boundaries)`` for a fixed run ordering."""
    seqs = [np.asarray(decoded_states[rid]) for rid in run_ids_order]
    run_boundaries = []
    offset = 0
    for seq in seqs:
        run_boundaries.append((offset, offset + len(seq)))
        offset += len(seq)
    concat = np.concatenate(seqs) if seqs else np.empty(0, dtype=int)
    return concat, run_boundaries


def _null_row_to_decoded(null_row, run_ids_order, run_boundaries):
    """Split a concatenated null TR sequence back into a ``{run_id: seq}`` dict."""
    return {rid: null_row[s:e] for rid, (s, e) in zip(run_ids_order, run_boundaries)}


def _epoch_feats_at_lag(block_records, content_features, lag):
    """Compute lag-aligned epoch-mean feature matrix.

    Pair content at ``(t - lag)`` with state at ``t``. Drops epochs where the
    lag-shifted window falls outside ``[0, len(content))`` or has fewer than 2
    TRs. Returns ``(filtered_records, states_arr, feats_mat)``.
    """
    n_features = len(CONTENT_FEATURE_COLUMNS)
    filtered_records = []
    feats_list = []
    states_list = []
    for rec in block_records:
        run_id = rec["run_id"]
        if run_id not in content_features:
            continue
        feats = content_features[run_id]
        start = rec["start_tr"] - lag
        end = rec["end_tr"] - lag
        if start < 0 or end <= start:
            continue
        end = min(end, len(feats))
        if end - start < 2:
            continue
        feats_list.append(np.nanmean(feats[start:end], axis=0))
        states_list.append(int(rec["state"]))
        filtered_records.append(rec)
    if not feats_list:
        return [], np.empty(0, dtype=int), np.empty((0, n_features))
    return (
        filtered_records,
        np.asarray(states_list, dtype=int),
        np.asarray(feats_list, dtype=np.float64),
    )


def _maybe_residualize(feats_mat, records, control_mode):
    """Apply C4 epoch-center polynomial residualization when in partial mode."""
    if control_mode != "partial" or feats_mat.size == 0:
        return feats_mat
    D = build_epoch_run_position_design(records, degree=3)
    return partial_effect_residualize(feats_mat, D)


def _observed_per_state_signature(
    decoded_states, content_features, eligible_set, recurrence_scores,
    target_states, lags, control_mode,
):
    """Compute the observed per-(lag, state, feature) AUC tensor and
    per-(lag, state) epoch counts.

    Returns a dict keyed by lag, each value a dict with:
      - "aucs": (n_states, n_features) array of AUCs
      - "signs": (n_states, n_features) array of signs (int8)
      - "n_epochs_state": (n_states,) array of per-state epoch counts
      - "n_epochs_other": (n_states,) array of complement counts
    """
    block_records = extract_state_block_records(
        decoded_states, recurrence_scores, include_states=eligible_set,
    )
    out = {}
    for lag in lags:
        filt, states_arr, feats_mat = _epoch_feats_at_lag(
            block_records, content_features, lag,
        )
        feats_mat = _maybe_residualize(feats_mat, filt, control_mode)
        aucs, signs = _compute_auc_grid(
            states_arr, feats_mat, target_states, compute_signs=True,
        )
        n_ep = len(states_arr)
        n_state = np.array(
            [int((states_arr == s).sum()) for s in target_states], dtype=np.int64,
        )
        n_other = n_ep - n_state
        out[lag] = {
            "aucs": aucs,
            "signs": signs,
            "n_epochs_state": n_state,
            "n_epochs_other": n_other,
            "n_epochs_total": n_ep,
            "filtered_records": filt,
        }
    return out


def _null_per_state_signature_one(
    null_row, run_ids_order, run_boundaries, content_features,
    recurrence_scores, eligible_set, target_states, lags, control_mode,
):
    """Single-permutation AUC tensor across lags.

    Returns a dict keyed by lag, each value a ``(n_states, n_features)`` array
    of null AUCs (NaN where either group lacks ≥ 2 finite values).
    """
    shifted = _null_row_to_decoded(null_row, run_ids_order, run_boundaries)
    block_records = extract_state_block_records(
        shifted, recurrence_scores, include_states=eligible_set,
    )
    perm_out = {}
    for lag in lags:
        filt, states_arr, feats_mat = _epoch_feats_at_lag(
            block_records, content_features, lag,
        )
        feats_mat = _maybe_residualize(feats_mat, filt, control_mode)
        aucs, _ = _compute_auc_grid(
            states_arr, feats_mat, target_states, compute_signs=False,
        )
        perm_out[lag] = aucs
    return perm_out


def _aggregate_null_aucs(null_perm_list, lags, n_states, n_features):
    """Stack per-permutation AUC dicts into ``(n_perm, n_states, n_features)`` per lag."""
    out = {}
    for lag in lags:
        stacked = np.stack([p[lag] for p in null_perm_list], axis=0)
        out[lag] = stacked  # (n_perm, K, F)
    return out


def _per_state_signature_pvalues(obs_aucs, null_aucs_stack):
    """Per-cell two-sided permutation p-value on ``|auc - 0.5|``.

    Parameters
    ----------
    obs_aucs : (K, F) array
    null_aucs_stack : (n_perm, K, F) array

    Returns
    -------
    p_perm : (K, F) — ``(1 + # |null-0.5| ≥ |obs-0.5|) / (1 + n_valid)``
    valid_perm_fraction : (K, F) — fraction of permutations producing a finite AUC.
    """
    n_perm, K, F = null_aucs_stack.shape
    obs_dist = np.abs(obs_aucs - 0.5)
    null_dist = np.abs(null_aucs_stack - 0.5)
    finite_null = np.isfinite(null_dist)
    n_valid = finite_null.sum(axis=0).astype(np.int64)  # (K, F)
    # Count "|null-0.5| >= |obs-0.5|" over finite null entries.
    cmp = np.zeros((K, F), dtype=np.int64)
    for i in range(n_perm):
        mask = finite_null[i]
        cmp[mask] += (null_dist[i][mask] >= obs_dist[mask]).astype(np.int64)
    p = np.full((K, F), np.nan)
    with np.errstate(invalid="ignore"):
        valid = (n_valid > 0) & np.isfinite(obs_dist)
        p[valid] = (1 + cmp[valid]) / (1 + n_valid[valid])
    valid_fraction = n_valid.astype(np.float64) / max(n_perm, 1)
    return p, valid_fraction


def _pack_lag_signature_dict(
    lag_data, target_states, recurrence_scores, p_perm, p_fdr_per_state,
    p_fdr_matrix, valid_frac,
):
    """Build the ``states`` sub-dict for a single lag matching §4.2 schema."""
    features = list(CONTENT_FEATURE_COLUMNS)
    aucs = lag_data["aucs"]
    signs = lag_data["signs"]
    n_state = lag_data["n_epochs_state"]
    n_other = lag_data["n_epochs_other"]

    def _to_jsonable(x):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return None
        return round(float(x), 6)

    states_dict = {}
    for si, state_id in enumerate(target_states):
        feat_auc = {}
        feat_sign = {}
        feat_p = {}
        feat_fdr_s = {}
        feat_fdr_m = {}
        feat_vpf = {}
        for fi, feat in enumerate(features):
            feat_auc[feat] = _to_jsonable(aucs[si, fi])
            feat_sign[feat] = int(signs[si, fi]) if signs is not None else 0
            feat_p[feat] = _to_jsonable(p_perm[si, fi])
            feat_fdr_s[feat] = _to_jsonable(p_fdr_per_state[si, fi])
            feat_fdr_m[feat] = _to_jsonable(p_fdr_matrix[si, fi])
            feat_vpf[feat] = _to_jsonable(valid_frac[si, fi])
        states_dict[str(int(state_id))] = {
            "recurrence_score": float(recurrence_scores[int(state_id)]),
            "n_epochs_state": int(n_state[si]),
            "n_epochs_other": int(n_other[si]),
            "feature_auc": feat_auc,
            "feature_sign": feat_sign,
            "feature_p": feat_p,
            "feature_p_fdr_per_state": feat_fdr_s,
            "feature_p_fdr_matrix": feat_fdr_m,
            "valid_perm_fraction": feat_vpf,
        }
    return states_dict


def _build_csv_rows(
    target_states, recurrence_scores, lag_data, p_perm, p_fdr_per_state,
    p_fdr_matrix, valid_frac, control_mode, n_permutations, lag=None,
):
    """One row per (state, feature) for the state_content_profiles CSV."""
    rows = []
    aucs = lag_data["aucs"]
    signs = lag_data["signs"]
    n_state = lag_data["n_epochs_state"]
    n_other = lag_data["n_epochs_other"]
    for si, state_id in enumerate(target_states):
        for fi, feat in enumerate(CONTENT_FEATURE_COLUMNS):
            row = {
                "state": int(state_id),
                "recurrence_score": float(recurrence_scores[int(state_id)]),
                "feature": feat,
                "control_mode": control_mode,
                "auc": float(aucs[si, fi]) if np.isfinite(aucs[si, fi]) else None,
                "sign": int(signs[si, fi]) if signs is not None else 0,
                "p_perm": float(p_perm[si, fi]) if np.isfinite(p_perm[si, fi]) else None,
                "p_fdr_per_state": (
                    float(p_fdr_per_state[si, fi])
                    if np.isfinite(p_fdr_per_state[si, fi]) else None
                ),
                "p_fdr_matrix": (
                    float(p_fdr_matrix[si, fi])
                    if np.isfinite(p_fdr_matrix[si, fi]) else None
                ),
                "n_epochs_state": int(n_state[si]),
                "n_epochs_other": int(n_other[si]),
                "n_permutations": int(n_permutations),
                "valid_perm_fraction": (
                    float(valid_frac[si, fi])
                    if np.isfinite(valid_frac[si, fi]) else None
                ),
            }
            if lag is not None:
                row["lag"] = int(lag)
            rows.append(row)
    return rows


def _plot_state_content_heatmap(
    aucs, p_fdr_matrix, target_states, recurrence_scores, features, out_path,
    title_suffix="",
):
    """K × F heatmap of AUC - 0.5 with q<0.05 cells annotated."""
    if len(target_states) == 0:
        return
    # Sort states by recurrence score (descending).
    rec_vec = np.array([float(recurrence_scores[int(s)]) for s in target_states])
    order = np.argsort(-rec_vec)
    aucs_sorted = aucs[order]
    p_sorted = p_fdr_matrix[order]
    states_sorted = [int(target_states[i]) for i in order]

    fig_w = max(8, 0.55 * len(features) + 2)
    fig_h = max(5, 0.3 * len(states_sorted) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    data = aucs_sorted - 0.5
    vmax = float(np.nanmax(np.abs(data))) if np.isfinite(np.nanmax(np.abs(data))) else 0.2
    vmax = max(vmax, 0.05)
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(states_sorted)))
    ax.set_yticklabels(
        [f"S{s} (r={rec_vec[order[i]]:.2f})" for i, s in enumerate(states_sorted)],
        fontsize=7,
    )
    # Mark q<0.05 with a small dot.
    for si in range(len(states_sorted)):
        for fi in range(len(features)):
            q = p_sorted[si, fi]
            if np.isfinite(q) and q < 0.05:
                ax.text(fi, si, "·", ha="center", va="center", color="black", fontsize=10)
    ax.set_xlabel("Content feature")
    ax.set_ylabel("State (sorted by recurrence)")
    ax.set_title(f"Per-state content signature (AUC − 0.5){title_suffix}")
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="AUC − 0.5")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Analysis 1: Per-state content signatures (2026-04-23 redesign) ───────────


def analysis_1_state_content_signatures(
    decoded_states, content_features, recurrence_scores, eligible_states,
    out_dir, n_permutations=500, n_jobs=1, control_mode="raw",
):
    """Per-state × per-feature content-signature profiling at lag=0.

    Replaces the legacy joint Kruskal-Wallis A1 with a per-(state, feature)
    two-sample AUC test. For each eligible state X and each content feature Y,
    reports the Mann-Whitney AUC of ``P(y | X > y | other)`` with a
    within-run circular-shift permutation null and two-layer BH-FDR
    (per-state across 16 features, matrix-level across K × 16 tests).

    Output files (per subject, per mode):
      - ``analysis_1_state_signatures{suffix}.json`` (§4.2 schema)
      - ``state_content_profiles{suffix}.csv`` (§4.3 schema, lag column omitted)
      - ``state_content_heatmap{suffix}.pdf`` (K × 16 AUC heatmap)

    See the design notes §3 for the
    full statistical spec.
    """
    suffix = CONTROL_MODE_SUFFIX.get(control_mode, "")
    label = f" [{control_mode}]" if control_mode != "raw" else ""
    logger.info("Analysis 1%s: Per-state content signatures", label)

    eligible_set = set(int(s) for s in eligible_states)
    run_ids_order = sorted(decoded_states.keys())
    concat_states, run_boundaries = _build_run_boundaries_ordered(
        run_ids_order, decoded_states,
    )

    # Observed per-state AUC grid at lag=0.
    lags = [0]
    # First pass: find states that actually show up in the eligible epoch set.
    observed = _observed_per_state_signature(
        decoded_states, content_features, eligible_set, recurrence_scores,
        target_states=sorted(eligible_set), lags=lags, control_mode=control_mode,
    )
    lag0_obs = observed[0]
    n_epochs_total = lag0_obs["n_epochs_total"]
    if n_epochs_total < 10:
        logger.warning("Analysis 1%s: too few epochs (%d) — skipping", label, n_epochs_total)
        return

    # Keep all eligible states (even 0-epoch ones) so the output matrix
    # dimension stays interpretable; AUC cells for degenerate states will be NaN.
    target_states = sorted(eligible_set)

    # Precompute null TR-level sequences once (mode-scoped).
    mode_idx = list(CONTROL_MODE_SUFFIX.keys()).index(control_mode)
    seed_base = PER_STATE_SEED_BASE_A1 + mode_idx * 1_000_000
    logger.info(
        "Analysis 1%s: %d permutations over %d eligible states × %d features",
        label, n_permutations, len(target_states), len(CONTENT_FEATURE_COLUMNS),
    )

    null_rows = Parallel(
        n_jobs=resolve_n_jobs(n_jobs), prefer="threads", verbose=0,
    )(
        delayed(_null_per_state_signature_one)(
            _shared_circular_shift(concat_states, run_boundaries, seed_base + i),
            run_ids_order, run_boundaries, content_features, recurrence_scores,
            eligible_set, target_states, lags, control_mode,
        )
        for i in range(n_permutations)
    )

    null_stack = _aggregate_null_aucs(
        null_rows, lags, len(target_states), len(CONTENT_FEATURE_COLUMNS),
    )[0]  # (n_perm, K, F)

    p_perm, valid_frac = _per_state_signature_pvalues(lag0_obs["aucs"], null_stack)
    p_fdr_ps, p_fdr_mx = _apply_two_layer_fdr(p_perm)

    # JSON output
    states_dict = _pack_lag_signature_dict(
        lag0_obs, target_states, recurrence_scores, p_perm, p_fdr_ps, p_fdr_mx,
        valid_frac,
    )
    payload = {
        "control_mode": control_mode,
        "n_states": len(target_states),
        "n_features": len(CONTENT_FEATURE_COLUMNS),
        "n_permutations": int(n_permutations),
        "effect_size_metric": "auc_mann_whitney",
        "fdr_strategy": "two_layer_bh",
        "features": list(CONTENT_FEATURE_COLUMNS),
        "states": states_dict,
        "cross_subject_defer": True,
    }
    out_json = os.path.join(out_dir, f"analysis_1_state_signatures{suffix}.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    # CSV: one row per (state, feature).
    csv_rows = _build_csv_rows(
        target_states, recurrence_scores, lag0_obs, p_perm, p_fdr_ps, p_fdr_mx,
        valid_frac, control_mode, n_permutations, lag=None,
    )
    pd.DataFrame(csv_rows).to_csv(
        os.path.join(out_dir, f"state_content_profiles{suffix}.csv"), index=False,
    )

    # Heatmap
    _plot_state_content_heatmap(
        lag0_obs["aucs"], p_fdr_mx, target_states, recurrence_scores,
        list(CONTENT_FEATURE_COLUMNS),
        os.path.join(out_dir, f"state_content_heatmap{suffix}.pdf"),
        title_suffix=f" [{control_mode}]" if control_mode != "raw" else "",
    )

    n_sig_matrix = int(np.sum(np.isfinite(p_fdr_mx) & (p_fdr_mx < 0.05)))
    sparse_null = int(np.sum(valid_frac < 0.5))
    logger.info(
        "Analysis 1%s complete: %d (state, feature) pairs at q_matrix<0.05; "
        "%d pairs flagged sparse-null (valid_perm_fraction<0.5)",
        label, n_sig_matrix, sparse_null,
    )


# ── Analysis 2: Content decoding ──────────────────────────────────────────────



# ── Module-level decoder helpers (shared between joint + per-state) ───────────
#
# These were originally nested inside ``analysis_2_decoding`` as closures over
# a local ``REG_C``. Hoisted in the 08g deep-review fix series so the new
# ``analysis_2_decoding_per_state`` function can reuse the EXACT same fold
# construction and CV math, guaranteeing per-state and joint AUCs are
# directly comparable. Bit-identical regression check on the joint output is
# part of the rollout.
REG_C = 0.01      # L2 regularization for the per-fold logistic CV — same observed and null
RIDGE_ALPHA = 10.0  # L2 regularization for the per-fold ridge CV


def _run_logistic_cv(X_mat, y_vec, folds):
    """Leave-one-run-out logistic regression with pre-computed folds.

    Returns ``(y_pred, auc_roc, auc_pr)`` or ``(y_pred, None, None)`` if the
    pooled fold predictions contain ≤ 10 samples or only one class.
    """
    y_pred = np.full(len(y_vec), np.nan)
    for train_idx, test_idx in folds:
        clf = LogisticRegression(solver="liblinear", max_iter=200, C=REG_C)
        clf.fit(X_mat[train_idx], y_vec[train_idx])
        y_pred[test_idx] = clf.predict_proba(X_mat[test_idx])[:, 1]
    valid = np.isfinite(y_pred)
    if np.sum(valid) <= 10 or len(np.unique(y_vec[valid])) < 2:
        return y_pred, None, None
    auc = float(roc_auc_score(y_vec[valid], y_pred[valid]))
    auc_pr = float(average_precision_score(y_vec[valid], y_pred[valid]))
    return y_pred, auc, auc_pr


def _run_ridge_cv(X_mat, y_vec, folds):
    """Leave-one-run-out Ridge regression with pre-computed folds.

    Returns the pooled-prediction R² or ``None`` if fewer than 10 finite
    predictions are pooled.
    """
    y_pred = np.full(len(y_vec), np.nan)
    for train_idx, test_idx in folds:
        reg = Ridge(alpha=RIDGE_ALPHA)
        reg.fit(X_mat[train_idx], y_vec[train_idx])
        y_pred[test_idx] = reg.predict(X_mat[test_idx])
    valid = np.isfinite(y_pred)
    if np.sum(valid) < 10:
        return None
    ss_res = np.sum((y_vec[valid] - y_pred[valid]) ** 2)
    ss_tot = np.sum((y_vec[valid] - y_vec[valid].mean()) ** 2)
    return float(1 - ss_res / max(ss_tot, 1e-10))


def _build_lo_run_folds(unique_runs, runs, seasons, y_binary):
    """Build leave-one-run-out folds for the joint and per-state decoders.

    Logistic folds additionally require:
      - test seasons ⊆ train seasons (so season covariates are not extrapolated);
      - both classes present in training labels.
    Ridge folds only require the size check.

    Returns ``(logistic_folds, ridge_folds, run_boundaries)``. ``run_boundaries``
    is a list of ``(start, end)`` index pairs in epoch order, suitable for the
    within-run circular-shift null helper.
    """
    logistic_folds = []
    ridge_folds = []
    for held_out in unique_runs:
        test_idx = np.where(runs == held_out)[0]
        train_idx = np.where(runs != held_out)[0]
        if len(train_idx) < 10 or len(test_idx) < 2:
            continue
        ridge_folds.append((train_idx, test_idx))
        train_seasons = set(seasons[train_idx])
        test_seasons = set(seasons[test_idx])
        if not test_seasons.issubset(train_seasons):
            continue
        if len(np.unique(y_binary[train_idx])) < 2:
            continue
        logistic_folds.append((train_idx, test_idx))

    run_boundaries = []
    for run_id in unique_runs:
        idx = np.where(runs == run_id)[0]
        if len(idx) > 0:
            run_boundaries.append((idx[0], idx[-1] + 1))

    return logistic_folds, ridge_folds, run_boundaries


def analysis_2_decoding(
    decoded_states, content_features, recurrence_scores, eligible_states,
    out_dir, n_permutations, physio_features=None, n_jobs=1,
    control_mode="raw",
):
    """Content decoding from brain state identity.

    2a: Binary (speech_presence) -- logistic regression, AUC-ROC + AUC-PR
    2b: Continuous (dialogue_rate) -- Ridge regression, R-squared
    Leave-one-run-out CV. Within-run circular shift null.
    Optional: re-run with HR + EDA covariates for arousal control.

    ``control_mode`` drives the output filename suffix (see
    ``CONTROL_MODE_SUFFIX``) so each mode writes to its own file without
    clobbering. When ``"partial"``, an epoch-center run-position cubic
    polynomial is appended as nuisance covariates in the LORO design matrix
    (C4 partial-effect negative control). The polynomial columns are invariant
    under the state-label circular-shift, so the null picks up the same
    position-conditioning as the observed fit.
    """
    suffix = CONTROL_MODE_SUFFIX.get(control_mode, "")
    banner = (
        f"Analysis 2 [{control_mode}]" if control_mode != "raw" else "Analysis 2"
    )
    logger.info("%s: Content decoding from brain states", banner)

    block_records = extract_state_block_records(
        decoded_states, recurrence_scores, include_states=set(eligible_states),
    )
    eligible_set = set(eligible_states)

    # Build epoch-level dataset
    epochs = []
    filtered_records = []
    for rec in block_records:
        run_id = rec["run_id"]
        if run_id not in content_features:
            continue
        feats = content_features[run_id]
        start, end = rec["start_tr"], rec["end_tr"]
        if end <= start or start >= len(feats) or (end - start) < 2:
            continue
        epoch_mean = np.nanmean(feats[start:end], axis=0)
        try:
            season = _get_season(run_id)
        except ValueError:
            season = 0

        row = {
            "state": rec["state"],
            "run_id": run_id,
            "season": season,
            "speech_presence": epoch_mean[0],
            "dialogue_rate": epoch_mean[1],
        }

        # Physio covariates (epoch-level means)
        if physio_features and run_id in physio_features:
            phys = physio_features[run_id]
            end_p = min(end, len(phys))
            if end_p > start:
                phys_epoch = phys[start:end_p]
                row["hr_mean"] = float(np.nanmean(phys_epoch[:, 0]))  # HR_bpm
                row["eda_mean"] = float(np.nanmean(phys_epoch[:, 4]))  # EDA_tonic

        epochs.append(row)
        filtered_records.append(rec)

    if len(epochs) < 20:
        logger.warning("Too few epochs (%d) for decoding — skipping", len(epochs))
        return
    # Invariant: epochs and filtered_records are appended in lockstep so
    # that position_dummies built from filtered_records aligns row-wise
    # with the design matrix X built from df.
    assert len(filtered_records) == len(epochs)

    df = pd.DataFrame(epochs)

    # One-hot encode states (vectorized)
    unique_states = sorted(df["state"].unique())
    state_to_idx = {s: i for i, s in enumerate(unique_states)}
    state_arr = df["state"].values
    X = build_state_design_matrix(state_arr, state_to_idx)

    # Add season covariates
    seasons = df["season"].values
    unique_seasons = sorted(set(seasons))
    season_dummies = None
    if len(unique_seasons) > 1:
        season_dummies = np.zeros((len(df), len(unique_seasons) - 1))
        season_to_idx = {s: i for i, s in enumerate(unique_seasons)}
        for i, s in enumerate(seasons):
            idx = season_to_idx[s]
            if idx > 0:
                season_dummies[i, idx - 1] = 1.0
        X = np.hstack([X, season_dummies])

    # C4 partial-effect: append epoch-center cubic-polynomial columns as
    # nuisance covariates. Invariant under state-label permutation, so the
    # null distribution conditions on the same position structure.
    position_dummies = None
    if control_mode == "partial":
        position_dummies = build_epoch_run_position_design(
            filtered_records, degree=3, intercept=False,
        )
        X = np.hstack([X, position_dummies])

    runs = df["run_id"].values
    unique_runs = sorted(set(runs))

    results = {"n_epochs": len(df), "n_states": len(unique_states), "n_runs": len(unique_runs)}

    # Targets and folds (built via the hoisted helper so the per-state
    # decoder shares the EXACT same logistic_folds / ridge_folds /
    # run_boundaries — see ``_build_lo_run_folds``).
    y_binary = (df["speech_presence"].values > 0.5).astype(float)
    y_cont = df["dialogue_rate"].values
    logistic_folds, ridge_folds, run_boundaries = _build_lo_run_folds(
        unique_runs, runs, seasons, y_binary,
    )

    # 2a: Binary decoding (speech_presence)
    base_rate = float(y_binary.mean())
    results["speech_base_rate"] = round(base_rate, 3)

    _, observed_auc, observed_auc_pr = _run_logistic_cv(X, y_binary, logistic_folds)
    if observed_auc is not None:
        results["binary_auc_roc"] = round(observed_auc, 4)
        results["binary_auc_pr"] = round(observed_auc_pr, 4)

        # Null distribution: within-run circular shift of state labels (parallelized)
        def _one_binary_perm(seed):
            """Single permutation: shift states, build X, run CV, return AUC."""
            states_perm = _shared_circular_shift(
                state_arr, run_boundaries, seed=seed, min_shift=1,
            )
            X_p = build_state_design_matrix(states_perm, state_to_idx, season_dummies)
            if position_dummies is not None:
                X_p = np.hstack([X_p, position_dummies])
            _, auc, _ = _run_logistic_cv(X_p, y_binary, logistic_folds)
            return auc

        # Explicit per-worker seeds keep the parallel null reproducible.
        # Use process-based parallelism for CPU-bound sklearn fits. The
        # installed joblib expects the portable hint "processes" rather than
        # the backend name "loky", which caused SLURM job 11258581 to fail.
        perm_seeds = np.arange(42, 42 + n_permutations)
        _n = resolve_n_jobs(n_jobs)
        if _n > 1:
            logger.info("Running %d Analysis 2 binary permutation jobs with n_jobs=%s",
                        n_permutations, _n)
            null_aucs_raw = Parallel(n_jobs=_n, prefer="processes", verbose=0)(
                delayed(_one_binary_perm)(seed) for seed in perm_seeds
            )
        else:
            null_aucs_raw = [_one_binary_perm(s) for s in perm_seeds]
        null_aucs = [a for a in null_aucs_raw if a is not None]

        if null_aucs:
            results["binary_p_value"] = round(
                finite_permutation_pvalue(null_aucs, observed_auc), 4,
            )
            results["binary_null_auc_mean"] = round(float(np.mean(null_aucs)), 4)
            results["binary_null_auc_std"] = round(float(np.std(null_aucs)), 4)
            results["binary_n_permutations"] = len(null_aucs)

    # 2b: Continuous decoding (dialogue_rate)
    observed_r2 = _run_ridge_cv(X, y_cont, ridge_folds)
    if observed_r2 is not None:
        results["continuous_r_squared"] = round(observed_r2, 4)

        def _one_continuous_perm(seed):
            """Single permutation for Ridge R²."""
            states_perm = _shared_circular_shift(
                state_arr, run_boundaries, seed=seed, min_shift=1,
            )
            X_p = build_state_design_matrix(states_perm, state_to_idx, season_dummies)
            if position_dummies is not None:
                X_p = np.hstack([X_p, position_dummies])
            return _run_ridge_cv(X_p, y_cont, ridge_folds)

        perm_seeds2 = np.arange(1000, 1000 + n_permutations)
        if _n > 1:
            logger.info("Running %d Analysis 2 continuous permutation jobs with n_jobs=%s",
                        n_permutations, _n)
            null_r2s_raw = Parallel(n_jobs=_n, prefer="processes", verbose=0)(
                delayed(_one_continuous_perm)(seed) for seed in perm_seeds2
            )
        else:
            null_r2s_raw = [_one_continuous_perm(s) for s in perm_seeds2]
        null_r2s = [r for r in null_r2s_raw if r is not None]

        if null_r2s:
            results["continuous_p_value"] = round(
                finite_permutation_pvalue(null_r2s, observed_r2), 4,
            )
            results["continuous_null_r2_mean"] = round(float(np.mean(null_r2s)), 4)
            results["continuous_null_r2_std"] = round(float(np.std(null_r2s)), 4)
            results["continuous_n_permutations"] = len(null_r2s)

    # 2c: Arousal-controlled decoding (optional)
    has_physio = "hr_mean" in df.columns and df["hr_mean"].notna().sum() > len(df) * 0.5
    if has_physio:
        hr = df["hr_mean"].fillna(0).values
        eda = df["eda_mean"].fillna(0).values
        arousal_cols = np.column_stack([hr, eda])
        X_arousal = np.hstack([X, arousal_cols])

        _, auc_ctrl, auc_pr_ctrl = _run_logistic_cv(X_arousal, y_binary, logistic_folds)
        r2_ctrl = _run_ridge_cv(X_arousal, y_cont, ridge_folds)
        results["with_arousal_control"] = {
            "binary_auc_roc": round(auc_ctrl, 4) if auc_ctrl is not None else None,
            "binary_auc_pr": round(auc_pr_ctrl, 4) if auc_pr_ctrl is not None else None,
            "continuous_r_squared": round(r2_ctrl, 4) if r2_ctrl is not None else None,
            "n_runs_with_physio": int(df["hr_mean"].notna().sum()),
        }

    out_path = os.path.join(out_dir, f"analysis_2_decoding{suffix}.json")
    results["_control_mode"] = control_mode
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("%s complete: AUC-ROC=%.3f, R²=%.3f",
                banner,
                results.get("binary_auc_roc", float("nan")),
                results.get("continuous_r_squared", float("nan")))


def analysis_2_decoding_per_state(
    decoded_states, content_features, recurrence_scores, eligible_states,
    out_dir, n_permutations, n_jobs=1,
):
    """Per-state univariate content decoding.

    For each ``content_eligible`` state, fits a univariate logistic / ridge
    decoder using a single one-hot column ("is this epoch in state s") plus
    season covariates, with the EXACT same leave-one-run-out folds and
    within-run circular-shift null as :func:`analysis_2_decoding`. The two
    functions share :func:`_build_lo_run_folds`, :func:`_run_logistic_cv`,
    and :func:`_run_ridge_cv`, so a per-state ``binary_auc_roc`` is directly
    comparable to the joint ``binary_auc_roc`` reported in
    ``analysis_2_decoding.json`` (the only difference is design-matrix
    rank — joint uses ``n_states`` columns, per-state uses 1).

    Why this exists: 08g D5 needs a per-state content score that is
    commensurate with 08d D2's per-state transformer AUC. The pre-existing
    ``analysis_6_selectivity.csv`` mixed bounded ``bernoulli_selectivity``
    with unbounded ``iqr`` in the same ``value`` column, so taking
    ``max(value)`` per state was apples-to-oranges. This function provides
    the clean per-state AUC that D5 actually wants.

    Output schema (``analysis_2_decoding_per_state.json``)::

        {
          "n_eligible_states": int,
          "n_permutations": int,
          "speech_base_rate": float,
          "per_state": {
              "<state_id>": {
                  "n_epochs_in_state": int,
                  "binary_auc_roc": float | None,
                  "binary_auc_pr": float | None,
                  "continuous_r_squared": float | None,
                  "binary_p_value": float | None,
                  "continuous_p_value": float | None,
                  "binary_p_fdr": float | None,
                  "continuous_p_fdr": float | None,
                  "binary_null_auc_mean": float | None,
                  "continuous_null_r2_mean": float | None,
              },
              ...
          }
        }

    BH-FDR is applied across the eligible states, separately for the binary
    and continuous p-value columns.
    """
    logger.info("Analysis 2 [per-state]: Univariate per-state content decoding")

    # Re-use the joint-decoder epoch construction. We deliberately rebuild
    # the same dataframe rather than threading state through main() because
    # (a) it makes this function callable independently and (b) the joint
    # decoder is already cheap relative to the per-state permutation cost.
    block_records = extract_state_block_records(
        decoded_states, recurrence_scores, include_states=set(eligible_states),
    )
    epochs = []
    for rec in block_records:
        run_id = rec["run_id"]
        if run_id not in content_features:
            continue
        feats = content_features[run_id]
        start, end = rec["start_tr"], rec["end_tr"]
        if end <= start or start >= len(feats) or (end - start) < 2:
            continue
        epoch_mean = np.nanmean(feats[start:end], axis=0)
        try:
            season = _get_season(run_id)
        except ValueError:
            season = 0
        epochs.append({
            "state": rec["state"],
            "run_id": run_id,
            "season": season,
            "speech_presence": epoch_mean[0],
            "dialogue_rate": epoch_mean[1],
        })

    if len(epochs) < 20:
        logger.warning(
            "Per-state: too few epochs (%d) — skipping", len(epochs),
        )
        return

    df = pd.DataFrame(epochs)
    state_arr = df["state"].values
    runs = df["run_id"].values
    seasons = df["season"].values
    unique_runs = sorted(set(runs))

    y_binary = (df["speech_presence"].values > 0.5).astype(float)
    y_cont = df["dialogue_rate"].values
    logistic_folds, ridge_folds, run_boundaries = _build_lo_run_folds(
        unique_runs, runs, seasons, y_binary,
    )

    # Season covariates (same scheme as the joint decoder).
    unique_seasons = sorted(set(seasons))
    if len(unique_seasons) > 1:
        season_dummies = np.zeros((len(df), len(unique_seasons) - 1))
        season_to_idx = {s: i for i, s in enumerate(unique_seasons)}
        for i, s in enumerate(seasons):
            idx = season_to_idx[s]
            if idx > 0:
                season_dummies[i, idx - 1] = 1.0
    else:
        season_dummies = None

    # Eligible states actually present in the epoch dataframe.
    states_in_df = set(int(s) for s in df["state"].unique())
    states_to_test = sorted(int(s) for s in eligible_states if int(s) in states_in_df)
    logger.info(
        "Per-state: testing %d eligible states across %d epochs / %d runs",
        len(states_to_test), len(df), len(unique_runs),
    )

    base_rate = float(y_binary.mean())

    def _univariate_X(state_id):
        col = (state_arr == state_id).astype(float).reshape(-1, 1)
        if season_dummies is None:
            return col
        return np.hstack([col, season_dummies])

    # Per-worker seeds: disjoint from the joint decoder's seed ranges
    # (joint binary uses 42..42+n_perm; joint continuous uses 1000..1000+n_perm)
    # AND from the 08-series correspondence seed slots (D1..rec×depth, which
    # max out at 100_000 + sub-offsets in 08g; see
    # ``utils.transformer_analysis.BOOTSTRAP_SEED_RECURRENCE_DEPTH``).
    # Per-state uses a base of 200_000 + state_id_index * (2 * n_perm) so each
    # state's binary and continuous nulls have non-overlapping seeds, and the
    # schedule is reproducible regardless of state ordering. With n_perm=1000
    # and ~50 eligible states per subject, the per-state range is
    # [200_000, 300_000), well above the 08-series ceiling.
    PER_STATE_SEED_BASE = 200_000

    def _one_state(idx_state):
        idx, state_id = idx_state
        X_uni = _univariate_X(state_id)
        n_in_state = int((state_arr == state_id).sum())

        out = {
            "n_epochs_in_state": n_in_state,
            "binary_auc_roc": None,
            "binary_auc_pr": None,
            "continuous_r_squared": None,
            "binary_p_value": None,
            "continuous_p_value": None,
            "binary_null_auc_mean": None,
            "continuous_null_r2_mean": None,
        }

        # Observed AUC + R²
        _, observed_auc, observed_auc_pr = _run_logistic_cv(
            X_uni, y_binary, logistic_folds,
        )
        observed_r2 = _run_ridge_cv(X_uni, y_cont, ridge_folds)

        if observed_auc is not None:
            out["binary_auc_roc"] = round(observed_auc, 4)
            out["binary_auc_pr"] = round(observed_auc_pr, 4)
        if observed_r2 is not None:
            out["continuous_r_squared"] = round(observed_r2, 4)

        # Within-run circular-shift null on the brain-state vector. Each
        # state's null is independent: we shift the FULL state vector but
        # only the column for `state_id` matters for the univariate decoder.
        seed_base_state = PER_STATE_SEED_BASE + idx * (2 * n_permutations)
        binary_seeds = np.arange(seed_base_state, seed_base_state + n_permutations)
        cont_seeds = np.arange(
            seed_base_state + n_permutations,
            seed_base_state + 2 * n_permutations,
        )

        null_aucs = []
        if observed_auc is not None:
            for seed in binary_seeds:
                states_perm = _shared_circular_shift(
                    state_arr, run_boundaries, seed=int(seed), min_shift=1,
                )
                col_p = (states_perm == state_id).astype(float).reshape(-1, 1)
                X_p = col_p if season_dummies is None else np.hstack(
                    [col_p, season_dummies],
                )
                _, auc_p, _ = _run_logistic_cv(X_p, y_binary, logistic_folds)
                if auc_p is not None:
                    null_aucs.append(auc_p)
            if null_aucs:
                out["binary_p_value"] = round(
                    finite_permutation_pvalue(null_aucs, observed_auc), 4,
                )
                out["binary_null_auc_mean"] = round(float(np.mean(null_aucs)), 4)

        null_r2s = []
        if observed_r2 is not None:
            for seed in cont_seeds:
                states_perm = _shared_circular_shift(
                    state_arr, run_boundaries, seed=int(seed), min_shift=1,
                )
                col_p = (states_perm == state_id).astype(float).reshape(-1, 1)
                X_p = col_p if season_dummies is None else np.hstack(
                    [col_p, season_dummies],
                )
                r2_p = _run_ridge_cv(X_p, y_cont, ridge_folds)
                if r2_p is not None:
                    null_r2s.append(r2_p)
            if null_r2s:
                out["continuous_p_value"] = round(
                    finite_permutation_pvalue(null_r2s, observed_r2), 4,
                )
                out["continuous_null_r2_mean"] = round(float(np.mean(null_r2s)), 4)

        return state_id, out

    # Parallelise over states. Each worker fits 2*n_permutations models, so
    # this is the dominant cost of the function. Mirrors the joint decoder's
    # joblib pattern (`prefer="processes"`).
    indexed_states = list(enumerate(states_to_test))
    per_state_results_list = _parallel_map(
        _one_state, indexed_states, n_jobs, "Analysis 2 per-state decoding",
    )

    per_state_results = {}
    binary_pvals = []
    binary_state_ids = []
    cont_pvals = []
    cont_state_ids = []
    for state_id, out in per_state_results_list:
        per_state_results[str(state_id)] = out
        if out["binary_p_value"] is not None:
            binary_pvals.append(out["binary_p_value"])
            binary_state_ids.append(str(state_id))
        if out["continuous_p_value"] is not None:
            cont_pvals.append(out["continuous_p_value"])
            cont_state_ids.append(str(state_id))

    # BH-FDR across states (separate correction for binary and continuous).
    if binary_pvals:
        bin_fdr = benjamini_hochberg(np.array(binary_pvals))
        for sid, q in zip(binary_state_ids, bin_fdr):
            per_state_results[sid]["binary_p_fdr"] = round(float(q), 4)
    if cont_pvals:
        cont_fdr = benjamini_hochberg(np.array(cont_pvals))
        for sid, q in zip(cont_state_ids, cont_fdr):
            per_state_results[sid]["continuous_p_fdr"] = round(float(q), 4)

    payload = {
        "n_eligible_states": len(states_to_test),
        "n_permutations": int(n_permutations),
        "n_epochs": int(len(df)),
        "n_runs": int(len(unique_runs)),
        "speech_base_rate": round(base_rate, 3),
        "per_state": per_state_results,
    }

    out_path = os.path.join(out_dir, "analysis_2_decoding_per_state.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    n_sig_binary = sum(
        1 for v in per_state_results.values()
        if v.get("binary_p_fdr") is not None and v["binary_p_fdr"] < 0.05
    )
    logger.info(
        "Analysis 2 [per-state] complete: %d states tested, %d binary q<0.05",
        len(states_to_test), n_sig_binary,
    )


# ── Analysis 3: Per-state multi-lag content signatures (2026-04-23 redesign) ─


def _plot_state_lag_profiles(
    per_lag_aucs, per_lag_p_fdr, target_states, recurrence_scores, lags,
    features, out_path, title_suffix="",
):
    """For each recurrent state with any significant feature, plot per-feature
    AUC-vs-lag curves, highlighting the expected HRF window (lag 3-4)."""
    if len(target_states) == 0:
        return

    aucs_tensor = np.stack([per_lag_aucs[lag] for lag in lags], axis=-1)  # (K, F, L)
    fdr_tensor = np.stack([per_lag_p_fdr[lag] for lag in lags], axis=-1)

    # Select top states by max |AUC - 0.5| at q_matrix < 0.05 in any lag.
    signif = np.any(np.isfinite(fdr_tensor) & (fdr_tensor < 0.05), axis=(1, 2))
    if signif.any():
        state_scores = np.nanmax(np.abs(aucs_tensor - 0.5), axis=(1, 2))
        state_scores[~signif] = -np.inf
        order = np.argsort(-state_scores)[:6]
        order = [i for i in order if signif[i]]
    else:
        # No significant state — plot top 4 most recurrent as a diagnostic.
        rec_vec = np.array([float(recurrence_scores[int(s)]) for s in target_states])
        order = list(np.argsort(-rec_vec)[:4])

    if not order:
        return

    ncols = min(3, len(order))
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows),
                             squeeze=False)
    for panel_idx, si in enumerate(order):
        ax = axes[panel_idx // ncols][panel_idx % ncols]
        state_id = int(target_states[si])
        rec = float(recurrence_scores[state_id])
        # Pick top-5 features by max |AUC - 0.5| across lags for this state.
        feat_scores = np.nanmax(np.abs(aucs_tensor[si] - 0.5), axis=1)
        top_feats = np.argsort(-feat_scores)[:5]
        for fi in top_feats:
            feat_name = features[fi]
            vals = aucs_tensor[si, fi, :]
            ax.plot(lags, vals, marker="o", label=feat_name, alpha=0.85)
        ax.axvspan(3, 4, alpha=0.1, color="gray", zorder=0)
        ax.axhline(0.5, color="black", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Lag (TRs)")
        ax.set_ylabel("AUC")
        ax.set_title(f"S{state_id} (r={rec:.2f})", fontsize=10)
        ax.legend(fontsize=6, loc="best")
    # Hide unused axes
    for k in range(len(order), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(f"Per-state content signatures × lag{title_suffix}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def analysis_3_state_content_multilag(
    decoded_states, content_features, recurrence_scores, eligible_states,
    out_dir, n_permutations=500, n_jobs=1, control_mode="raw",
):
    """Per-state × per-feature × per-lag content-signature profiling.

    Extends A1's per-state AUC testing across lags τ ∈ {0..6}. Same two-sample
    Mann-Whitney AUC, same within-run circular-shift null, same two-layer
    BH-FDR — computed independently at each lag. Peak lag per (state, feature)
    is the lag maximizing ``|AUC − 0.5|``.

    Output files (per subject, per mode):
      - ``analysis_3_state_multilag{suffix}.json`` — nested per_lag → states → feature_*
      - ``state_lag_profiles{suffix}.pdf`` — per-state AUC-vs-lag curves
      - ``state_content_profiles_multilag{suffix}.csv`` — long-form (state, feature, lag)

    See the design notes §3.5 for spec.
    """
    suffix = CONTROL_MODE_SUFFIX.get(control_mode, "")
    label = f" [{control_mode}]" if control_mode != "raw" else ""
    logger.info("Analysis 3%s: Per-state multi-lag content signatures", label)

    eligible_set = set(int(s) for s in eligible_states)
    target_states = sorted(eligible_set)
    lags = list(range(0, 7))

    if not target_states:
        logger.warning("Analysis 3%s: no eligible states — skipping", label)
        return

    run_ids_order = sorted(decoded_states.keys())
    concat_states, run_boundaries = _build_run_boundaries_ordered(
        run_ids_order, decoded_states,
    )

    # Observed per-lag AUC tensors.
    observed = _observed_per_state_signature(
        decoded_states, content_features, eligible_set, recurrence_scores,
        target_states=target_states, lags=lags, control_mode=control_mode,
    )
    total_epochs = sum(v["n_epochs_total"] for v in observed.values())
    if total_epochs == 0:
        logger.warning("Analysis 3%s: 0 eligible epochs across all lags — skipping",
                       label)
        return

    # Precompute null TR sequences once (mode-scoped), reused across all lags.
    mode_idx = list(CONTROL_MODE_SUFFIX.keys()).index(control_mode)
    seed_base = PER_STATE_SEED_BASE_A3 + mode_idx * 1_000_000
    logger.info(
        "Analysis 3%s: %d permutations × %d lags × %d states × %d features",
        label, n_permutations, len(lags), len(target_states),
        len(CONTENT_FEATURE_COLUMNS),
    )

    null_rows = Parallel(
        n_jobs=resolve_n_jobs(n_jobs), prefer="threads", verbose=0,
    )(
        delayed(_null_per_state_signature_one)(
            _shared_circular_shift(concat_states, run_boundaries, seed_base + i),
            run_ids_order, run_boundaries, content_features, recurrence_scores,
            eligible_set, target_states, lags, control_mode,
        )
        for i in range(n_permutations)
    )
    null_by_lag = _aggregate_null_aucs(
        null_rows, lags, len(target_states), len(CONTENT_FEATURE_COLUMNS),
    )

    # Per-lag p-values + two-layer FDR.
    per_lag_p_perm = {}
    per_lag_valid_frac = {}
    per_lag_p_fdr_ps = {}
    per_lag_p_fdr_mx = {}
    for lag in lags:
        p_perm, vf = _per_state_signature_pvalues(
            observed[lag]["aucs"], null_by_lag[lag],
        )
        ps, mx = _apply_two_layer_fdr(p_perm)
        per_lag_p_perm[lag] = p_perm
        per_lag_valid_frac[lag] = vf
        per_lag_p_fdr_ps[lag] = ps
        per_lag_p_fdr_mx[lag] = mx

    # JSON: per_lag.<lag>.states.<state>.feature_*
    per_lag_out = {}
    for lag in lags:
        states_dict = _pack_lag_signature_dict(
            observed[lag], target_states, recurrence_scores,
            per_lag_p_perm[lag], per_lag_p_fdr_ps[lag], per_lag_p_fdr_mx[lag],
            per_lag_valid_frac[lag],
        )
        per_lag_out[str(lag)] = {
            "n_states": len(target_states),
            "n_epochs_total": int(observed[lag]["n_epochs_total"]),
            "states": states_dict,
        }

    # Peak-lag per (state, feature): lag maximizing |AUC - 0.5|.
    n_states = len(target_states)
    n_features = len(CONTENT_FEATURE_COLUMNS)
    aucs_tensor = np.stack([observed[lag]["aucs"] for lag in lags], axis=-1)
    dist_tensor = np.abs(aucs_tensor - 0.5)
    peak_lag_out = {}
    for si, state_id in enumerate(target_states):
        per_feat = {}
        for fi, feat in enumerate(CONTENT_FEATURE_COLUMNS):
            col = dist_tensor[si, fi, :]
            if not np.any(np.isfinite(col)):
                per_feat[feat] = {"lag": None, "auc": None, "signed_distance": None}
                continue
            best_lag_idx = int(np.nanargmax(col))
            best_lag = int(lags[best_lag_idx])
            best_auc = float(aucs_tensor[si, fi, best_lag_idx])
            per_feat[feat] = {
                "lag": best_lag,
                "auc": round(best_auc, 6),
                "signed_distance": round(best_auc - 0.5, 6),
            }
        peak_lag_out[str(int(state_id))] = per_feat

    # Cross-subject hemodynamic sanity check (§3.5 Q_E): distribution of peak
    # lags. We write a per-feature summary; cross-subject aggregation happens
    # in findings.
    peak_lag_hist = {feat: {str(l): 0 for l in lags} for feat in CONTENT_FEATURE_COLUMNS}
    for feats in peak_lag_out.values():
        for feat, entry in feats.items():
            if entry["lag"] is not None:
                peak_lag_hist[feat][str(int(entry["lag"]))] += 1

    payload = {
        "control_mode": control_mode,
        "n_states": int(n_states),
        "n_features": int(n_features),
        "n_permutations": int(n_permutations),
        "effect_size_metric": "auc_mann_whitney",
        "fdr_strategy": "two_layer_bh",
        "features": list(CONTENT_FEATURE_COLUMNS),
        "lags": list(lags),
        "per_lag": per_lag_out,
        "peak_lag_per_state_feature": peak_lag_out,
        "peak_lag_histogram_per_feature": peak_lag_hist,
        "cross_subject_defer": True,
    }
    out_json = os.path.join(out_dir, f"analysis_3_state_multilag{suffix}.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    # Long-form CSV: one row per (state, feature, lag).
    csv_rows = []
    for lag in lags:
        csv_rows.extend(
            _build_csv_rows(
                target_states, recurrence_scores, observed[lag],
                per_lag_p_perm[lag], per_lag_p_fdr_ps[lag], per_lag_p_fdr_mx[lag],
                per_lag_valid_frac[lag], control_mode, n_permutations, lag=lag,
            )
        )
    pd.DataFrame(csv_rows).to_csv(
        os.path.join(out_dir, f"state_content_profiles_multilag{suffix}.csv"),
        index=False,
    )

    # Per-state lag profile plots.
    _plot_state_lag_profiles(
        {lag: observed[lag]["aucs"] for lag in lags},
        per_lag_p_fdr_mx, target_states, recurrence_scores, lags,
        list(CONTENT_FEATURE_COLUMNS),
        os.path.join(out_dir, f"state_lag_profiles{suffix}.pdf"),
        title_suffix=f" [{control_mode}]" if control_mode != "raw" else "",
    )

    # Sparse-null safeguard warnings.
    sparse = 0
    for lag in lags:
        sparse += int(np.sum(per_lag_valid_frac[lag] < 0.5))
    if sparse > 0:
        logger.warning(
            "Analysis 3%s: %d (lag, state, feature) cells have "
            "valid_perm_fraction < 0.5", label, sparse,
        )

    n_sig_any = int(
        sum(
            np.sum(np.isfinite(per_lag_p_fdr_mx[lag]) & (per_lag_p_fdr_mx[lag] < 0.05))
            for lag in lags
        )
    )
    logger.info(
        "Analysis 3%s complete: %d (lag, state, feature) triples at q_matrix<0.05",
        label, n_sig_any,
    )


# ── Analysis 4: Transition-triggered averages ─────────────────────────────────


def analysis_4_tta(
    decoded_states, content_features, recurrence_scores, eligible_states,
    out_dir, n_permutations, n_jobs=1,
):
    """Content features around state transitions — all active states."""
    logger.info("Analysis 4: Transition-triggered averages")

    window = 10  # TRs before and after
    eligible_set = set(eligible_states)

    tta_into = {s: [] for s in eligible_states}
    tta_outof = {s: [] for s in eligible_states}

    for run_id, state_seq in decoded_states.items():
        if run_id not in content_features:
            continue
        feats = content_features[run_id]
        n_trs = min(len(state_seq), len(feats))
        state_seq = np.asarray(state_seq[:n_trs])

        transitions = np.flatnonzero(state_seq[1:] != state_seq[:-1]) + 1
        for t in transitions:
            if t - window < 0 or t + window >= n_trs:
                continue
            snippet = feats[t - window: t + window]
            state_after = int(state_seq[t])
            state_before = int(state_seq[t - 1])

            if state_after in eligible_set:
                tta_into[state_after].append(snippet)
            if state_before in eligible_set:
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
            arr = np.array(snippets)  # (n_trans, 2*window, 9)
            mean_tta = np.nanmean(arr, axis=0)
            for t_idx in range(2 * window):
                row = {
                    "state": state,
                    "recurrence_score": float(recurrence_scores[state]),
                    "direction": label, "relative_tr": t_idx - window,
                    "n_transitions": len(snippets),
                }
                for i, feat in enumerate(CONTENT_FEATURE_COLUMNS):
                    row[feat] = float(mean_tta[t_idx, i])
                tta_results.append(row)

    if tta_results:
        df = pd.DataFrame(tta_results)
        df.to_csv(os.path.join(out_dir, "analysis_4_ttas.csv"), index=False)

        # Null distribution: circular shift, compute post-pre for ALL 9 features
        null_tta_stats = {feat: [] for feat in CONTENT_FEATURE_COLUMNS}

        def _one_tta_permutation(seed):
            rng = np.random.default_rng(seed)
            null_snippets = []
            for run_id, state_seq in decoded_states.items():
                if run_id not in content_features:
                    continue
                feats = content_features[run_id]
                n_trs = min(len(state_seq), len(feats))
                if n_trs <= 2 * window:
                    continue
                shift = rng.integers(window, n_trs - window)
                shifted_seq = np.roll(np.asarray(state_seq[:n_trs]), shift)
                transitions = np.flatnonzero(shifted_seq[1:] != shifted_seq[:-1]) + 1
                for t in transitions:
                    if t - window < 0 or t + window >= n_trs:
                        continue
                    state_after = int(shifted_seq[t])
                    if state_after in eligible_set:
                        null_snippets.append(feats[t - window: t + window])
            if len(null_snippets) >= 5:
                arr = np.array(null_snippets)
                perm_result = {}
                for feat_idx, feat in enumerate(CONTENT_FEATURE_COLUMNS):
                    pre = np.nanmean(arr[:, :window, feat_idx])
                    post = np.nanmean(arr[:, window:, feat_idx])
                    perm_result[feat] = float(post - pre)
                return perm_result
            return {}

        tta_perm_seeds = np.arange(20_000, 20_000 + min(n_permutations, 200))
        tta_perm_results = _parallel_map(
            _one_tta_permutation, tta_perm_seeds, n_jobs, "Analysis 4 permutation",
        )
        for perm_result in tta_perm_results:
            for feat, value in perm_result.items():
                null_tta_stats[feat].append(value)

        # Compute observed post-pre and permutation p-values
        tta_perm_results = {}
        for feat_idx, feat in enumerate(CONTENT_FEATURE_COLUMNS):
            into_df = df[df["direction"] == "into"]
            if len(into_df) == 0:
                continue
            pre_vals = into_df[into_df["relative_tr"] < 0].groupby("state")[feat].mean()
            post_vals = into_df[into_df["relative_tr"] >= 0].groupby("state")[feat].mean()
            common = pre_vals.index.intersection(post_vals.index)
            if len(common) == 0:
                continue
            obs_change = float((post_vals.loc[common] - pre_vals.loc[common]).mean())
            null_arr = np.array(null_tta_stats.get(feat, []))
            tta_perm_results[feat] = {
                "observed_change": round(obs_change, 4),
                "p_perm": round(
                    finite_permutation_pvalue(null_arr, obs_change, absolute=True), 4,
                ) if len(null_arr) > 0 else np.nan,
                "n_permutations": len(null_arr),
            }

        with open(os.path.join(out_dir, "analysis_4_null_stats.json"), "w") as f:
            json.dump(tta_perm_results, f, indent=2)

        # Plot TTA for speech_presence (top-5 states by transition count)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, direction in zip(axes, ["into", "outof"]):
            sub = df[df["direction"] == direction]
            top_states = (sub.groupby("state")["n_transitions"].first()
                          .nlargest(5).index)
            for state in top_states:
                s = sub[sub["state"] == state]
                rec = float(recurrence_scores[state])
                ax.plot(s["relative_tr"], s["speech_presence"],
                        label=f"S{state} (r={rec:.2f})")
            ax.axvline(0, color="gray", linestyle="--")
            ax.set_xlabel("TR relative to transition")
            ax.set_ylabel("Speech presence")
            ax.set_title(f"Transitions {direction} state")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "tta_content.pdf"), dpi=150)
        plt.close(fig)

    logger.info("Analysis 4 complete: %d TTA records", len(tta_results))


# ── Analysis 5: Cross-episode consistency ─────────────────────────────────────


def analysis_5_consistency(
    decoded_states, content_features, recurrence_scores, eligible_states, out_dir, n_jobs=1,
):
    """Cross-episode content consistency.

    Uses SD (not CV) to avoid z-scored mean-near-zero issue.
    """
    logger.info("Analysis 5: Cross-episode consistency")

    eligible_set = set(eligible_states)

    # Per-run, per-state mean content features
    run_state_means = []
    for run_id, state_seq in decoded_states.items():
        if run_id not in content_features:
            continue
        feats = content_features[run_id]
        n_trs = min(len(state_seq), len(feats))
        state_seq = np.asarray(state_seq[:n_trs])
        try:
            season = _get_season(run_id)
        except ValueError:
            season = 0

        for state in np.unique(state_seq):
            state = int(state)
            if state not in eligible_set:
                continue
            mask = state_seq == state
            if np.sum(mask) < 2:
                continue
            mean_feats = np.nanmean(feats[mask], axis=0)
            run_state_means.append({
                "run_id": run_id, "season": season,
                "state": state,
                "recurrence_score": float(recurrence_scores[state]),
                **{CONTENT_FEATURE_COLUMNS[i]: mean_feats[i] for i in range(len(CONTENT_FEATURE_COLUMNS))},
            })

    if not run_state_means:
        logger.warning("No data for consistency analysis")
        return

    df = pd.DataFrame(run_state_means)
    df.to_csv(os.path.join(out_dir, "analysis_5_consistency.csv"), index=False)

    # Per-state SD across episodes
    consistency = []
    # Skip binary features (speech_presence, speaker_change, scene_boundary, characters)
    binary_idx = {0, 3, 6, 9, 10, 11, 12, 13, 14}
    continuous_feats = [f for i, f in enumerate(CONTENT_FEATURE_COLUMNS)
                        if i not in binary_idx]
    for state in sorted(df["state"].unique()):
        sub = df[df["state"] == state]
        if len(sub) < 5:
            continue
        rec = float(recurrence_scores[state])
        for feat in continuous_feats:
            vals = sub[feat].dropna()
            if len(vals) < 5:
                continue
            consistency.append({
                "state": int(state), "recurrence_score": rec, "feature": feat,
                "sd": float(vals.std()),
                "iqr": float(vals.quantile(0.75) - vals.quantile(0.25)),
                "n_episodes": len(vals),
            })

    if consistency:
        cv_df = pd.DataFrame(consistency)
        cv_df.to_csv(os.path.join(out_dir, "analysis_5_consistency_metrics.csv"), index=False)

        # Split-half reliability (1000 random splits)
        unique_runs = df["run_id"].unique()

        def _one_split_half(seed):
            rng = np.random.default_rng(seed)
            half = rng.choice(unique_runs, size=len(unique_runs) // 2, replace=False)
            half_set = set(half)
            h1 = df[df["run_id"].isin(half_set)]
            h2 = df[~df["run_id"].isin(half_set)]

            p1 = h1.groupby("state")[continuous_feats].mean()
            p2 = h2.groupby("state")[continuous_feats].mean()
            common = p1.index.intersection(p2.index)
            if len(common) < 3:
                return None
            v1 = p1.loc[common].values.ravel()
            v2 = p2.loc[common].values.ravel()
            valid = np.isfinite(v1) & np.isfinite(v2)
            if np.sum(valid) < 5:
                return None
            r, _ = stats.pearsonr(v1[valid], v2[valid])
            return float(r)

        split_seeds = np.arange(30_000, 31_000)
        split_half_corrs = [
            r for r in _parallel_map(
                _one_split_half, split_seeds, n_jobs, "Analysis 5 split-half",
            )
            if r is not None
        ]

        reliability_result = {}
        if split_half_corrs:
            mean_r = float(np.mean(split_half_corrs))
            # Spearman-Brown correction: 2r / (1 + r)
            if mean_r > -1:
                sb_r = 2 * mean_r / (1 + mean_r)
            else:
                sb_r = -1.0
            reliability_result = {
                "split_half_r_mean": round(mean_r, 4),
                "split_half_r_std": round(float(np.std(split_half_corrs)), 4),
                "spearman_brown_corrected": round(sb_r, 4),
                "n_splits": len(split_half_corrs),
            }

        with open(os.path.join(out_dir, "analysis_5_reliability.json"), "w") as f:
            json.dump(reliability_result, f, indent=2)

    # ── Occupancy-content correlation with FDR ────────────────────────
    occ_content_results = {}
    all_p_values = []
    all_keys = []

    for state in sorted(df["state"].unique()):
        state_df = df[df["state"] == state]
        if len(state_df) < 5:
            continue
        run_occ = {}
        for run_id, seq in decoded_states.items():
            if run_id not in content_features:
                continue
            seq = np.asarray(seq)
            run_occ[run_id] = float(np.mean(seq == state))

        for feat in CONTENT_FEATURE_COLUMNS:
            fo_vals, feat_vals = [], []
            for _, row in state_df.iterrows():
                rid = row["run_id"]
                if rid in run_occ and np.isfinite(row[feat]):
                    fo_vals.append(run_occ[rid])
                    feat_vals.append(row[feat])
            if len(fo_vals) < 5:
                continue
            rho, p = stats.spearmanr(fo_vals, feat_vals)
            occ_content_results.setdefault(int(state), {})[feat] = {
                "rho": round(float(rho), 4),
                "p": round(float(p), 4),
                "n_runs": len(fo_vals),
            }
            all_p_values.append(p)
            all_keys.append((int(state), feat))

    # Apply FDR across all state x feature tests
    if all_p_values:
        fdr_p = benjamini_hochberg(np.array(all_p_values))
        for i, (state, feat) in enumerate(all_keys):
            occ_content_results[state][feat]["p_fdr"] = round(float(fdr_p[i]), 4)

    with open(os.path.join(out_dir, "analysis_5_occupancy_content.json"), "w") as f:
        json.dump(occ_content_results, f, indent=2)

    logger.info("Analysis 5 complete: %d consistency records, %d states with occ-content corr",
                len(consistency), len(occ_content_results))


# ── Analysis 6: Content selectivity ───────────────────────────────────────────


def analysis_6_selectivity(
    decoded_states, content_features, recurrence_scores, eligible_states, out_dir,
    n_jobs=1, control_mode="raw",
):
    """Content selectivity: IQR/variance for continuous, Bernoulli entropy for binary.

    Spearman: recurrence_score vs selectivity metric per feature.

    When ``control_mode == "partial"``, the C4 negative control applies:
    epoch-level feature values are residualized against an epoch-center
    run-position cubic polynomial (computed once across all epochs of all
    states — a single global design) before selectivity is evaluated. This
    removes variation attributable to within-run drift before measuring
    per-state content selectivity. Other control_mode values reuse raw logic
    (the mask/state-set substitution is applied in the caller).
    """
    suffix = CONTROL_MODE_SUFFIX.get(control_mode, "")
    label = f" [{control_mode}]" if control_mode != "raw" else ""
    logger.info("Analysis 6%s: Content selectivity", label)

    block_records = extract_state_block_records(
        decoded_states, recurrence_scores, include_states=set(eligible_states),
    )

    # Collect epoch-level feature values per state (paired with their records
    # so we can residualize globally in partial mode).
    state_epoch_feats = {s: [] for s in eligible_states}
    all_records = []
    all_means = []
    for rec in block_records:
        run_id = rec["run_id"]
        if run_id not in content_features:
            continue
        feats = content_features[run_id]
        start, end = rec["start_tr"], rec["end_tr"]
        if end <= start or start >= len(feats) or (end - start) < 2:
            continue
        epoch_mean = np.nanmean(feats[start:end], axis=0)
        all_records.append(rec)
        all_means.append(epoch_mean)

    if control_mode == "partial" and all_means:
        mat = np.array(all_means, dtype=np.float64)
        D = build_epoch_run_position_design(all_records, degree=3)
        mat = partial_effect_residualize(mat, D)
        for rec, row in zip(all_records, mat):
            state_epoch_feats[rec["state"]].append(row)
    else:
        for rec, row in zip(all_records, all_means):
            state_epoch_feats[rec["state"]].append(row)

    # Binary: speech_presence(0), speaker_change(3), scene_boundary(6),
    # character presence(9-14)
    binary_cols = {0, 3, 6, 9, 10, 11, 12, 13, 14}

    def _one_state_selectivity(state):
        epochs = state_epoch_feats[state]
        if len(epochs) < 10:
            return []
        arr = np.array(epochs)  # (n_epochs, 9)
        rec = float(recurrence_scores[state])
        state_rows = []

        for feat_idx, feat in enumerate(CONTENT_FEATURE_COLUMNS):
            vals = arr[:, feat_idx]
            valid = vals[np.isfinite(vals)]
            if len(valid) < 5:
                continue

            if feat_idx in binary_cols:
                # Bernoulli entropy: H(p) = -p*log(p) - (1-p)*log(1-p)
                p = np.clip(np.mean(valid), 1e-10, 1 - 1e-10)
                entropy = float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))
                selectivity_score = 1.0 - entropy
                state_rows.append({
                    "state": int(state), "recurrence_score": rec, "feature": feat,
                    "metric": "bernoulli_selectivity", "value": round(selectivity_score, 4),
                    "raw_entropy": round(entropy, 4),
                    "p_positive": round(float(p), 4),
                    "n_epochs": len(valid),
                })
            else:
                iqr = float(np.percentile(valid, 75) - np.percentile(valid, 25))
                var = float(np.var(valid))
                state_rows.append({
                    "state": int(state), "recurrence_score": rec, "feature": feat,
                    "metric": "iqr", "value": round(iqr, 4),
                    "variance": round(var, 4),
                    "n_epochs": len(valid),
                })
        return state_rows

    selectivity = []
    for state_rows in _parallel_map(
        _one_state_selectivity, eligible_states, n_jobs, "Analysis 6 state selectivity",
    ):
        selectivity.extend(state_rows)

    if selectivity:
        sel_df = pd.DataFrame(selectivity)
        sel_df.to_csv(
            os.path.join(out_dir, f"analysis_6_selectivity{suffix}.csv"), index=False,
        )

        # Spearman: recurrence_score vs selectivity per feature
        recurrence_selectivity = {}
        all_p = []
        all_feat_keys = []
        for feat in CONTENT_FEATURE_COLUMNS:
            feat_df = sel_df[sel_df["feature"] == feat]
            if len(feat_df) < 5:
                continue
            rho, p = stats.spearmanr(feat_df["recurrence_score"], feat_df["value"])
            recurrence_selectivity[feat] = {
                "rho": round(float(rho), 4), "p": round(float(p), 4),
                "n_states": len(feat_df),
            }
            all_p.append(p)
            all_feat_keys.append(feat)

        if all_p:
            fdr_p = benjamini_hochberg(np.array(all_p))
            for i, f in enumerate(all_feat_keys):
                recurrence_selectivity[f]["p_fdr"] = round(float(fdr_p[i]), 4)
        recurrence_selectivity["_control_mode"] = control_mode

        with open(os.path.join(out_dir, f"analysis_6_selectivity{suffix}.json"), "w") as f:
            json.dump(recurrence_selectivity, f, indent=2)

    logger.info("Analysis 6%s complete: %d selectivity records", label, len(selectivity))


# ── Analysis 7: Sensory confound control ──────────────────────────────────────


def analysis_7_sensory_control(
    sub_id, parc, recurrence_scores, eligible_states, out_dir,
):
    """Classify states by dominant network to assess sensory confound.

    States dominated by SomMot or Vis activation may show content associations
    driven by low-level auditory/visual processing rather than narrative content.
    """
    logger.info("Analysis 7: Sensory confound control")

    # Load state means in parcel space
    means_base = os.path.join(
        SCRATCH_DIR, "output", "04_combined_hdphmm", parc, sub_id,
        "final",
    ).replace("//", "/")
    try:
        means_path = resolve_stage_file(
            means_base, "state_means_parcel.npy", "state means",
        )
    except FileNotFoundError as e:
        logger.warning("%s — skipping Analysis 7", e)
        return {}

    state_means = np.load(means_path)  # (K, n_parcels)

    # Load parcel labels for network assignment
    try:
        from utils.viz_yabplot import load_parcel_labels
        label_df = load_parcel_labels(parc)
    except Exception as e:
        logger.warning("Could not load parcel labels: %s — skipping Analysis 7", e)
        return {}

    # Build parcel-to-network mapping
    n_parcels = state_means.shape[1]
    parcel_networks = []
    for idx in range(n_parcels):
        row = label_df[label_df["index"] == idx + 1]  # 1-based index
        if len(row) == 0:
            parcel_networks.append("Unknown")
            continue
        label = row.iloc[0]["label"]
        # Try subcortical assignment first
        net = assign_network(label)
        if net is None:
            # Cortical: use network_label column
            net = row.iloc[0].get("network_label", "Unknown")
        parcel_networks.append(net)

    parcel_networks = np.array(parcel_networks)
    sensory_networks = {"Vis", "SomMot"}

    # Per-state network classification
    classifications = []
    for state in eligible_states:
        if state >= len(state_means):
            continue
        mean_vec = state_means[state]
        abs_mean = np.abs(mean_vec)

        # Mean |activation| per network
        net_activation = {}
        for net in NETWORK_ORDER:
            mask = parcel_networks == net
            if np.any(mask):
                net_activation[net] = float(np.mean(abs_mean[mask]))

        if not net_activation:
            continue

        dominant_net = max(net_activation, key=net_activation.get)
        sensory_dominated = dominant_net in sensory_networks

        # Sensory loading: fraction of top-10 parcels in sensory networks
        top10_idx = np.argsort(abs_mean)[-10:]
        n_sensory_top10 = sum(1 for i in top10_idx if parcel_networks[i] in sensory_networks)
        sensory_loading = n_sensory_top10 / 10.0

        classifications.append({
            "state": int(state),
            "recurrence_score": float(recurrence_scores[state]),
            "dominant_network": dominant_net,
            "sensory_dominated": sensory_dominated,
            "sensory_loading": round(sensory_loading, 2),
            "network_activations": {k: round(v, 4) for k, v in net_activation.items()},
        })

    if classifications:
        with open(os.path.join(out_dir, "state_network_classification.json"), "w") as f:
            json.dump(classifications, f, indent=2)

        n_sensory = sum(1 for c in classifications if c["sensory_dominated"])
        n_assoc = len(classifications) - n_sensory
        logger.info(
            "Analysis 7 complete: %d sensory-dominated, %d association-network states",
            n_sensory, n_assoc,
        )

    return {c["state"]: c for c in classifications}


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    sub_id = args.sub_id
    parc = normalize_parcellation_name(args.parcellation)
    n_jobs = resolve_n_jobs(args.n_jobs)
    force = args.force
    requested = set(args.analyses) if args.analyses else {"A1", "A2", "A3", "A4", "A5", "A6", "A7"}

    out_dir = os.path.join(
        SCRATCH_DIR, "output", "08b_content_state_correspondence", parc, sub_id,
    )
    os.makedirs(out_dir, exist_ok=True)

    logger.info(
        "Analyses requested: %s | force=%s",
        " ".join(sorted(requested)), force,
    )

    # -- Checkpoint file definitions per analysis --
    # Each maps analysis label -> list of base filenames (no suffix). For mode-
    # varying analyses (A1/A2/A3/A6) we check per-(analysis, control_mode)
    # using CONTROL_MODE_SUFFIX; A4/A5/A7 are not mode-varied.
    CHECKPOINT_FILES_BASE = {
        "A1": ["analysis_1_state_signatures"],
        "A2": ["analysis_2_decoding"],
        "A3": ["analysis_3_state_multilag"],
        "A4": ["analysis_4_ttas.csv"],
        "A5": ["analysis_5_consistency.csv", "analysis_5_occupancy_content.json"],
        "A6": ["analysis_6_selectivity"],
        "A7": ["state_network_classification.json"],
    }
    MODE_VARIED = {"A1", "A2", "A3", "A6"}
    control_modes = list(args.control_modes)

    def _checkpoint_files(analysis, mode):
        bases = CHECKPOINT_FILES_BASE[analysis]
        suffix = CONTROL_MODE_SUFFIX.get(mode, "")
        out = []
        for base in bases:
            # A4/A5/A7 entries already include their extension.
            if base.endswith((".csv", ".json")):
                out.append(base)
            else:
                out.append(f"{base}{suffix}.json")
        # A2 per-state file is only produced in raw mode (no control_mode).
        if analysis == "A2" and mode == "raw":
            out.append("analysis_2_decoding_per_state.json")
        return out

    # Plan: one task per (analysis, mode) for mode-varied analyses; one task
    # per analysis for the rest. Skip anything already checkpointed unless
    # --force was passed.
    tasks = []  # list of (analysis, mode)
    for a in sorted(requested):
        modes = control_modes if a in MODE_VARIED else ["raw"]
        for mode in modes:
            files = _checkpoint_files(a, mode)
            label = f"{a}[{mode}]" if a in MODE_VARIED else a
            if check_checkpoint(out_dir, files, label, force=force):
                continue
            tasks.append((a, mode))

    if not tasks:
        logger.info("All requested analyses already checkpointed. Nothing to do.")
        return

    logger.info(
        "Will run %d tasks: %s",
        len(tasks),
        ", ".join(f"{a}[{m}]" if a in MODE_VARIED else a for a, m in tasks),
    )

    decoded_states, content_features, content_features_raw, recurrence_scores = load_inputs(
        sub_id, parc,
    )

    # Content-eligibility via the project-wide 05e_a4 convention (§6 of
    # 2026-04-09_08_transformer_refactor_design.md). Falls back to the 05a
    # sub-HRF filter with a prominent warning if state_flags.csv is missing.
    eligibility = get_content_eligibility(sub_id, parc, vt=args.vt)
    eligible_states = eligibility["content_eligible"]
    run_onset_anchored_states = eligibility["run_onset_anchored"]

    n_active = sum(1 for r in recurrence_scores if r > 0)
    logger.info(
        "States: %d total, %d active, %d content_eligible, %d run_onset_anchored "
        "(eligibility_source=%s)",
        len(recurrence_scores), n_active, len(eligible_states),
        len(run_onset_anchored_states), eligibility["eligibility_source"],
    )
    logger.info("Using n_jobs=%d for joblib-parallelized analyses", n_jobs)

    # Persist the eligibility source for downstream consumers.
    with open(os.path.join(out_dir, "eligibility_source.json"), "w") as f:
        json.dump({
            "eligibility_source": eligibility["eligibility_source"],
            "n_content_eligible": len(eligible_states),
            "n_run_onset_anchored": len(run_onset_anchored_states),
            "content_eligible_ids": eligible_states,
            "run_onset_anchored_ids": run_onset_anchored_states,
        }, f, indent=2)

    # Load physio if requested
    physio_features = None
    if args.include_physio:
        physio_features = load_physio_features(sub_id, decoded_states)
        if not physio_features:
            logger.warning("--include_physio set but no physio data found; continuing without")
            physio_features = None

    # -- Per-task data prep helper (mask/state-set swap by control_mode) --

    def _prepare_for_mode(mode):
        """Return (decoded, content, content_raw, states) tuple for this mode."""
        if mode == "mask33a":
            return (
                mask_a_run_opening(decoded_states, MASK33A_TR),
                mask_a_run_opening(content_features, MASK33A_TR),
                mask_a_run_opening(content_features_raw, MASK33A_TR),
                eligible_states,
            )
        if mode == "run_onset_anchored":
            return (
                decoded_states, content_features, content_features_raw,
                run_onset_anchored_states,
            )
        # raw / partial both use the unmasked decoded_states and content_eligible
        return (
            decoded_states, content_features, content_features_raw, eligible_states,
        )

    # -- Run selected tasks with checkpoint guards --

    for a, mode in tasks:
        if a in MODE_VARIED and mode == "run_onset_anchored" and not run_onset_anchored_states:
            logger.warning(
                "Skipping %s[%s]: no run_onset_anchored states available", a, mode,
            )
            continue

        ds, cf, cfr, states = _prepare_for_mode(mode)

        if a == "A1":
            analysis_1_state_content_signatures(
                ds, cf, recurrence_scores, states, out_dir,
                n_permutations=args.n_permutations_per_state,
                n_jobs=n_jobs, control_mode=mode,
            )
        elif a == "A2":
            analysis_2_decoding(
                ds, cfr, recurrence_scores, states, out_dir, args.n_permutations,
                physio_features=physio_features, n_jobs=n_jobs, control_mode=mode,
            )
            if mode == "raw":
                # Per-state univariate decoder feeds 08g D5 with per-state AUCs
                # that are commensurate with 08d D2 per-state AUCs. Only raw.
                analysis_2_decoding_per_state(
                    ds, cfr, recurrence_scores, states, out_dir,
                    args.n_permutations, n_jobs=n_jobs,
                )
        elif a == "A3":
            analysis_3_state_content_multilag(
                ds, cfr, recurrence_scores, states, out_dir,
                n_permutations=args.n_permutations_per_state,
                n_jobs=n_jobs, control_mode=mode,
            )
        elif a == "A4":
            analysis_4_tta(
                decoded_states, content_features, recurrence_scores, eligible_states,
                out_dir, args.n_permutations, n_jobs=n_jobs,
            )
        elif a == "A5":
            analysis_5_consistency(
                decoded_states, content_features, recurrence_scores, eligible_states,
                out_dir, n_jobs=n_jobs,
            )
        elif a == "A6":
            analysis_6_selectivity(
                ds, cfr, recurrence_scores, states, out_dir,
                n_jobs=n_jobs, control_mode=mode,
            )
        elif a == "A7":
            analysis_7_sensory_control(
                sub_id, parc, recurrence_scores, eligible_states, out_dir,
            )

    logger.info(
        "Completed %d tasks across analyses. Output: %s",
        len(tasks), out_dir,
    )


if __name__ == "__main__":
    main()
