#!/usr/bin/env python3
"""
05e_temporal_trend_a3.py - Within-session FO trends via LME.

Tests whether fractional occupancy (FO) for each brain state trends
systematically across runs within scanning sessions, using a linear
mixed-effects model with random intercepts per session.

A negative slope is consistent with within-session habituation but could
also reflect scanner drift, fatigue, or content effects. Interpret as
"within-session FO trend" rather than proven habituation.

Model (per state k):
    FO_k ~ run_index_centered + (1 | session)
    - Fixed slope β₁ = population-level within-session FO trend rate
    - Random intercept = session-specific baseline FO
    - No random slope (unidentifiable with n=3-6 runs per session)

Inference:
    Hybrid approach - LME slope β₁ as test statistic, permutation test for
    p-values (shuffling run indices within sessions). FO is zero-inflated for
    many states, violating Gaussian assumptions for Wald tests; permutation
    gives exact Type I error control.

    Speed optimization: fit LME once to extract variance components, then use
    fixed VCs for GLS slope computation under each permutation (no iterative
    optimization per permutation).

Known limitation:
    Shuffling run indices within sessions breaks temporal autocorrelation
    structure (lag-1 r ≈ 0.09-0.22 across subjects). This is standard
    practice - the null hypothesis is "no monotonic trend in run position."

Prerequisites:
    - 04_combined_hdphmm.py (mode: select) completed for this subject
    - 05a_recurrence_analysis.py completed for this subject
    - 00_get_scan output with acquisition times CSV

Outputs (saved to {SCRATCH_DIR}/output/05e_temporal_trend_a3/{parc}/{sub_id}/[vt{VT}/]):
    - habituation_results.json        full LME results + convergence stats
    - habituation_metrics.csv          per-state summary table
    - habituation_per_state/           per-state FO-vs-run PNGs
    - habituation_summary.png/pdf      multi-panel overview

See also:
    05e_temporal_trend_a1.py - cross-episode temporal trends (Scales 1, 2, 3)
    05e_temporal_trend_a2.py - within-run temporal position

Design doc: the design notes
"""

import argparse
import csv
import json
import logging
import os
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.regression.mixed_linear_model import MixedLM

sys.path.insert(0, str(Path(__file__).parent))
from utils.stats import benjamini_hochberg, fdr_with_nan as _fdr_with_nan
from utils.plot_style import (
    recurrence_color, make_recurrence_colorbar, apply_publication_style,
    NETWORK_ORDER, NETWORK_COLORS, load_parcel_networks, compute_dominant_networks,
)
from utils.common import normalize_parcellation_name
from utils.state_blocks import load_eligible_states

from dotenv import load_dotenv

load_dotenv()
SCRATCH_DIR = os.getenv('SCRATCH_DIR')
if SCRATCH_DIR is None:
    raise ValueError("SCRATCH_DIR must be set in the .env file")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

apply_publication_style()

# Minimum runs within a session to include in analysis
MIN_SESSION_SIZE = 3


# =============================================================================
# Data loading
# =============================================================================

def load_acquisition_times(sub_id):
    """Load run-level relative acquisition times from 00_get_scan output.

    Returns:
        acq_dict: dict run_id -> float (rel_acq_time in days), or None.
        session_map: dict run_id -> str (BIDS session_id), or None.
    """
    csv_path = os.path.join(
        SCRATCH_DIR, 'output', '00_get_scan', sub_id,
        f'{sub_id}_run_acquisition_times.csv'
    )
    if not os.path.exists(csv_path):
        logger.warning("Acquisition time CSV not found: %s", csv_path)
        return None, None

    acq_dict = {}
    session_map = {}
    has_session_col = False
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        has_session_col = 'session_id' in reader.fieldnames
        for row in reader:
            acq_dict[row['run_id']] = float(row['rel_acq_time'])
            if has_session_col:
                session_map[row['run_id']] = row['session_id']

    logger.info("Loaded acquisition times for %d runs from %s",
                len(acq_dict), csv_path)
    if has_session_col:
        n_unique = len(set(session_map.values()))
        logger.info("  BIDS session_id available: %d unique sessions", n_unique)
    else:
        logger.info("  No session_id column; will fall back to gap-based clustering")
        session_map = None
    return acq_dict, session_map


# =============================================================================
# Session assignment
# =============================================================================

def _cluster_scanning_sessions(acq_dict, gap_days=7):
    """Cluster runs into scanning sessions by acquisition time gaps.

    Fallback when BIDS session_id is unavailable.
    Runs separated by > gap_days are placed in different sessions.

    Returns:
        session_map: dict run_id -> int (session index, 0-based)
        n_sessions: int
    """
    if not acq_dict:
        return {}, 0
    sorted_runs = sorted(acq_dict.items(), key=lambda x: x[1])
    session_map = {}
    session_idx = 0
    prev_time = sorted_runs[0][1]
    for run_id, t in sorted_runs:
        if t - prev_time > gap_days:
            session_idx += 1
        session_map[run_id] = session_idx
        prev_time = t
    return session_map, session_idx + 1


def assign_sessions(run_ids, acq_dict, session_map, session_gap_days=7):
    """Assign runs to integer session labels.

    Uses BIDS session_id when available; falls back to gap-based clustering.

    Returns:
        session_labels: np.ndarray (n_runs,) of int session indices (-1 = unknown)
        has_sessions: bool
        session_source: str
    """
    if session_map is not None:
        raw_labels = [session_map.get(r, '__unknown__') for r in run_ids]
        unique_bids = sorted(set(raw_labels) - {'__unknown__'})
        bids_to_int = {s: i for i, s in enumerate(unique_bids)}
        labels = np.array([bids_to_int.get(lbl, -1) for lbl in raw_labels])
        source = 'bids_session_id'
        logger.info("Using BIDS session_id (%d unique sessions)", len(unique_bids))
        return labels, True, source

    if acq_dict is not None:
        gap_map, _ = _cluster_scanning_sessions(acq_dict, gap_days=session_gap_days)
        labels = np.array([gap_map.get(r, -1) for r in run_ids])
        source = f'gap_clustering_{session_gap_days}d'
        logger.info("Using gap-based clustering (gap_days=%d)", session_gap_days)
        return labels, True, source

    logger.warning("No session info available")
    return np.full(len(run_ids), -1, dtype=int), False, 'none'


# =============================================================================
# Statistical helpers
# =============================================================================

def _safe_spearman(x, y):
    """Spearman rho with NaN guard."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) < 1e-10 or np.std(y[valid]) < 1e-10:
        return np.nan, np.nan
    rho, p = spearmanr(x[valid], y[valid])
    return float(rho), float(p)


def _permutation_p_twosided(observed, null_dist):
    """Two-sided permutation p-value with Phipson & Smyth correction."""
    from utils.stats import permutation_pvalue
    return permutation_pvalue(observed, null_dist, alternative='two-sided')


# =============================================================================
# LME fitting
# =============================================================================

def _fit_ri_lme(fo_vec, run_idx_centered, session_ids):
    """Fit random-intercept LME for one state via statsmodels.

    Model: fo_vec ~ run_idx_centered + (1 | session)

    Args:
        fo_vec: (n_runs,) array of FO values
        run_idx_centered: (n_runs,) array of within-session centered run index
        session_ids: (n_runs,) array of integer session labels

    Returns dict with:
        slope, se, sigma2_session, sigma2_resid, icc, converged, singular
    """
    nan_result = dict(
        slope=np.nan, se=np.nan,
        sigma2_session=np.nan, sigma2_resid=np.nan,
        icc=np.nan, converged=False, singular=False,
    )

    # Check for degenerate data
    if np.std(fo_vec) < 1e-15:
        return nan_result

    df = pd.DataFrame({
        'fo': fo_vec,
        'run_idx': run_idx_centered,
        'session': session_ids,
    })

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = MixedLM.from_formula(
                'fo ~ run_idx', data=df, groups=df['session'],
            )
            result = model.fit(reml=True, method='lbfgs', maxiter=200)
    except Exception as e:
        logger.warning("LME fit failed: %s", e)
        return nan_result

    slope = float(result.fe_params.get('run_idx', np.nan))
    se = float(result.bse.get('run_idx', np.nan))
    sigma2_resid = float(result.scale)

    # Random intercept variance
    cov_re = result.cov_re
    if hasattr(cov_re, 'iloc'):
        sigma2_session = float(cov_re.iloc[0, 0])
    elif hasattr(cov_re, 'values'):
        sigma2_session = float(cov_re.values[0, 0])
    else:
        sigma2_session = float(np.asarray(cov_re).flat[0])

    singular = sigma2_session < 1e-10
    if singular:
        sigma2_session = 0.0

    total_var = sigma2_session + sigma2_resid
    icc = sigma2_session / total_var if total_var > 1e-15 else 0.0

    return dict(
        slope=slope, se=se,
        sigma2_session=sigma2_session, sigma2_resid=sigma2_resid,
        icc=icc, converged=True, singular=singular,
    )


def _gls_slope_fixed_vc(fo_vec, run_idx_centered, session_masks,
                         sigma2_session, sigma2_resid):
    """Compute GLS slope with fixed variance components (for permutation).

    Block-diagonal V = σ²_session * Z Z' + σ²_resid * I
    GLS: β = (X' V⁻¹ X)⁻¹ X' V⁻¹ y
    Slope-only design is equivalent to the full model (with intercept)
    because run_idx is centered within each session.

    Each session block is small (3-6), so invert independently via
    Woodbury identity.

    Args:
        fo_vec: (n_runs,) FO values
        run_idx_centered: (n_runs,) centered run index (may be permuted)
        session_masks: list of boolean masks, one per usable session
        sigma2_session: float, between-session variance
        sigma2_resid: float, residual variance

    Returns: slope (float)
    """
    if sigma2_resid < 1e-15:
        return np.nan

    # Accumulate X'V^{-1}X and X'V^{-1}y across session blocks
    xtvinvx = 0.0
    xtvinvy = 0.0

    for mask in session_masks:
        n_s = int(mask.sum())
        if n_s < MIN_SESSION_SIZE:
            continue

        y_s = fo_vec[mask]
        x_s = run_idx_centered[mask]

        # V_block = sigma2_session * 1 1' + sigma2_resid * I  (n_s × n_s)
        # Use Woodbury identity for efficient inversion:
        # V^{-1} = (1/sigma2_resid) * (I - sigma2_session / (sigma2_resid + n_s * sigma2_session) * 1 1')
        inv_resid = 1.0 / sigma2_resid
        denom = sigma2_resid + n_s * sigma2_session
        c = sigma2_session / denom if denom > 1e-15 else 0.0

        # V^{-1} x = inv_resid * (x - c * sum(x) * 1)
        sum_x = x_s.sum()
        vinv_x = inv_resid * (x_s - c * sum_x)

        # V^{-1} y = inv_resid * (y - c * sum(y) * 1)
        sum_y = y_s.sum()
        vinv_y = inv_resid * (y_s - c * sum_y)

        xtvinvx += x_s @ vinv_x
        xtvinvy += x_s @ vinv_y

    if abs(xtvinvx) < 1e-30:
        return np.nan

    return xtvinvy / xtvinvx


def _session_boundary_reset(fo_matrix, session_labels, unique_sessions):
    """Compute mean FO jump at session boundaries.

    Returns (n_states,) or None.
    """
    reset_deltas = []
    for si in range(1, len(unique_sessions)):
        prev_mask = session_labels == unique_sessions[si - 1]
        curr_mask = session_labels == unique_sessions[si]
        prev_indices = np.where(prev_mask)[0]
        curr_indices = np.where(curr_mask)[0]
        if len(prev_indices) > 0 and len(curr_indices) > 0:
            last_prev = fo_matrix[prev_indices[-1]]
            first_curr = fo_matrix[curr_indices[0]]
            reset_deltas.append(first_curr - last_prev)

    if reset_deltas:
        return np.mean(reset_deltas, axis=0)
    return None


# =============================================================================
# Session detrending
# =============================================================================

def compute_detrended_fo(fo_dict, results, n_states):
    """Compute session-detrended FO by removing within-session linear trends.

    For each state k with a converged LME fit:
        FO_detrended[run, k] = FO[run, k] - β₁_k × run_idx_centered[run]

    For failed fits (NaN slope): keep original FO (β₁ assumed 0).
    For runs in sessions with <MIN_SESSION_SIZE runs: keep original FO.
    Negative values clipped to 0.  No renormalization (unit-sum not preserved).

    Returns
    -------
    detrended_fo_dict : dict[str, np.ndarray]
        Same format as upstream fractional_occupancy.pkl.
    clip_info : dict
        Diagnostic: n_clipped, max_clip_magnitude, pct_clipped.
    """
    run_ids = results.get('run_ids')
    run_idx_centered = results.get('run_idx_centered')
    slopes = results.get('lme_slope')
    fo_matrix = results.get('fo_matrix')

    if run_ids is None or fo_matrix is None or slopes is None:
        logger.warning("Missing data for detrending; returning original FO.")
        return dict(fo_dict), dict(n_clipped=0, max_clip_magnitude=0.0,
                                   pct_clipped=0.0)

    n_runs = fo_matrix.shape[0]
    detrended = fo_matrix.copy()  # (n_runs, n_states)

    # Replace NaN slopes with 0 (no detrending for failed fits)
    safe_slopes = np.where(np.isfinite(slopes), slopes, 0.0)

    for i in range(n_runs):
        rc = run_idx_centered[i] if run_idx_centered is not None else np.nan
        if np.isfinite(rc):
            detrended[i, :] -= safe_slopes * rc
        # else: keep original (run not in a usable session)

    # Clip negatives
    neg_mask = detrended < 0
    n_clipped = int(neg_mask.sum())
    max_clip = float(np.abs(detrended[neg_mask]).max()) if n_clipped > 0 else 0.0
    total_entries = n_runs * n_states
    pct_clipped = 100.0 * n_clipped / total_entries if total_entries > 0 else 0.0

    detrended = np.clip(detrended, 0, None)

    logger.info("Detrending: %d entries clipped to 0 (%.2f%%, max magnitude=%.6f)",
                n_clipped, pct_clipped, max_clip)

    # Convert back to dict format
    detrended_fo_dict = {}
    for i, rid in enumerate(run_ids):
        detrended_fo_dict[rid] = detrended[i, :]

    clip_info = dict(
        n_clipped=n_clipped,
        max_clip_magnitude=max_clip,
        pct_clipped=round(pct_clipped, 4),
    )
    return detrended_fo_dict, clip_info


# =============================================================================
# Main analysis
# =============================================================================

def run_lme_habituation(fo_dict, n_states, acq_dict, session_map,
                        session_gap_days=7, n_perm=5000, seed=44):
    """LME-based within-session habituation analysis for all states.

    For each state k:
    1. Fit FO_k ~ run_idx_centered + (1 | session) via statsmodels
    2. Extract β₁, SE, ICC, variance components
    3. Permutation test: shuffle run indices within sessions,
       compute GLS slope with fixed VCs → null distribution → p-value
    4. Also compute global Spearman as complementary metric
    5. Compute session-boundary reset metric

    Returns dict with all results.
    """
    # ── Sort runs chronologically ────────────────────────────────────────
    run_ids = sorted(fo_dict.keys(),
                     key=lambda r: acq_dict.get(r, 0) if acq_dict else 0)
    n_runs = len(run_ids)

    nan_arr = np.full(n_states, np.nan)
    empty_result = dict(
        lme_slope=nan_arr.copy(), lme_se=nan_arr.copy(),
        lme_icc=nan_arr.copy(),
        lme_sigma2_session=nan_arr.copy(), lme_sigma2_resid=nan_arr.copy(),
        lme_converged=np.zeros(n_states, dtype=bool),
        perm_p=nan_arr.copy(),
        rho_global=nan_arr.copy(), p_global=nan_arr.copy(),
        fo_matrix=None, run_indices=None, session_labels=None,
        reset_fo=None, has_sessions=False, session_source='none',
        n_sessions=0, n_sessions_usable=0, session_sizes=[],
    )
    if n_runs == 0:
        return empty_result

    fo_matrix = np.array([fo_dict[r] for r in run_ids])  # (n_runs, n_states)
    run_indices = np.arange(n_runs, dtype=float)

    # ── Global Spearman (always computed) ────────────────────────────────
    rho_global = np.full(n_states, np.nan)
    p_global = np.full(n_states, np.nan)
    for k in range(n_states):
        rho_global[k], p_global[k] = _safe_spearman(fo_matrix[:, k], run_indices)

    # ── Assign sessions ──────────────────────────────────────────────────
    session_labels, has_sessions, session_source = assign_sessions(
        run_ids, acq_dict, session_map, session_gap_days
    )

    if not has_sessions:
        empty_result.update(
            rho_global=rho_global, p_global=p_global,
            fo_matrix=fo_matrix, run_indices=run_indices,
        )
        return empty_result

    unique_sessions = np.sort(np.unique(session_labels[session_labels >= 0]))
    n_sessions = len(unique_sessions)

    # Pre-compute centered run indices and session sizes
    session_sizes = []
    run_idx_centered = np.full(n_runs, np.nan)

    for sess in unique_sessions:
        mask = session_labels == sess
        n_s = int(mask.sum())
        session_sizes.append(n_s)
        if n_s >= MIN_SESSION_SIZE:
            idx = np.arange(n_s, dtype=float)
            run_idx_centered[mask] = idx - idx.mean()

    n_sessions_usable = sum(1 for s in session_sizes if s >= MIN_SESSION_SIZE)
    logger.info("%d total sessions, %d usable (>= %d runs)",
                n_sessions, n_sessions_usable, MIN_SESSION_SIZE)

    # Mask for usable runs (in sessions with enough runs)
    usable_mask = np.isfinite(run_idx_centered)
    usable_indices = np.where(usable_mask)[0]
    n_usable = len(usable_indices)

    if n_usable < 6 or n_sessions_usable < 2:
        logger.warning("Too few usable runs (%d) or sessions (%d) for LME",
                       n_usable, n_sessions_usable)
        empty_result.update(
            rho_global=rho_global, p_global=p_global,
            fo_matrix=fo_matrix, run_indices=run_indices,
            session_labels=session_labels, has_sessions=True,
            session_source=session_source,
            n_sessions=n_sessions, n_sessions_usable=n_sessions_usable,
            session_sizes=session_sizes,
        )
        return empty_result

    # Subset to usable runs
    fo_usable = fo_matrix[usable_mask]
    ridx_usable = run_idx_centered[usable_mask]
    sess_usable = session_labels[usable_mask]

    # Pre-compute usable session masks (for GLS permutation)
    usable_unique = np.sort(np.unique(sess_usable))
    usable_session_masks = [sess_usable == s for s in usable_unique]

    # ── Fit LME per state ────────────────────────────────────────────────
    lme_slope = np.full(n_states, np.nan)
    lme_se = np.full(n_states, np.nan)
    lme_icc = np.full(n_states, np.nan)
    lme_sigma2_session = np.full(n_states, np.nan)
    lme_sigma2_resid = np.full(n_states, np.nan)
    lme_converged = np.zeros(n_states, dtype=bool)

    n_singular = 0
    n_failed = 0

    for k in range(n_states):
        result = _fit_ri_lme(fo_usable[:, k], ridx_usable, sess_usable)
        lme_slope[k] = result['slope']
        lme_se[k] = result['se']
        lme_icc[k] = result['icc']
        lme_sigma2_session[k] = result['sigma2_session']
        lme_sigma2_resid[k] = result['sigma2_resid']
        lme_converged[k] = result['converged']
        if result['singular']:
            n_singular += 1
        if not result['converged']:
            n_failed += 1

    logger.info("LME fits: %d converged (%d singular), %d failed",
                int(lme_converged.sum()), n_singular, n_failed)

    # ── Permutation test with fixed VCs ──────────────────────────────────
    logger.info("Running permutation test (n_perm=%d)...", n_perm)
    rng = np.random.default_rng(seed)
    perm_p = np.full(n_states, np.nan)
    null_slopes = np.zeros((n_perm, n_states))

    for pi in range(n_perm):
        ridx_perm = ridx_usable.copy()
        # Shuffle run indices within each session
        for mask in usable_session_masks:
            indices = np.where(mask)[0]
            if len(indices) >= MIN_SESSION_SIZE:
                perm_vals = ridx_perm[indices].copy()
                rng.shuffle(perm_vals)
                ridx_perm[indices] = perm_vals

        for k in range(n_states):
            if not lme_converged[k]:
                null_slopes[pi, k] = np.nan
                continue
            null_slopes[pi, k] = _gls_slope_fixed_vc(
                fo_usable[:, k], ridx_perm, usable_session_masks,
                lme_sigma2_session[k], lme_sigma2_resid[k],
            )

    for k in range(n_states):
        if np.isfinite(lme_slope[k]):
            perm_p[k] = _permutation_p_twosided(lme_slope[k], null_slopes[:, k])

    # ── Session-boundary reset ───────────────────────────────────────────
    reset_fo = _session_boundary_reset(fo_matrix, session_labels, unique_sessions)

    return dict(
        lme_slope=lme_slope,
        lme_se=lme_se,
        lme_icc=lme_icc,
        lme_sigma2_session=lme_sigma2_session,
        lme_sigma2_resid=lme_sigma2_resid,
        lme_converged=lme_converged,
        perm_p=perm_p,
        rho_global=rho_global,
        p_global=p_global,
        fo_matrix=fo_matrix,
        run_indices=run_indices,
        run_ids=run_ids,
        run_idx_centered=run_idx_centered,
        session_labels=session_labels,
        reset_fo=reset_fo,
        has_sessions=True,
        session_source=session_source,
        n_sessions=n_sessions,
        n_sessions_usable=n_sessions_usable,
        session_sizes=session_sizes,
        n_singular=n_singular,
        n_failed=n_failed,
    )


# =============================================================================
# Plotting
# =============================================================================

def plot_per_state(results, recurrence_scores, sub_id, out_dir, n_states):
    """Per-state plot: run-level FO vs global run index with session boundaries."""
    fo_matrix = results['fo_matrix']
    if fo_matrix is None:
        logger.warning("No run-level FO data; skipping per-state plots")
        return

    run_indices = results['run_indices']
    session_labels = results['session_labels']
    has_sessions = results['has_sessions']
    n_sessions = results['n_sessions']
    n_usable = results['n_sessions_usable']

    state_dir = os.path.join(out_dir, 'habituation_per_state')
    os.makedirs(state_dir, exist_ok=True)

    for k in range(n_states):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        color = recurrence_color(recurrence_scores[k])
        y = fo_matrix[:, k]

        ax.scatter(run_indices, y, s=20, alpha=0.7, color=color,
                   edgecolors='none')

        # Global trend line
        if len(run_indices) >= 2 and np.std(y) > 1e-10:
            z = np.polyfit(run_indices, y, 1)
            x_line = np.linspace(run_indices.min(), run_indices.max(), 50)
            ax.plot(x_line, np.polyval(z, x_line),
                    '--', color=color, alpha=0.5, linewidth=1,
                    label='global trend')

        # Session boundaries
        if has_sessions and session_labels is not None:
            unique_sess = np.unique(session_labels[session_labels >= 0])
            for si in range(1, len(unique_sess)):
                prev_end = np.where(session_labels == unique_sess[si - 1])[0][-1]
                curr_start = np.where(session_labels == unique_sess[si])[0][0]
                boundary_x = (run_indices[prev_end] + run_indices[curr_start]) / 2
                ax.axvline(boundary_x, color='red', linestyle=':', linewidth=0.7,
                           alpha=0.4)
            ax.plot([], [], color='red', linestyle=':', linewidth=0.7,
                    alpha=0.4, label=f'{n_sessions} sessions ({n_usable} usable)')

        # Annotation
        slope = results['lme_slope'][k]
        icc = results['lme_icc'][k]
        slope_str = f'β₁={slope:.4f}' if np.isfinite(slope) else 'β₁=NaN'
        icc_str = f'ICC={icc:.3f}' if np.isfinite(icc) else 'ICC=NaN'

        ax.set_xlabel('Run index (chronological)', fontsize=7)
        ax.set_ylabel('FO', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=6, loc='upper right')
        ax.set_title(
            f'State {k} - Within-Session Habituation (LME)\n'
            f'{slope_str}  {icc_str}  rec={recurrence_scores[k]:.2f} - {sub_id}',
            fontsize=9,
        )
        fig.tight_layout()

        out_png = os.path.join(state_dir, f'state_{k:03d}.png')
        fig.savefig(out_png, bbox_inches='tight', dpi=100)
        plt.close(fig)

    logger.info("Saved %d per-state plots to %s", n_states, state_dir)


def plot_summary(results, recurrence_scores, q_values, is_eligible,
                 sub_id, out_dir, n_states,
                 mean_fo=None, dominant_networks=None):
    """Summary plot: slope distribution, ICC distribution, volcano, slope vs mean FO."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    slope = results['lme_slope']
    icc = results['lme_icc']
    converged = results['lme_converged']
    valid = converged & np.isfinite(slope)

    # Color by recurrence
    colors = np.array([recurrence_color(r) for r in recurrence_scores])

    # (a) Slope distribution
    ax = axes[0, 0]
    valid_slopes = slope[valid]
    if len(valid_slopes) > 0:
        ax.hist(valid_slopes, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
        median_slope = np.median(valid_slopes)
        ax.axvline(median_slope, color='red', linestyle='--', linewidth=0.8,
                   label=f'median={median_slope:.5f}')
        ax.legend(fontsize=7)
    ax.set_xlabel('LME slope (β₁)', fontsize=8)
    ax.set_ylabel('Count', fontsize=8)
    ax.set_title('(a) Within-session slope distribution', fontsize=9)
    ax.tick_params(labelsize=7)

    # (b) ICC distribution
    ax = axes[0, 1]
    valid_icc = icc[valid]
    if len(valid_icc) > 0:
        ax.hist(valid_icc, bins=30, color='darkorange', alpha=0.7, edgecolor='white')
        median_icc = np.median(valid_icc)
        ax.axvline(median_icc, color='red', linestyle='--', linewidth=0.8,
                   label=f'median={median_icc:.5f}')
        ax.legend(fontsize=7)
    ax.set_xlabel('ICC', fontsize=8)
    ax.set_ylabel('Count', fontsize=8)
    ax.set_title('(b) Intraclass correlation distribution', fontsize=9)
    ax.tick_params(labelsize=7)

    # (c) Volcano plot: slope vs -log10(q)
    ax = axes[1, 0]
    if q_values is not None:
        valid_q = valid & np.isfinite(q_values)
    else:
        valid_q = np.zeros(n_states, dtype=bool)
    if valid_q.sum() > 0:
        neg_log_q = -np.log10(np.clip(q_values[valid_q], 1e-20, None))
        for i, k in enumerate(np.where(valid_q)[0]):
            marker = 'o' if is_eligible[k] else 'x'
            ax.scatter(slope[k], neg_log_q[i], s=25, color=colors[k],
                       marker=marker, alpha=0.7, edgecolors='none')
        ax.axhline(-np.log10(0.05), color='grey', linestyle='--',
                   linewidth=0.8, label='q=0.05')
        ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
        # Add sub-HRF marker to legend
        ax.scatter([], [], marker='x', color='gray', s=25, label='Sub-HRF')
        ax.legend(fontsize=7)
    ax.set_xlabel('LME slope (β₁)', fontsize=8)
    ax.set_ylabel('-log₁₀(q)', fontsize=8)
    ax.set_title('(c) Volcano plot', fontsize=9)
    ax.tick_params(labelsize=7)

    # (d) Slope vs mean FO, colored by dominant network
    ax = axes[1, 1]
    has_network_data = (mean_fo is not None and dominant_networks is not None)
    if has_network_data:
        plot_mask = valid & (mean_fo > 0)
        if plot_mask.sum() > 0:
            present_nets = set()
            for k in np.where(plot_mask)[0]:
                entry = dominant_networks.get(int(k), ("Unknown", "+"))
                # Support both tuple (net, sign) and plain string
                if isinstance(entry, tuple):
                    net, sign = entry
                else:
                    net, sign = entry, ""
                color = NETWORK_COLORS.get(net, "#888888")
                marker = 'o' if is_eligible[k] else 'x'
                if sign == '-' and marker == 'o':
                    ax.scatter(mean_fo[k], slope[k], s=25, facecolors='none',
                               edgecolors=color, linewidths=1.0,
                               marker=marker, alpha=0.7)
                else:
                    ax.scatter(mean_fo[k], slope[k], s=25, color=color,
                               marker=marker, alpha=0.7, edgecolors='none')
                present_nets.add((net, sign))
            ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
            # Network color legend (signed labels)
            handles = []
            for n in NETWORK_ORDER:
                signs_for_net = sorted({s for nn, s in present_nets if nn == n})
                for s in signs_for_net:
                    label = f"{n}{s}" if s else n
                    net_color = NETWORK_COLORS.get(n, '#888')
                    if s == '-':
                        handles.append(plt.Line2D(
                            [0], [0], marker='o', color='w',
                            markerfacecolor='none', markeredgecolor=net_color,
                            markeredgewidth=1.0, markersize=5, label=label))
                    else:
                        handles.append(plt.Line2D(
                            [0], [0], marker='o', color='w',
                            markerfacecolor=net_color,
                            markersize=5, label=label))
            # Marker shape legend (sub-HRF only)
            handles.append(plt.Line2D([0], [0], marker='x', color='gray',
                                      markersize=5, label='Sub-HRF',
                                      linestyle='none'))
            if handles:
                ax.legend(handles=handles, fontsize=5, ncol=2, loc='best',
                          framealpha=0.7, handletextpad=0.3, columnspacing=0.8)
        ax.set_xlabel('Mean fractional occupancy', fontsize=8)
        ax.set_ylabel('LME slope (β₁)', fontsize=8)
        ax.set_title('(d) Slope vs mean FO (color = dominant network)', fontsize=9)
    else:
        # Fallback: slope vs recurrence (original panel)
        if valid.sum() > 0:
            for k in np.where(valid)[0]:
                marker = 'o' if is_eligible[k] else 'x'
                ax.scatter(recurrence_scores[k], slope[k], s=25, color=colors[k],
                           marker=marker, alpha=0.7, edgecolors='none')
            ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Recurrence score', fontsize=8)
        ax.set_ylabel('LME slope (β₁)', fontsize=8)
        ax.set_title('(d) Slope vs recurrence (fallback)', fontsize=9)
    ax.tick_params(labelsize=7)

    fig.suptitle(f'Within-Session Habituation (LME) - {sub_id}', fontsize=11)
    fig.tight_layout()

    for ext in ('png', 'pdf'):
        out_path = os.path.join(out_dir, f'habituation_summary.{ext}')
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    logger.info("Saved summary plot to %s", out_dir)


# =============================================================================
# Output serialization
# =============================================================================

def _fmt(val):
    """Format a scalar for CSV/JSON: finite -> str, else 'NaN'."""
    if isinstance(val, (float, np.floating)):
        return f'{val:.6g}' if np.isfinite(val) else 'NaN'
    return str(val)


def _arr(arr):
    """Convert numpy array to JSON-serializable list (NaN -> None)."""
    if arr is None:
        return None
    return [None if not np.isfinite(v) else round(float(v), 8) for v in arr]


def save_csv(results, q_values, recurrence_scores, is_eligible, out_dir, n_states):
    """Save per-state metrics CSV."""
    csv_path = os.path.join(out_dir, 'habituation_metrics.csv')
    fieldnames = [
        'state', 'recurrence_score', 'eligible',
        'lme_slope', 'lme_se', 'lme_icc',
        'lme_sigma2_session', 'lme_sigma2_resid',
        'perm_p', 'q_fdr', 'converged',
        'rho_global', 'p_global',
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k in range(n_states):
            writer.writerow({
                'state': k,
                'recurrence_score': _fmt(recurrence_scores[k]),
                'eligible': int(is_eligible[k]),
                'lme_slope': _fmt(results['lme_slope'][k]),
                'lme_se': _fmt(results['lme_se'][k]),
                'lme_icc': _fmt(results['lme_icc'][k]),
                'lme_sigma2_session': _fmt(results['lme_sigma2_session'][k]),
                'lme_sigma2_resid': _fmt(results['lme_sigma2_resid'][k]),
                'perm_p': _fmt(results['perm_p'][k]),
                'q_fdr': _fmt(q_values[k]) if q_values is not None else 'NaN',
                'converged': int(results['lme_converged'][k]),
                'rho_global': _fmt(results['rho_global'][k]),
                'p_global': _fmt(results['p_global'][k]),
            })
    logger.info("Saved CSV to %s", csv_path)


def save_json(results, q_values, recurrence_scores, is_eligible,
              sub_id, parc, n_states, args, out_dir, clip_info=None):
    """Save full results JSON."""
    n_converged = int(results['lme_converged'].sum())
    n_testable = int(np.sum(np.isfinite(results['perm_p'])))
    n_sig = 0
    if q_values is not None:
        n_sig = int(np.sum(np.isfinite(q_values) & (q_values < 0.05)))

    output = {
        'metadata': {
            'sub_id': sub_id,
            'parcellation': parc,
            'n_states': n_states,
            'n_perm': args.n_perm,
            'session_gap_days': args.session_gap_days,
            'vt': args.vt,
            'exclude_sub_hrf': args.exclude_sub_hrf,
            'script': '05e_temporal_trend_a3.py',
        },
        'session_info': {
            'has_sessions': results['has_sessions'],
            'session_source': results['session_source'],
            'n_sessions': results['n_sessions'],
            'n_sessions_usable': results['n_sessions_usable'],
            'session_sizes': results['session_sizes'],
        },
        'convergence': {
            'n_converged': n_converged,
            'n_singular': results.get('n_singular', 0),
            'n_failed': results.get('n_failed', 0),
        },
        'lme_results': {
            'description': (
                'Random-intercept LME: FO_k ~ run_idx_centered + (1 | session). '
                'Fixed slope β₁ = within-session FO trend rate (FO change per run). '
                'Permutation p-values from shuffling run indices within sessions '
                'with fixed variance components.'
            ),
            'interpretation': (
                'Negative slope = FO decreases across runs within session (habituation); '
                'positive slope = FO increases (sensitization); '
                'near-zero = stable within-session FO.'
            ),
            'n_testable': n_testable,
            'n_sig_fdr05': n_sig,
            'slope': _arr(results['lme_slope']),
            'se': _arr(results['lme_se']),
            'icc': _arr(results['lme_icc']),
            'sigma2_session': _arr(results['lme_sigma2_session']),
            'sigma2_resid': _arr(results['lme_sigma2_resid']),
            'perm_p': _arr(results['perm_p']),
            'q_fdr': _arr(q_values) if q_values is not None else None,
            'converged': [bool(v) for v in results['lme_converged']],
        },
        'global_spearman': {
            'description': (
                'Global Spearman rho(FO_k, run_index) as complementary metric. '
                'WARNING: conflates within-session and between-session trends. '
                'Do not use for inference about within-session effects; use LME slope instead.'
            ),
            'rho': _arr(results['rho_global']),
            'p': _arr(results['p_global']),
        },
        'reset_fo': {
            'description': (
                'Mean FO jump at session boundaries '
                '(first run of session N+1 minus last run of session N).'
            ),
            'mean': _arr(results['reset_fo']),
        },
        'icc_interpretation': {
            'description': (
                'ICC = σ²_session / (σ²_session + σ²_resid). '
                'High ICC (>0.5): state occupancy is session-level trait. '
                'Low ICC (<0.2): state occupancy varies within sessions. '
                'Expected: moderate ICC for recurring states, low for specific.'
            ),
        },
        'detrended_fo': {
            'description': (
                'Session-detrended FO: FO_k - β₁_k × run_idx_centered. '
                'Removes within-session linear trend. Clipped to [0, ∞), '
                'no renormalization. Saved as fractional_occupancy_detrended.pkl.'
            ),
            'clip_info': clip_info if clip_info is not None else {},
        },
    }

    json_path = os.path.join(out_dir, 'habituation_results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info("Saved JSON to %s", json_path)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Within-session FO habituation via LME (05e_a3)"
    )
    parser.add_argument('--sub_id', type=str, required=True)
    parser.add_argument('--parcellation', type=str, default='atlas-4S156Parcels')
    parser.add_argument('--vt', type=str, default=None,
                        help="Variance threshold subdir (e.g. 0.95).")
    parser.add_argument('--n_perm', type=int, default=5000,
                        help="Number of permutations (default: 5000)")
    parser.add_argument('--session_gap_days', type=int, default=7,
                        help="Gap threshold for fallback session clustering (default: 7)")
    parser.add_argument('--exclude_sub_hrf', action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Exclude sub-HRF states from significance counts "
                             "(default: True). Use --no-exclude_sub_hrf to include all.")
    return parser.parse_args()


def main():
    args = parse_args()
    sub_id = args.sub_id
    parc = normalize_parcellation_name(args.parcellation)

    # ── Input paths ──────────────────────────────────────────────────────
    recurrence_base = os.path.join(
        SCRATCH_DIR, 'output', '05a_recurrence_analysis', parc, sub_id
    )
    hmm_base = os.path.join(
        SCRATCH_DIR, 'output', '04_combined_hdphmm', parc, sub_id, 'final'
    )
    if args.vt is not None:
        recurrence_base = os.path.join(recurrence_base, f'vt{args.vt}')
        hmm_base = os.path.join(hmm_base, f'vt{args.vt}')

    fo_path = os.path.join(recurrence_base, 'fractional_occupancy.pkl')
    summary_path = os.path.join(recurrence_base, 'recurrence_summary.json')
    scores_path = os.path.join(recurrence_base, 'recurrence_scores.npy')

    for p in (fo_path, summary_path, scores_path):
        if not os.path.exists(p):
            logger.error("Missing required input: %s", p)
            sys.exit(1)

    # ── Output directory ─────────────────────────────────────────────────
    out_dir = os.path.join(
        SCRATCH_DIR, 'output', '05e_temporal_trend_a3', parc, sub_id
    )
    if args.vt is not None:
        out_dir = os.path.join(out_dir, f'vt{args.vt}')
    if not args.exclude_sub_hrf:
        out_dir = os.path.join(out_dir, 'all_states')
    os.makedirs(out_dir, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    logger.info("Loading data...")
    with open(fo_path, 'rb') as f:
        fo_dict = pickle.load(f)
    with open(summary_path, 'r') as f:
        recurrence_summary = json.load(f)

    recurrence_scores = np.load(scores_path)
    n_states = recurrence_summary['n_states']

    logger.info("n_states=%d, n_runs=%d", n_states, len(fo_dict))

    # Load eligible states
    try:
        eligible_ids, excluded_ids, _ = load_eligible_states(recurrence_base)
        eligible_set = set(eligible_ids)
        excluded_set = set(excluded_ids)
        logger.info("Eligible states: %d / %d (sub-HRF excluded: %d)",
                    len(eligible_set), n_states, len(excluded_set))
    except FileNotFoundError:
        logger.warning("eligible_states.json not found; all states treated as eligible.")
        eligible_set = set(range(n_states))
        excluded_set = set()

    is_eligible = np.array([k in eligible_set for k in range(n_states)])

    # ── Load acquisition times ───────────────────────────────────────────
    acq_dict, bids_session_map = load_acquisition_times(sub_id)
    if acq_dict is None and bids_session_map is None:
        logger.error("No acquisition time or session data available. "
                     "Run 00_get_scan before 05e_temporal_trend_a3.")
        sys.exit(1)

    # ── Run LME analysis ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Running LME within-session FO trend analysis...")
    logger.info("=" * 60)

    results = run_lme_habituation(
        fo_dict, n_states, acq_dict, bids_session_map,
        session_gap_days=args.session_gap_days, n_perm=args.n_perm,
    )

    # ── FDR correction ───────────────────────────────────────────────────
    q_values = None
    if results['has_sessions'] and np.any(np.isfinite(results['perm_p'])):
        q_values = _fdr_with_nan(results['perm_p'])
        n_testable = int(np.sum(np.isfinite(results['perm_p'])))
        n_sig = int(np.sum(np.isfinite(q_values) & (q_values < 0.05)))
        logger.info("FDR correction: %d testable, %d significant (q<0.05)",
                    n_testable, n_sig)

        # Count among eligible only
        if args.exclude_sub_hrf:
            elig_mask = is_eligible & np.isfinite(q_values)
            n_elig_sig = int(np.sum(elig_mask & (q_values < 0.05)))
            logger.info("  Among eligible: %d significant", n_elig_sig)

    # ── Compute and save detrended FO ────────────────────────────────────
    detrended_fo_dict, clip_info = compute_detrended_fo(
        fo_dict, results, n_states
    )
    detrended_path = os.path.join(out_dir, 'fractional_occupancy_detrended.pkl')
    with open(detrended_path, 'wb') as f:
        pickle.dump(detrended_fo_dict, f, protocol=4)
    logger.info("Saved detrended FO to %s", detrended_path)

    # ── Save outputs ─────────────────────────────────────────────────────
    save_csv(results, q_values, recurrence_scores, is_eligible, out_dir, n_states)
    save_json(results, q_values, recurrence_scores, is_eligible,
              sub_id, parc, n_states, args, out_dir,
              clip_info=clip_info)

    # ── Network data for panel (d) ────────────────────────────────────
    dominant_networks = None
    state_means_path = os.path.join(hmm_base, 'state_means_parcel.npy')
    try:
        state_means = np.load(state_means_path)
        parcel_networks = load_parcel_networks(parc)
        if parcel_networks is not None:
            dominant_networks = compute_dominant_networks(
                state_means, np.arange(n_states), parcel_networks,
                include_sign=True)
            logger.info("Loaded dominant networks for %d states", len(dominant_networks))
    except Exception as e:
        logger.warning("Could not load network data: %s - panel (d) will use fallback", e)

    fo_matrix = results.get('fo_matrix')
    mean_fo = np.mean(fo_matrix, axis=0) if fo_matrix is not None else None

    # ── Plots ────────────────────────────────────────────────────────────
    plot_per_state(results, recurrence_scores, sub_id, out_dir, n_states)
    plot_summary(results, recurrence_scores, q_values, is_eligible,
                 sub_id, out_dir, n_states,
                 mean_fo=mean_fo, dominant_networks=dominant_networks)

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Analysis complete!")
    logger.info("Output directory: %s", out_dir)
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
