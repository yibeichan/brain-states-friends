#!/usr/bin/env python3
"""
05e_temporal_trend_a1.py - Cross-episode temporal trends in brain state occupancy.

Investigates whether brain states exhibit systematic temporal trends across
episodes at four hierarchical scales, plus two supplementary diagnostics:

Scale 1 — Cross-season (coarsest; 6 data points per state):
    Mann-Kendall tau(mean_FO_per_season, season_number).
    Uses per_season_mean_fo.json from 05a. Exploratory (very low power).

Scale 2 — Within-season episode position (~24 broadcast episodes/season):
    FO aggregated to broadcast-episode level (a+b -> one episode; for 4-part
    episodes, c+d -> next episode number). Spearman rho per season, combined
    via permutation test on mean_rho across seasons.

Scale 3 — Multi-predictor variance partitioning:
    Semi-partial R^2 decomposition: global_position (proxy for scanning date),
    season (categorical), within-season position (ordinal). Permutation inference.

Diagnostic 1 — Motion confound check:
    Run-level mean FD trend + FD-controlled partial correlations.

Diagnostic 2 — Anti-correlated state pair analysis:
    Tests whether emission-space anti-correlated states show opposite FO trends.

No state classification — all analyses produce continuous effect sizes and
FDR-corrected q-values. All states are analyzed; eligible (non-sub-HRF) states
flagged in output.

Inputs (from 05a, 05e_a1, 04; vt-aware):
    - per_season_mean_fo.json, fractional_occupancy.pkl, recurrence_scores.npy,
      recurrence_summary.json, eligible_states.json (from 05a)
    - decoded_states.pkl, best_model.pkl (from 04)
    - fMRIPrep confound TSVs (from DATA_DIR)

Outputs (saved to {SCRATCH_DIR}/output/05e_temporal_trend_a1/{parc}/{sub_id}/[vt{VT}/]):
    - temporal_trend_results.json
    - temporal_trend_metrics.csv
    - scale1_all_states_cross_season.png/pdf
    - scale2_within_season/ (per-state 6-panel PNGs)
    - scale3_variance_partition.png/pdf
    - trend_vs_mean_fo.png/pdf
    - motion_confound_check.png/pdf
    - state_pair_trends.png/pdf

See also:
    05e_temporal_trend_a2.py — within-run temporal position
    05e_temporal_trend_a3.py — within-session FO habituation (LME)
Design doc: the design notes
"""

import argparse
import csv
import glob
import json
import logging
import os
import pickle
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).parent))
from utils.stats import benjamini_hochberg, fdr_with_nan as _fdr_with_nan
from utils.plot_style import recurrence_color, make_recurrence_colorbar, apply_publication_style
from utils.common import (
    normalize_parcellation_name, _get_season, parse_episode_order_key,
    group_runs_to_broadcast_episodes, aggregate_fo_broadcast,
)
from utils.state_blocks import load_eligible_states

apply_publication_style()

from dotenv import load_dotenv

load_dotenv()
SCRATCH_DIR = os.getenv('SCRATCH_DIR')
DATA_DIR = os.getenv('DATA_DIR')
if SCRATCH_DIR is None:
    raise ValueError("SCRATCH_DIR must be set in the .env file")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# =============================================================================
# Acquisition time loading
# =============================================================================

def load_acquisition_times(sub_id):
    """Load run-level relative acquisition times from 00_get_scan output.

    Returns:
        acq_dict: dict run_id -> float (rel_acq_time in days), or None if CSV not found.
        session_map: dict run_id -> str (BIDS session_id, e.g. 'ses-001'),
                     or None if session_id column missing or CSV not found.
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

    logger.info("Loaded acquisition times for %d runs from %s", len(acq_dict), csv_path)
    if has_session_col:
        n_unique = len(set(session_map.values()))
        logger.info("  BIDS session_id available: %d unique sessions", n_unique)
    else:
        logger.info("  No session_id column; will fall back to gap-based clustering")
        session_map = None
    return acq_dict, session_map


# =============================================================================
# Shared statistical helpers
# =============================================================================

def _safe_spearman(x, y):
    """Spearman rho with NaN guard. Returns (rho, p) or (nan, nan)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan, np.nan
    if np.std(x[valid]) < 1e-10 or np.std(y[valid]) < 1e-10:
        return np.nan, np.nan
    rho, p = spearmanr(x[valid], y[valid])
    return float(rho), float(p)


def _permutation_p_twosided(observed, null_dist):
    """Two-sided permutation p-value with Phipson & Smyth correction."""
    from utils.stats import permutation_pvalue
    return permutation_pvalue(observed, null_dist, alternative='two-sided')


def _parse_run_order_key(run_id):
    """Parse run_id into (season, episode_num, part_order) for chronological sorting."""
    season, episode_num = parse_episode_order_key(run_id)
    match = re.match(r'^s\d+e\d+([a-z])$', run_id)
    part_order = ord(match.group(1)) - ord('a') if match else 0
    return season, episode_num, part_order


def _lag1_autocorrelation(fo_matrix):
    """Compute mean lag-1 autocorrelation of FO across episodes (vectorized).

    Args:
        fo_matrix: np.array(n_episodes, n_states) — rows in chronological order.

    Returns:
        mean_lag1_autocorr: float — mean across states of lag-1 autocorrelation.
        per_state_lag1: np.array(n_states,) — per-state lag-1 autocorrelation.
        effective_n_approx: float — approximate effective sample size
            (Bayley & Hammersley 1946: n_eff ≈ n * (1 - r1) / (1 + r1)).
    """
    n_eps, n_states = fo_matrix.shape
    if n_eps < 4:
        return np.nan, np.full(n_states, np.nan), np.nan

    # Vectorized lag-1 autocorrelation per state
    x = fo_matrix[:-1]  # (n_eps-1, n_states)
    y = fo_matrix[1:]   # (n_eps-1, n_states)
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    x_c = x - x_mean
    y_c = y - y_mean
    numer = np.sum(x_c * y_c, axis=0)
    denom = np.sqrt(np.sum(x_c ** 2, axis=0) * np.sum(y_c ** 2, axis=0))
    per_state_lag1 = np.where(denom > 1e-20, numer / denom, np.nan)

    mean_r1 = float(np.nanmean(per_state_lag1))
    # Effective N: Bayley & Hammersley (1946) approximation
    if abs(mean_r1) < 1.0:
        effective_n = n_eps * (1 - mean_r1) / (1 + mean_r1)
    else:
        effective_n = np.nan

    return mean_r1, per_state_lag1, float(effective_n)


# =============================================================================
# Scale 1: Cross-season
# =============================================================================

def scale1_cross_season(per_season_fo, n_states):
    """Mann-Kendall tau of per-season mean FO vs season number.

    Uses kendalltau(method='auto'): exact permutation when no ties,
    asymptotic otherwise. n=6 -> very low power; exploratory only.
    """
    seasons = sorted(per_season_fo.keys(), key=int)
    season_nums = np.array([int(s) for s in seasons], dtype=float)
    fo_matrix = np.array([np.array(per_season_fo[s], dtype=float) for s in seasons])

    tau = np.full(n_states, np.nan)
    p_val = np.full(n_states, np.nan)

    for k in range(n_states):
        fo_k = fo_matrix[:, k]
        valid = np.isfinite(fo_k)
        if valid.sum() < 3:
            continue
        if np.std(fo_k[valid]) < 1e-10:
            continue
        t, p = kendalltau(fo_k, season_nums, method='auto')
        tau[k], p_val[k] = float(t), float(p)

    return tau, p_val


# =============================================================================
# Scale 2: Within-season episode position
# =============================================================================

def _batch_spearman_ranks(fo_matrix):
    """Pre-rank each column of fo_matrix for vectorized Spearman.

    Returns:
        ranked: np.array same shape, rank-transformed per column.
        constant_mask: np.array(n_states,) — True if column is constant.
    """
    from scipy.stats import rankdata
    n_obs, n_states = fo_matrix.shape
    ranked = np.empty_like(fo_matrix)
    constant_mask = np.zeros(n_states, dtype=bool)
    for k in range(n_states):
        col = fo_matrix[:, k]
        if np.std(col) < 1e-10:
            constant_mask[k] = True
            ranked[:, k] = np.nan
        else:
            ranked[:, k] = rankdata(col)
    return ranked, constant_mask


def _pearson_with_vec(x_centered, y_ranked_matrix):
    """Pearson correlation of centered vector x with each column of centered Y.

    Args:
        x_centered: np.array(n,) — already centered (mean-subtracted).
        y_ranked_matrix: np.array(n, n_states) — already centered per column.

    Returns:
        rho: np.array(n_states,) — correlation per column.
    """
    ss_x = np.dot(x_centered, x_centered)
    if ss_x < 1e-20:
        return np.full(y_ranked_matrix.shape[1], np.nan)
    xy = x_centered @ y_ranked_matrix  # (n_states,)
    ss_y = np.sum(y_ranked_matrix ** 2, axis=0)  # (n_states,)
    denom = np.sqrt(ss_x * ss_y)
    return np.where(denom > 1e-20, xy / denom, np.nan)


def scale2_within_season(episode_fo, broadcast_meta, n_states, n_perm=5000, seed=42):
    """Spearman rho per season, combined via permutation test on mean_rho.

    Vectorized: pre-ranks FO columns per season, then computes Spearman as
    Pearson on ranks across all states simultaneously.

    Args:
        episode_fo: dict broadcast_id -> np.array(n_states,)
        broadcast_meta: dict broadcast_id -> {'season', 'episode_num', ...}
        n_states: int
        n_perm: number of permutations
        seed: random seed

    Returns:
        mean_rho: np.array(n_states,)
        perm_p: np.array(n_states,) — permutation p-values
        per_season_rho: np.array(n_seasons, n_states) — per-season rho
    """
    from scipy.stats import rankdata

    # Build arrays sorted by season and episode position
    bcast_ids = sorted(broadcast_meta.keys(),
                       key=lambda b: (broadcast_meta[b]['season'],
                                      broadcast_meta[b]['episode_num']))
    seasons_arr = np.array([broadcast_meta[b]['season'] for b in bcast_ids])
    ep_nums_arr = np.array([broadcast_meta[b]['episode_num'] for b in bcast_ids], dtype=float)
    fo_matrix = np.array([episode_fo[b] for b in bcast_ids])  # (n_episodes, n_states)

    unique_seasons = np.sort(np.unique(seasons_arr))
    n_seasons = len(unique_seasons)

    # Pre-compute per-season ranked FO and centered ranks
    season_data = []  # list of (mask, n_s, ep_ranks_centered, fo_ranked_centered, const_mask)
    for s in unique_seasons:
        mask = seasons_arr == s
        n_s = int(mask.sum())
        ep_s = ep_nums_arr[mask]
        fo_s = fo_matrix[mask]

        # Rank episode positions and center
        ep_ranks = rankdata(ep_s)
        ep_ranks_c = ep_ranks - ep_ranks.mean()

        # Rank FO columns and center
        fo_ranked, const_mask = _batch_spearman_ranks(fo_s)
        fo_ranked_c = fo_ranked - np.nanmean(fo_ranked, axis=0, keepdims=True)
        fo_ranked_c[:, const_mask] = np.nan

        season_data.append((mask, n_s, ep_s, ep_ranks_c, fo_ranked_c, const_mask))

    # Compute observed per-season rho (vectorized across states)
    per_season_rho = np.full((n_seasons, n_states), np.nan)
    for si, (mask, n_s, ep_s, ep_ranks_c, fo_ranked_c, const_mask) in enumerate(season_data):
        if n_s < 3:
            continue
        rho_vec = _pearson_with_vec(ep_ranks_c, fo_ranked_c)
        per_season_rho[si, :] = rho_vec

    mean_rho = np.nanmean(per_season_rho, axis=0)

    # Permutation test: shuffle episode positions within each season
    rng = np.random.default_rng(seed)
    null_mean_rhos = np.zeros((n_perm, n_states))

    for pi in range(n_perm):
        perm_rho = np.full((n_seasons, n_states), np.nan)
        for si, (mask, n_s, ep_s, _, fo_ranked_c, const_mask) in enumerate(season_data):
            if n_s < 3:
                continue
            # Shuffle and re-rank episode positions
            ep_perm = ep_s.copy()
            rng.shuffle(ep_perm)
            ep_ranks_perm = rankdata(ep_perm)
            ep_ranks_perm_c = ep_ranks_perm - ep_ranks_perm.mean()
            perm_rho[si, :] = _pearson_with_vec(ep_ranks_perm_c, fo_ranked_c)
        null_mean_rhos[pi] = np.nanmean(perm_rho, axis=0)

    perm_p = np.full(n_states, np.nan)
    for k in range(n_states):
        if np.isfinite(mean_rho[k]):
            perm_p[k] = _permutation_p_twosided(mean_rho[k], null_mean_rhos[:, k])

    return mean_rho, perm_p, per_season_rho


# =============================================================================
# Scale 3: Variance partitioning
# =============================================================================

def scale3_variance_partition(episode_fo, broadcast_meta, n_states,
                              episode_acq_times=None, n_perm=5000, seed=45):
    """Semi-partial R^2 decomposition with permutation inference.

    Predictors: global_position, season (dummy), within_season_position.
    Vectorized across all states: uses np.linalg.lstsq on (n_eps, n_states)
    multivariate response instead of per-state sklearn fits.

    Args:
        episode_acq_times: optional dict broadcast_id -> float (rel_acq_time in days).
            If provided, used as global_pos. Otherwise falls back to ordinal index.

    Returns dict with keys:
        r2_full, dr2_global, dr2_season, dr2_within, shared_r2,
        p_global, p_season, p_within (permutation p-values)
    """
    bcast_ids = sorted(broadcast_meta.keys(),
                       key=lambda b: (broadcast_meta[b]['season'],
                                      broadcast_meta[b]['episode_num']))
    n_eps = len(bcast_ids)

    # Build predictor matrix
    seasons = np.array([broadcast_meta[b]['season'] for b in bcast_ids])
    ep_nums = np.array([broadcast_meta[b]['episode_num'] for b in bcast_ids], dtype=float)

    # Global position: real acquisition times if available, else ordinal index
    if episode_acq_times is not None:
        global_pos = np.array([episode_acq_times[b] for b in bcast_ids], dtype=float)
    else:
        global_pos = np.arange(n_eps, dtype=float)

    # Within-season position: normalized rank within each season
    within_pos = np.zeros(n_eps)
    unique_seasons = np.sort(np.unique(seasons))
    for s in unique_seasons:
        mask = seasons == s
        nums_s = ep_nums[mask]
        if nums_s.max() > nums_s.min():
            within_pos[mask] = (nums_s - nums_s.min()) / (nums_s.max() - nums_s.min())
        else:
            within_pos[mask] = 0.5

    # Season dummies (drop first = season 1)
    season_dummies = np.zeros((n_eps, len(unique_seasons) - 1))
    for i, s in enumerate(unique_seasons[1:]):
        season_dummies[:, i] = (seasons == s).astype(float)

    # Build design matrices
    X_global = global_pos.reshape(-1, 1)
    X_season = season_dummies
    X_within = within_pos.reshape(-1, 1)

    X_full = np.hstack([X_global, X_season, X_within])
    X_no_global = np.hstack([X_season, X_within])
    X_no_season = np.hstack([X_global, X_within])
    X_no_within = np.hstack([X_global, X_season])

    fo_matrix = np.array([episode_fo[b] for b in bcast_ids])  # (n_eps, n_states)

    # ── Vectorized batch R^2 ──────────────────────────────────────────────
    # Pre-compute SS_tot once (constant across all regressions)
    Y_mean = fo_matrix.mean(axis=0)
    ss_tot = np.sum((fo_matrix - Y_mean) ** 2, axis=0)  # (n_states,)

    def _r2_batch(X_int, Y, ss_tot=ss_tot):
        """Batch OLS R^2 for multivariate response Y (n_obs, n_states).

        X_int must already include the intercept column.
        Returns R^2 array of shape (n_states,). NaN for constant states.
        """
        B, residuals, rank, sv = np.linalg.lstsq(X_int, Y, rcond=None)
        Y_pred = X_int @ B
        ss_res = np.sum((Y - Y_pred) ** 2, axis=0)  # (n_states,)
        r2 = np.where(ss_tot > 1e-20, np.maximum(0.0, 1.0 - ss_res / ss_tot), np.nan)
        return r2

    # Pre-build intercept-augmented design matrices (avoids repeated hstack)
    ones_col = np.ones((n_eps, 1))
    X_full_int = np.hstack([ones_col, X_full])
    X_no_g_int = np.hstack([ones_col, X_no_global])
    X_no_s_int = np.hstack([ones_col, X_no_season])
    X_no_w_int = np.hstack([ones_col, X_no_within])

    # ── Observed R^2 values ───────────────────────────────────────────────
    r2_full_arr = _r2_batch(X_full_int, fo_matrix)
    r2_no_g_arr = _r2_batch(X_no_g_int, fo_matrix)
    r2_no_s_arr = _r2_batch(X_no_s_int, fo_matrix)
    r2_no_w_arr = _r2_batch(X_no_w_int, fo_matrix)

    dr2_global = np.where(np.isfinite(r2_full_arr) & np.isfinite(r2_no_g_arr),
                          np.maximum(0.0, r2_full_arr - r2_no_g_arr), np.nan)
    dr2_season = np.where(np.isfinite(r2_full_arr) & np.isfinite(r2_no_s_arr),
                          np.maximum(0.0, r2_full_arr - r2_no_s_arr), np.nan)
    dr2_within = np.where(np.isfinite(r2_full_arr) & np.isfinite(r2_no_w_arr),
                          np.maximum(0.0, r2_full_arr - r2_no_w_arr), np.nan)

    all_finite = (np.isfinite(dr2_global) & np.isfinite(dr2_season) & np.isfinite(dr2_within))
    shared_r2 = np.where(all_finite,
                         np.maximum(0.0, r2_full_arr - dr2_global - dr2_season - dr2_within),
                         np.nan)

    # Mask for testable states (non-NaN R^2_full)
    testable = np.isfinite(r2_full_arr)

    # ── Permutation inference (vectorized across states) ──────────────────
    rng = np.random.default_rng(seed)

    null_dr2_global = np.zeros((n_perm, n_states))
    null_dr2_season = np.zeros((n_perm, n_states))
    null_dr2_within = np.zeros((n_perm, n_states))

    # Cache reduced-model R^2 that are constant across permutations:
    # - For global perm: reduced = X_season + X_within (unchanged) → r2_no_g_arr
    # - For season perm: reduced = X_global + X_within (unchanged) → r2_no_s_arr
    # - For within perm: reduced = X_global + X_season (unchanged) → r2_no_w_arr

    # Pre-allocate permutation design matrix buffers (reuse across iterations)
    X_perm_buf = X_full_int.copy()  # (n_eps, 1 + n_predictors)
    # Column indices in the intercept-augmented matrix:
    #   col 0 = intercept, col 1 = global_pos,
    #   cols 2..(2+n_season_dummies-1) = season dummies,
    #   last col = within_pos
    col_global = 1
    col_season_start = 2
    col_season_end = 2 + season_dummies.shape[1]
    col_within = col_season_end

    for pi in range(n_perm):
        # Global position permutation
        gp_perm = global_pos.copy()
        rng.shuffle(gp_perm)
        X_perm_buf[:] = X_full_int
        X_perm_buf[:, col_global] = gp_perm
        r2_full_pg = _r2_batch(X_perm_buf, fo_matrix)
        null_dr2_global[pi] = np.where(
            testable, np.maximum(0.0, r2_full_pg - r2_no_g_arr), 0.0)

        # Season permutation
        s_perm = seasons.copy()
        rng.shuffle(s_perm)
        X_perm_buf[:] = X_full_int
        for i, s in enumerate(unique_seasons[1:]):
            X_perm_buf[:, col_season_start + i] = (s_perm == s).astype(float)
        r2_full_ps = _r2_batch(X_perm_buf, fo_matrix)
        null_dr2_season[pi] = np.where(
            testable, np.maximum(0.0, r2_full_ps - r2_no_s_arr), 0.0)

        # Within-season position permutation
        wp_perm = within_pos.copy()
        for s in unique_seasons:
            mask = seasons == s
            sub = wp_perm[mask].copy()
            rng.shuffle(sub)
            wp_perm[mask] = sub
        X_perm_buf[:] = X_full_int
        X_perm_buf[:, col_within] = wp_perm
        r2_full_pw = _r2_batch(X_perm_buf, fo_matrix)
        null_dr2_within[pi] = np.where(
            testable, np.maximum(0.0, r2_full_pw - r2_no_w_arr), 0.0)

    # p-values (one-sided: ΔR^2 >= observed, NaN-safe)
    # Count only finite null values per state; NaN in null or observed → NaN p
    def _vectorized_perm_p(null_arr, obs_arr):
        """One-sided permutation p-value, NaN-safe, vectorized across states."""
        finite_null = np.isfinite(null_arr)                     # (n_perm, n_states)
        n_finite = finite_null.sum(axis=0)                      # (n_states,)
        # Replace NaN with -inf so comparison yields False (not counted)
        safe_null = np.where(finite_null, null_arr, -np.inf)
        count = np.sum(safe_null >= obs_arr, axis=0)            # (n_states,)
        p = (count + 1) / (n_finite + 1)
        # NaN where observed is NaN or no finite null values
        p = np.where(np.isfinite(obs_arr) & (n_finite > 0), p, np.nan)
        return p

    p_global = np.where(testable, _vectorized_perm_p(null_dr2_global, dr2_global), np.nan)
    p_season = np.where(testable, _vectorized_perm_p(null_dr2_season, dr2_season), np.nan)
    p_within = np.where(testable, _vectorized_perm_p(null_dr2_within, dr2_within), np.nan)

    return {
        'r2_full': r2_full_arr,
        'dr2_global': dr2_global,
        'dr2_season': dr2_season,
        'dr2_within': dr2_within,
        'shared_r2': shared_r2,
        'p_global': p_global,
        'p_season': p_season,
        'p_within': p_within,
    }


# =============================================================================
# Diagnostic 1: Motion confound
# =============================================================================

def load_median_fd(sub_id, run_ids, data_dir):
    """Load run-level median framewise displacement from fMRIPrep confound TSVs.

    Uses median FD (robust to single-spike outliers) rather than mean FD,
    following Parkes et al. (2018) and Ciric et al. (2017).

    Returns dict run_id -> float (median FD), or None if files not found.
    """
    fd_dict = {}
    missing = 0
    for run_id in run_ids:
        pattern = os.path.join(
            data_dir, 'cneuromod.processed', 'fmriprep', 'friends',
            sub_id, 'ses-*', 'func',
            f'{sub_id}_ses-*_task-{run_id}_desc-confounds_timeseries.tsv'
        )
        matches = glob.glob(pattern)
        if not matches:
            missing += 1
            continue
        try:
            df = pd.read_csv(matches[0], sep='\t', usecols=['framewise_displacement'])
            fd_dict[run_id] = float(df['framewise_displacement'].median(skipna=True))
        except Exception as e:
            logger.warning("Failed to load FD for %s: %s", run_id, e)
            missing += 1

    if missing > 0:
        logger.warning("Missing confound files for %d / %d runs", missing, len(run_ids))
    return fd_dict if fd_dict else None


def motion_confound_check(fo_dict, fd_dict, n_states):
    """Check if FD trends across runs and compute FD-controlled partial Spearman.

    Returns:
        fd_trend_rho, fd_trend_p: FD vs global run index
        rho_uncorrected: np.array(n_states,) — FO vs run index
        rho_corrected: np.array(n_states,) — FO vs run index, controlling for FD
    """
    # Align run_ids
    common_runs = sorted(set(fo_dict.keys()) & set(fd_dict.keys()),
                         key=_parse_run_order_key)
    n_runs = len(common_runs)

    if n_runs < 10:
        logger.warning("Too few runs with FD data (%d); skipping motion check", n_runs)
        return None

    global_idx = np.arange(n_runs, dtype=float)
    fd_arr = np.array([fd_dict[r] for r in common_runs])
    fo_matrix = np.array([fo_dict[r][:n_states] for r in common_runs])

    # FD trend
    fd_trend_rho, fd_trend_p = _safe_spearman(fd_arr, global_idx)

    # Uncorrected and FD-corrected Spearman per state
    rho_uncorrected = np.full(n_states, np.nan)
    rho_corrected = np.full(n_states, np.nan)

    # Partial Spearman: rank-regress both variables on FD, correlate residuals
    from scipy.stats import rankdata
    fd_ranks = rankdata(fd_arr)
    idx_ranks = rankdata(global_idx)

    # Regress index ranks on FD ranks
    lr_idx = LinearRegression().fit(fd_ranks.reshape(-1, 1), idx_ranks)
    idx_resid = idx_ranks - lr_idx.predict(fd_ranks.reshape(-1, 1))

    for k in range(n_states):
        fo_k = fo_matrix[:, k]
        rho_uncorrected[k], _ = _safe_spearman(fo_k, global_idx)

        # Partial: regress FO ranks on FD ranks, get residuals, correlate
        fo_ranks = rankdata(fo_k)
        lr_fo = LinearRegression().fit(fd_ranks.reshape(-1, 1), fo_ranks)
        fo_resid = fo_ranks - lr_fo.predict(fd_ranks.reshape(-1, 1))

        if np.std(fo_resid) > 1e-10 and np.std(idx_resid) > 1e-10:
            rho_corrected[k] = float(np.corrcoef(fo_resid, idx_resid)[0, 1])

    return {
        'n_runs_with_fd': n_runs,
        'fd_trend_rho': fd_trend_rho,
        'fd_trend_p': fd_trend_p,
        'rho_uncorrected': rho_uncorrected,
        'rho_corrected': rho_corrected,
    }


# =============================================================================
# Diagnostic 2: Anti-correlated state pairs
# =============================================================================

def state_pair_analysis(model_path, pca_path, rho_global, n_states):
    """Test whether emission-space anti-correlated states show opposite FO trends.

    Uses back-projected emission means (parcel space) for correlation, since
    PCA-space anti-correlation does not straightforwardly map to parcel-space
    network opponency. Falls back to PCA space if PCA model unavailable.

    Returns dict or None if model not loadable.
    """
    if not os.path.exists(model_path):
        logger.warning("Model file not found: %s; skipping state pair analysis", model_path)
        return None

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except (ModuleNotFoundError, ImportError) as e:
        logger.warning("Cannot load model (missing dependency: %s); skipping state pair analysis", e)
        return None

    means = model.means_  # (n_states_model, n_pcs)
    n_model = means.shape[0]
    n_use = min(n_model, n_states)

    # Back-project to parcel space if PCA model available
    space_label = 'pca'
    if pca_path and os.path.exists(pca_path):
        try:
            with open(pca_path, 'rb') as f:
                pca = pickle.load(f)
            means_parcel = pca.inverse_transform(means[:n_use])
            emission_corr = np.corrcoef(means_parcel)
            space_label = 'parcel'
            logger.info("State pair analysis: using back-projected parcel-space emission means")
        except Exception as e:
            logger.warning("Failed to back-project emissions: %s; using PCA space", e)
            emission_corr = np.corrcoef(means[:n_use])
    else:
        emission_corr = np.corrcoef(means[:n_use])
        logger.info("State pair analysis: PCA model not found; using PCA-space correlations")

    # Find anti-correlated pairs (r < -0.5)
    pairs = []
    for i in range(n_use):
        for j in range(i + 1, n_use):
            r = emission_corr[i, j]
            if r < -0.5 and np.isfinite(rho_global[i]) and np.isfinite(rho_global[j]):
                pairs.append({
                    'state_i': int(i),
                    'state_j': int(j),
                    'emission_r': float(r),
                    'trend_rho_i': float(rho_global[i]),
                    'trend_rho_j': float(rho_global[j]),
                    'trend_diff': float(rho_global[i] - rho_global[j]),
                    'opposite_trend': bool(
                        (rho_global[i] > 0 and rho_global[j] < 0) or
                        (rho_global[i] < 0 and rho_global[j] > 0)
                    ),
                })

    if not pairs:
        return {'n_pairs': 0, 'pairs': [], 'trend_vs_emission_r': np.nan,
                'emission_space': space_label}

    # Correlation of emission r with trend difference
    e_rs = [p['emission_r'] for p in pairs]
    t_diffs = [p['trend_diff'] for p in pairs]
    if len(pairs) >= 3:
        trend_vs_emission_rho, _ = _safe_spearman(e_rs, t_diffs)
    else:
        trend_vs_emission_rho = np.nan

    n_opposite = sum(1 for p in pairs if p['opposite_trend'])

    return {
        'n_pairs': len(pairs),
        'n_opposite_trend': n_opposite,
        'trend_vs_emission_r': float(trend_vs_emission_rho) if np.isfinite(trend_vs_emission_rho) else None,
        'emission_space': space_label,
        'pairs': pairs,
    }


# =============================================================================
# Plots
# =============================================================================

def plot_scale1_catalog(per_season_fo, recurrence_scores, tau_s1, q_s1,
                        sub_id, out_dir):
    """All-state catalog: FO vs season, sorted by recurrence score descending (log y-axis)."""
    n_states = len(recurrence_scores)
    seasons = sorted(per_season_fo.keys(), key=int)
    season_nums = [int(s) for s in seasons]
    fo_matrix = np.array([np.array(per_season_fo[s], dtype=float) for s in seasons])

    # Compute shared log-scale y-limits across all states
    all_fo = fo_matrix.ravel()
    nonzero_fo = all_fo[all_fo > 0]
    if len(nonzero_fo) > 0:
        global_ymin = nonzero_fo.min() * 0.5
        global_ymax = all_fo.max() * 2.0
    else:
        global_ymin, global_ymax = 1e-6, 1.0

    # Sort by recurrence score descending
    order = np.argsort(recurrence_scores)[::-1]

    n_cols = 10
    n_rows = int(np.ceil(n_states / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.2 * n_rows),
                             sharex=True, sharey=True)
    axes = np.array(axes).flatten()

    for idx, state_id in enumerate(order):
        ax = axes[idx]
        fo_k = fo_matrix[:, state_id]
        tau_k = tau_s1[state_id]
        q_k = q_s1[state_id]
        rec = recurrence_scores[state_id]

        color = recurrence_color(rec)
        # Replace zeros with floor for log scale
        fo_plot = np.where(fo_k > 0, fo_k, global_ymin)
        ax.plot(season_nums, fo_plot, 'o-', color=color, linewidth=1.2, markersize=4)

        q_str = f'q={q_k:.3f}' if np.isfinite(q_k) else 'q=NaN'
        tau_str = f'τ={tau_k:.2f}' if np.isfinite(tau_k) else 'τ=NaN'
        ax.set_title(f'S{state_id} {tau_str}\n{q_str} rec={rec:.2f}', fontsize=6)
        ax.set_xticks(season_nums)
        ax.set_yscale('log')
        ax.tick_params(labelsize=5)
        ax.grid(True, alpha=0.2)

        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Season', fontsize=6)
        if idx % n_cols == 0:
            ax.set_ylabel('Mean FO (log)', fontsize=6)

    # Set shared y-limits (only need to set on first visible axes)
    axes[0].set_ylim(global_ymin, global_ymax)

    for ax in axes[n_states:]:
        ax.set_visible(False)

    fig.suptitle(
        f'Scale 1: Cross-Season FO Profiles — All States by Recurrence (Mann-Kendall, n=6, exploratory)\n{sub_id}',
        fontsize=10, y=1.01,
    )
    fig.tight_layout()

    out_png = os.path.join(out_dir, 'scale1_all_states_cross_season.png')
    fig.savefig(out_png, bbox_inches='tight', dpi=150)
    fig.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved Scale 1 catalog: %s", out_png)


def plot_scale2_per_state(episode_fo, broadcast_meta, per_season_rho, mean_rho,
                          q_s2, recurrence_scores, sub_id, out_dir):
    """Per-state 6-panel figure: FO vs episode position within each season."""
    n_states = len(recurrence_scores)
    bcast_ids = list(broadcast_meta.keys())
    unique_seasons = sorted(set(m['season'] for m in broadcast_meta.values()))

    scale2_dir = os.path.join(out_dir, 'scale2_within_season')
    os.makedirs(scale2_dir, exist_ok=True)

    for k in range(n_states):
        fig, axes = plt.subplots(1, len(unique_seasons),
                                 figsize=(3 * len(unique_seasons), 3),
                                 sharey=True)
        if len(unique_seasons) == 1:
            axes = [axes]

        for si, s in enumerate(unique_seasons):
            ax = axes[si]
            # Get episodes in this season
            eps_s = [(b, broadcast_meta[b]) for b in bcast_ids
                     if broadcast_meta[b]['season'] == s]
            eps_s.sort(key=lambda x: x[1]['episode_num'])

            ep_positions = [m['episode_num'] for _, m in eps_s]
            fo_vals = [episode_fo[b][k] for b, _ in eps_s]

            color = recurrence_color(recurrence_scores[k])
            ax.scatter(ep_positions, fo_vals, s=20, alpha=0.7, color=color,
                       edgecolors='none')

            # Linear trend line
            if len(ep_positions) >= 2:
                z = np.polyfit(ep_positions, fo_vals, 1)
                x_line = np.linspace(min(ep_positions), max(ep_positions), 50)
                ax.plot(x_line, np.polyval(z, x_line), '--', color=color,
                        alpha=0.5, linewidth=1)

            rho_sk = per_season_rho[si, k] if np.isfinite(per_season_rho[si, k]) else float('nan')
            ax.set_title(f'Season {s}\nρ={rho_sk:.2f}', fontsize=8)
            ax.set_xlabel('Episode position', fontsize=7)
            if si == 0:
                ax.set_ylabel('FO', fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.2)

        q_str = f'q={q_s2[k]:.3f}' if np.isfinite(q_s2[k]) else 'q=NaN'
        mean_rho_str = f'mean ρ={mean_rho[k]:.3f}' if np.isfinite(mean_rho[k]) else 'mean ρ=NaN'
        fig.suptitle(
            f'State {k} — Scale 2: Within-Season Episode Trend\n'
            f'{mean_rho_str} {q_str} rec={recurrence_scores[k]:.2f} — {sub_id}',
            fontsize=9, y=1.03,
        )
        fig.tight_layout()

        out_png = os.path.join(scale2_dir, f'state_{k:03d}.png')
        fig.savefig(out_png, bbox_inches='tight', dpi=100)
        plt.close(fig)

    logger.info("Saved %d Scale 2 per-state plots to %s", n_states, scale2_dir)


def plot_scale3_variance(s3_results, recurrence_scores, sub_id, out_dir, n_states,
                         acq_time_source=''):
    """Summary stacked bar: variance decomposition per state, sorted by recurrence."""
    order = np.argsort(recurrence_scores)[::-1]

    fig, ax = plt.subplots(1, 1, figsize=(max(12, n_states * 0.3), 5))

    x = np.arange(n_states)
    bottoms = np.zeros(n_states)

    components = [
        ('dr2_global', 'Global position', '#1b9e77'),
        ('dr2_season', 'Season', '#d95f02'),
        ('dr2_within', 'Within-season pos', '#7570b3'),
        ('shared_r2', 'Shared', '#999999'),
    ]

    for key, label, color in components:
        vals = np.array([s3_results[key][order[i]] if np.isfinite(s3_results[key][order[i]]) else 0.0
                         for i in range(n_states)])
        ax.bar(x, vals, bottom=bottoms, width=0.8, label=label, color=color, alpha=0.85)
        bottoms += vals

    # Unexplained
    unexplained = np.array([
        max(0.0, 1.0 - bottoms[i]) if np.isfinite(s3_results['r2_full'][order[i]]) else 1.0
        for i in range(n_states)
    ])
    ax.bar(x, unexplained, bottom=bottoms, width=0.8, label='Unexplained',
           color='#e0e0e0', alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([str(order[i]) for i in range(n_states)], fontsize=5, rotation=90)
    ax.set_xlabel('State (sorted by recurrence score)', fontsize=8)
    ax.set_ylabel('R²', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc='upper right')
    if 'rel_acq_time' in acq_time_source:
        gp_label = 'scan date in days'
    else:
        gp_label = 'ordinal episode index'
    ax.set_title(
        f'Scale 3: Variance Partitioning — {sub_id}\n'
        f'global_position ({gp_label}) | season | within-season position',
        fontsize=9,
    )
    ax.grid(True, alpha=0.15, axis='y')
    fig.tight_layout()

    out_png = os.path.join(out_dir, 'scale3_variance_partition.png')
    fig.savefig(out_png, bbox_inches='tight', dpi=150)
    fig.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved Scale 3 variance partition: %s", out_png)


def plot_trend_vs_mean_fo(mean_fo, recurrence_scores, results_per_scale, sub_id, out_dir):
    """Multi-panel scatter: mean FO (x) vs effect size (y), colored by recurrence."""
    n_panels = len(results_per_scale)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax, res in zip(axes, results_per_scale):
        rho = res['rho']
        q = res.get('q')
        label = res['label']

        colors = [recurrence_color(s) for s in recurrence_scores]
        ax.scatter(mean_fo, rho, s=25, alpha=0.6, c=colors,
                   edgecolors='none', label='all states')

        if q is not None:
            sig = np.isfinite(q) & (q < 0.05)
            if sig.any():
                ax.scatter(
                    mean_fo[sig], rho[sig],
                    s=70, alpha=0.9,
                    c=[recurrence_color(s) for s in recurrence_scores[sig]],
                    edgecolors='black', linewidths=0.8, zorder=5,
                    label=f'FDR q<0.05 (n={sig.sum()})',
                )

        ax.axhline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Mean fractional occupancy')
        ax.set_ylabel(res.get('ylabel', 'Effect size'))
        ax.set_title(label)
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.15)

    make_recurrence_colorbar(axes[-1])
    fig.suptitle(f'Cross-Episode Temporal Trends vs State Occupancy — {sub_id}',
                 fontsize=12, y=1.01)
    fig.tight_layout()

    out_png = os.path.join(out_dir, 'trend_vs_mean_fo.png')
    fig.savefig(out_png, bbox_inches='tight', dpi=150)
    fig.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved trend vs mean FO scatter: %s", out_png)


def plot_motion_diagnostic(motion_results, recurrence_scores, sub_id, out_dir, n_states):
    """Histogram of FD correction shifts + paired lollipop of per-state rho."""
    if motion_results is None:
        return

    rho_unc = np.asarray(motion_results['rho_uncorrected'], dtype=float)
    rho_cor = np.asarray(motion_results['rho_corrected'], dtype=float)
    diff = rho_unc - rho_cor
    colors = [recurrence_color(s) for s in recurrence_scores]

    # Filter valid (non-NaN) states for both panels
    valid = np.isfinite(rho_unc) & np.isfinite(rho_cor)
    n_valid = int(np.sum(valid))

    fig_h = max(5, n_valid * 0.15 + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, fig_h),
                             gridspec_kw={'width_ratios': [1, 1.2]})

    # --- Panel A: Histogram of correction shifts ---
    ax = axes[0]
    diff_valid = diff[valid]
    ax.hist(diff_valid, bins=min(20, max(8, n_valid // 3)),
            color='#4c72b0', alpha=0.7, edgecolor='white', linewidth=0.5)
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
    med_shift = np.nanmedian(diff_valid)
    ax.axvline(med_shift, color='#c44e52', linestyle='-', linewidth=1.2, alpha=0.8)
    ax.set_xlabel('Correction shift  (ρ_uncorrected − ρ_FD-controlled)', fontsize=8)
    ax.set_ylabel('Number of states', fontsize=8)
    ax.set_title('Distribution of FD Correction Shifts', fontsize=9)
    ax.grid(True, alpha=0.15, axis='y')

    mean_abs = np.nanmean(np.abs(diff_valid))
    max_abs = np.nanmax(np.abs(diff_valid))
    fd_rho = motion_results['fd_trend_rho']
    fd_p = motion_results['fd_trend_p']
    stats_text = (f'median shift = {med_shift:.4f}\n'
                  f'mean |shift| = {mean_abs:.4f}\n'
                  f'max  |shift| = {max_abs:.4f}\n'
                  f'FD~run: ρ={fd_rho:.3f}, p={fd_p:.3f}')
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=7,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3',
            facecolor='white', edgecolor='grey', alpha=0.8))

    # --- Panel B: Paired lollipop (per-state before/after) ---
    ax = axes[1]
    valid_idx = np.where(valid)[0]
    order = valid_idx[np.argsort(rho_unc[valid_idx])[::-1]]

    for rank, idx in enumerate(order):
        c = colors[idx]
        ax.plot([rho_unc[idx], rho_cor[idx]], [rank, rank],
                color=c, linewidth=1.5, alpha=0.7, zorder=1)
        ax.scatter(rho_unc[idx], rank, s=25, color=c, marker='o',
                   edgecolors='black', linewidths=0.3, zorder=2)
        ax.scatter(rho_cor[idx], rank, s=25, color=c, marker='D',
                   edgecolors='black', linewidths=0.3, zorder=2)

    ax.axvline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('ρ(FO, run index)', fontsize=8)
    ax.set_ylabel('State (sorted by uncorrected ρ)', fontsize=8)
    ax.set_title('Per-State: Uncorrected (●) vs FD-Controlled (◆)', fontsize=9)
    ax.set_yticks([])
    ax.grid(True, alpha=0.15, axis='x')
    make_recurrence_colorbar(ax)

    fig.suptitle(f'Motion Confound Check — {sub_id}', fontsize=11, y=1.01)
    fig.tight_layout()

    out_png = os.path.join(out_dir, 'motion_confound_check.png')
    fig.savefig(out_png, bbox_inches='tight', dpi=150)
    fig.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved motion diagnostic: %s", out_png)


def plot_state_pairs(pair_results, recurrence_scores, sub_id, out_dir):
    """Diverging dumbbell plot: per-pair temporal trends for anti-correlated states."""
    if pair_results is None or pair_results['n_pairs'] == 0:
        logger.info("No anti-correlated pairs to plot.")
        return

    pairs = pair_results['pairs']
    n_total = pair_results['n_pairs']
    # Sort by |trend_diff| descending, show top 30
    pairs_sorted = sorted(pairs, key=lambda p: abs(p['trend_diff']), reverse=True)
    max_show = 30
    pairs_show = pairs_sorted[:max_show]
    n_show = len(pairs_show)

    col_opp = '#d95f02'
    col_same = '#7570b3'

    fig_h = max(4, n_show * 0.3 + 1.5)
    fig, ax = plt.subplots(1, 1, figsize=(7, fig_h))

    for rank, p in enumerate(reversed(pairs_show)):  # top = largest diff
        y = rank
        rho_i, rho_j = p['trend_rho_i'], p['trend_rho_j']
        line_color = col_opp if p['opposite_trend'] else col_same
        ax.plot([rho_i, rho_j], [y, y], color=line_color, linewidth=1.5,
                alpha=0.7, zorder=1)
        ci = recurrence_color(recurrence_scores[p['state_i']])
        cj = recurrence_color(recurrence_scores[p['state_j']])
        ax.scatter(rho_i, y, s=30, color=ci, edgecolors='black',
                   linewidths=0.4, zorder=2)
        ax.scatter(rho_j, y, s=30, color=cj, edgecolors='black',
                   linewidths=0.4, zorder=2)

    ax.axvline(0, color='black', linestyle='-', linewidth=0.6, alpha=0.4)

    # Y-axis: pair labels with emission correlation
    labels = [
        f'S{p["state_i"]}–S{p["state_j"]}  (r\u2009=\u2009{p["emission_r"]:.2f})'
        for p in reversed(pairs_show)
    ]
    ax.set_yticks(range(n_show))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Temporal trend ρ(FO, run index)')

    # Spine cleanup
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    # Summary stats
    n_opp = pair_results['n_opposite_trend']
    pct = 100 * n_opp / n_total if n_total > 0 else 0
    corr_str = ''
    if pair_results.get('trend_vs_emission_r') is not None:
        corr_str = f',  trend~emission ρ\u2009=\u2009{pair_results["trend_vs_emission_r"]:.3f}'
    ax.set_title(
        f'Anti-correlated state pairs (n\u2009=\u2009{n_total}, '
        f'{n_opp} opposite [{pct:.0f}%]{corr_str})',
    )

    cbar = make_recurrence_colorbar(ax)

    fig.tight_layout()

    # Place legend right-aligned with colorbar (after layout is finalized)
    cbar_bbox = cbar.ax.get_position()
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=col_opp, linewidth=2, label='Opposite trend'),
            plt.Line2D([0], [0], color=col_same, linewidth=2, label='Same-sign trend'),
        ],
        bbox_to_anchor=(cbar_bbox.x1, cbar_bbox.y0),
        bbox_transform=fig.transFigure,
        loc='upper right',
        frameon=True, edgecolor='none', facecolor='white', framealpha=0.8,
    )
    out_png = os.path.join(out_dir, 'state_pair_trends.png')
    fig.savefig(out_png, bbox_inches='tight')
    fig.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved state pair trends: %s", out_png)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-episode temporal trends in brain state occupancy (a2)."
    )
    parser.add_argument('--sub_id', type=str, required=True)
    parser.add_argument('--parcellation', type=str, default='atlas-4S156Parcels')
    parser.add_argument('--vt', type=str, default=None,
                        help="Variance threshold subdir (e.g. 0.95).")
    parser.add_argument('--n_perm', type=int, default=5000,
                        help="Number of permutations (default: 5000)")
    return parser.parse_args()


def main():
    args = parse_args()
    sub_id = args.sub_id
    parc = normalize_parcellation_name(args.parcellation)
    n_perm = args.n_perm

    # ── Input paths ───────────────────────────────────────────────────────────
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
    season_fo_path = os.path.join(recurrence_base, 'per_season_mean_fo.json')
    summary_path = os.path.join(recurrence_base, 'recurrence_summary.json')
    scores_path = os.path.join(recurrence_base, 'recurrence_scores.npy')
    decoded_path = os.path.join(hmm_base, 'decoded_states.pkl')
    model_path = os.path.join(hmm_base, 'best_model.pkl')
    pca_path = os.path.join(hmm_base, 'pca_model.pkl')

    for p in (fo_path, season_fo_path, summary_path, scores_path, decoded_path):
        if not os.path.exists(p):
            logger.error("Missing required input: %s", p)
            sys.exit(1)

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = os.path.join(
        SCRATCH_DIR, 'output', '05e_temporal_trend_a1', parc, sub_id
    )
    if args.vt is not None:
        out_dir = os.path.join(out_dir, f'vt{args.vt}')
    os.makedirs(out_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading data...")
    with open(fo_path, 'rb') as f:
        fo_dict = pickle.load(f)
    with open(summary_path, 'r') as f:
        recurrence_summary = json.load(f)
    with open(season_fo_path, 'r') as f:
        per_season_fo = json.load(f)
    with open(decoded_path, 'rb') as f:
        decoded_states = pickle.load(f)

    recurrence_scores = np.load(scores_path)
    n_states = recurrence_summary['n_states']
    run_ids = sorted(fo_dict.keys(), key=_parse_run_order_key)

    logger.info("n_states=%d, n_runs=%d, n_seasons=%d",
                n_states, len(fo_dict), len(per_season_fo))

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
    is_sub_hrf = np.array([k in excluded_set for k in range(n_states)])

    # ── Broadcast episode aggregation ─────────────────────────────────────────
    logger.info("Aggregating runs to broadcast episodes...")
    broadcast_episodes, broadcast_meta = group_runs_to_broadcast_episodes(run_ids)
    episode_fo, episode_n_trs = aggregate_fo_broadcast(
        fo_dict, decoded_states, broadcast_episodes
    )
    n_episodes = len(broadcast_episodes)
    logger.info("Broadcast episodes: %d (from %d runs)", n_episodes, len(run_ids))

    # Log 4-part splits
    for bid, meta in broadcast_meta.items():
        if '_ab' in bid or '_cd' in bid:
            logger.info("  4-part split: %s -> season %d, ep %d, runs %s",
                        bid, meta['season'], meta['episode_num'], meta['run_ids'])

    # ── Load acquisition times ──────────────────────────────────────────────
    acq_dict, _ = load_acquisition_times(sub_id)
    episode_acq_times = None
    acq_time_source = 'global_episode_index (ordinal proxy)'
    if acq_dict is not None:
        # Compute mean acq time per broadcast episode
        episode_acq_times = {}
        for bid, meta in broadcast_meta.items():
            times = [acq_dict[r] for r in meta['run_ids'] if r in acq_dict]
            if times:
                episode_acq_times[bid] = float(np.mean(times))
            else:
                logger.warning("No acq times for broadcast episode %s", bid)
                episode_acq_times = None
                break
        if episode_acq_times is not None:
            acq_time_source = 'rel_acq_time (days from 00_get_scan)'
            t_range = max(episode_acq_times.values()) - min(episode_acq_times.values())
            logger.info("Using real acquisition times: %.1f day span, %d episodes",
                        t_range, len(episode_acq_times))
    else:
        logger.info("No acquisition times available; using ordinal episode index")

    # ── Scale 1: Cross-season ─────────────────────────────────────────────────
    logger.info("Scale 1: cross-season Mann-Kendall (n=6, exploratory)...")
    tau_s1, p_s1 = scale1_cross_season(per_season_fo, n_states)
    q_s1 = _fdr_with_nan(p_s1)
    n_testable_s1 = int(np.sum(np.isfinite(p_s1)))
    sig_s1 = int(np.sum(np.isfinite(q_s1) & (q_s1 < 0.05)))
    logger.info("  Testable: %d / %d; FDR q<0.05: %d", n_testable_s1, n_states, sig_s1)

    # ── Scale 2: Within-season episode position ───────────────────────────────
    logger.info("Scale 2: within-season episode position (permutation, n_perm=%d)...", n_perm)
    mean_rho_s2, perm_p_s2, per_season_rho_s2 = scale2_within_season(
        episode_fo, broadcast_meta, n_states, n_perm=n_perm
    )
    q_s2 = _fdr_with_nan(perm_p_s2)
    n_testable_s2 = int(np.sum(np.isfinite(perm_p_s2)))
    sig_s2 = int(np.sum(np.isfinite(q_s2) & (q_s2 < 0.05)))
    logger.info("  Testable: %d / %d; FDR q<0.05: %d", n_testable_s2, n_states, sig_s2)

    # ── Scale 3: Variance partitioning ────────────────────────────────────────
    logger.info("Scale 3: variance partitioning (permutation, n_perm=%d)...", n_perm)
    s3_results = scale3_variance_partition(
        episode_fo, broadcast_meta, n_states,
        episode_acq_times=episode_acq_times, n_perm=n_perm
    )
    q_s3_global = _fdr_with_nan(s3_results['p_global'])
    q_s3_season = _fdr_with_nan(s3_results['p_season'])
    q_s3_within = _fdr_with_nan(s3_results['p_within'])

    sig_s3_g = int(np.sum(np.isfinite(q_s3_global) & (q_s3_global < 0.05)))
    sig_s3_s = int(np.sum(np.isfinite(q_s3_season) & (q_s3_season < 0.05)))
    sig_s3_w = int(np.sum(np.isfinite(q_s3_within) & (q_s3_within < 0.05)))
    logger.info("  Sig FDR<0.05: global=%d, season=%d, within=%d",
                sig_s3_g, sig_s3_s, sig_s3_w)
    logger.info("  Mean R²_full = %.4f (eligible states only: %.4f)",
                float(np.nanmean(s3_results['r2_full'])),
                float(np.nanmean(s3_results['r2_full'][is_eligible])))

    # ── Lag-1 autocorrelation diagnostic ────────────────────────────────────
    fo_global_matrix_chron = np.array([episode_fo[b] for b in sorted(
        broadcast_meta.keys(),
        key=lambda b: (broadcast_meta[b]['season'], broadcast_meta[b]['episode_num'])
    )])
    mean_lag1_autocorr, per_state_lag1, effective_n_approx = _lag1_autocorrelation(
        fo_global_matrix_chron
    )
    logger.info("Lag-1 autocorrelation: mean=%.3f, effective_n≈%.1f (actual n=%d)",
                mean_lag1_autocorr, effective_n_approx, len(fo_global_matrix_chron))

    # ── Diagnostic 1: Motion confound ─────────────────────────────────────────
    motion_results = None
    if DATA_DIR and os.path.isdir(DATA_DIR):
        logger.info("Diagnostic 1: motion confound check...")
        fd_dict = load_median_fd(sub_id, run_ids, DATA_DIR)
        if fd_dict:
            motion_results = motion_confound_check(fo_dict, fd_dict, n_states)
            if motion_results:
                logger.info("  FD trend: rho=%.3f p=%.3f (n=%d runs with FD)",
                            motion_results['fd_trend_rho'],
                            motion_results['fd_trend_p'],
                            motion_results['n_runs_with_fd'])
    else:
        logger.info("Diagnostic 1: DATA_DIR not set or not found; skipping motion check")

    # ── Diagnostic 2: State pairs ─────────────────────────────────────────────
    # Use Scale 2 mean_rho as the "global trend" for state pair analysis
    # (since Scale 3 is variance partition, not a single rho)
    logger.info("Diagnostic 2: anti-correlated state pair analysis...")
    # Compute simple global Spearman for state pair analysis reference
    rho_simple_global = np.full(n_states, np.nan)
    bcast_ids_sorted = sorted(broadcast_meta.keys(),
                              key=lambda b: (broadcast_meta[b]['season'],
                                             broadcast_meta[b]['episode_num']))
    # Use real acq times if available, else ordinal index
    if episode_acq_times is not None:
        global_idx = np.array([episode_acq_times[b] for b in bcast_ids_sorted], dtype=float)
    else:
        global_idx = np.arange(n_episodes, dtype=float)
    fo_global_matrix = np.array([episode_fo[b] for b in bcast_ids_sorted])
    for k in range(n_states):
        rho_simple_global[k], _ = _safe_spearman(fo_global_matrix[:, k], global_idx)

    pair_results = state_pair_analysis(model_path, pca_path, rho_simple_global, n_states)
    if pair_results:
        logger.info("  Anti-correlated pairs (r<-0.5): %d, opposite-trend: %d",
                    pair_results['n_pairs'], pair_results['n_opposite_trend'])

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, 'temporal_trend_metrics.csv')
    fieldnames = [
        'state', 'recurrence_score', 'is_eligible', 'is_sub_hrf',
        # Scale 1
        'tau_cross_season', 'p_cross_season', 'q_cross_season',
        # Scale 2
        'rho_within_season', 'p_within_season', 'q_within_season',
        # Scale 3
        'r2_full', 'dr2_global', 'p_s3_global', 'q_s3_global',
        'dr2_season', 'p_s3_season', 'q_s3_season',
        'dr2_within', 'p_s3_within', 'q_s3_within',
        'shared_r2',
        # Simple global rho
        'rho_global_simple',
    ]

    def _fmt(v):
        if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
            return ''
        return str(v)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k in range(n_states):
            writer.writerow({
                'state': k,
                'recurrence_score': _fmt(float(recurrence_scores[k])),
                'is_eligible': bool(is_eligible[k]),
                'is_sub_hrf': bool(is_sub_hrf[k]),
                'tau_cross_season': _fmt(tau_s1[k]),
                'p_cross_season': _fmt(p_s1[k]),
                'q_cross_season': _fmt(q_s1[k]),
                'rho_within_season': _fmt(mean_rho_s2[k]),
                'p_within_season': _fmt(perm_p_s2[k]),
                'q_within_season': _fmt(q_s2[k]),
                'r2_full': _fmt(s3_results['r2_full'][k]),
                'dr2_global': _fmt(s3_results['dr2_global'][k]),
                'p_s3_global': _fmt(s3_results['p_global'][k]),
                'q_s3_global': _fmt(q_s3_global[k]),
                'dr2_season': _fmt(s3_results['dr2_season'][k]),
                'p_s3_season': _fmt(s3_results['p_season'][k]),
                'q_s3_season': _fmt(q_s3_season[k]),
                'dr2_within': _fmt(s3_results['dr2_within'][k]),
                'p_s3_within': _fmt(s3_results['p_within'][k]),
                'q_s3_within': _fmt(q_s3_within[k]),
                'shared_r2': _fmt(s3_results['shared_r2'][k]),
                'rho_global_simple': _fmt(rho_simple_global[k]),
            })
    logger.info("Saved metrics CSV: %s", csv_path)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    def _arr(v):
        return [float(x) if np.isfinite(x) else None for x in v]

    unique_seasons = sorted(set(m['season'] for m in broadcast_meta.values()))
    runs_per_season = {}
    for s in unique_seasons:
        runs_per_season[int(s)] = sum(
            1 for m in broadcast_meta.values() if m['season'] == s
        )

    results = {
        'analysis_scope': 'single_subject',
        'analysis_type': 'cross_episode_temporal_trends_v2',
        'sub_id': sub_id,
        'parcellation': parc,
        'n_states': n_states,
        'n_runs': len(run_ids),
        'n_broadcast_episodes': n_episodes,
        'n_seasons': len(unique_seasons),
        'seasons': unique_seasons,
        'episodes_per_season': runs_per_season,
        'n_eligible_states': int(is_eligible.sum()),
        'n_sub_hrf_states': int(is_sub_hrf.sum()),
        'n_permutations': n_perm,
        'mean_lag1_autocorr': float(mean_lag1_autocorr) if np.isfinite(mean_lag1_autocorr) else None,
        'effective_n_approx': float(effective_n_approx) if np.isfinite(effective_n_approx) else None,
        'lag1_note': (
            'Mean lag-1 autocorrelation of run-level FO across chronological episodes. '
            'Effective N approximated via Bayley & Hammersley (1946): '
            'n_eff ≈ n * (1 - r1) / (1 + r1). Values near 0 indicate minimal '
            'temporal dependence; permutation p-values are only mildly liberal.'
        ),
        'fdr_method': 'Benjamini-Hochberg per scale/predictor',
        'broadcast_episode_rule': (
            'Standard episodes: a+b -> one broadcast episode. '
            '4-part episodes (s04e23, s05e23, s06e15, s06e24): '
            'a+b -> episode N, c+d -> episode N+1.'
        ),
        'note': (
            'Scale 1 (cross-season): only 6 data points — purely descriptive. '
            'FDR q-values reported for completeness but should NOT be interpreted as '
            'confirmatory evidence; minimum detectable |tau| at n=6 is extreme. '
            'Session-order confound dominates cross-season FO structure in this dataset '
            '(05c episode decodability: order/season accuracy ratio 0.92-1.02 across subjects). '
            'Scale 2 uses permutation tests (not parametric). '
            'Scale 3 variance partition is a statistical decomposition, NOT a causal '
            'attribution. Global_position (real acquisition times when available), '
            'season, and within-season position are structurally confounded '
            '(episodes watched in order): unique variance estimates reflect what '
            'survives after partialing, not the true causal contribution of each factor. '
            'Unchecked confounds: time-of-day, within-session fatigue, physio state. '
            'Content-controlled analysis deferred to stage 08.'
        ),
        'scale1_cross_season': {
            'description': (
                'Mann-Kendall tau of mean_FO_per_season vs season_number (n=6). '
                'PURELY DESCRIPTIVE — do not treat any result as confirmatory.'
            ),
            'n_points': len(unique_seasons),
            'n_testable_states': n_testable_s1,
            'n_sig_fdr05': sig_s1,
            'tau': _arr(tau_s1),
            'p': _arr(p_s1),
            'q': _arr(q_s1),
        },
        'scale2_within_season': {
            'description': (
                'Spearman rho per season (episode_FO vs episode_position), '
                'combined via permutation test on mean_rho'
            ),
            'n_episodes_total': n_episodes,
            'inference': 'permutation (episode-position shuffle within season)',
            'n_testable_states': n_testable_s2,
            'n_sig_fdr05': sig_s2,
            'mean_rho': _arr(mean_rho_s2),
            'p': _arr(perm_p_s2),
            'q': _arr(q_s2),
        },
        'scale3_variance_partition': {
            'description': (
                f'Semi-partial R^2: global_position ({acq_time_source}), '
                'season (categorical), within-season position (ordinal)'
            ),
            'global_position_source': acq_time_source,
            'caveat': (
                'Descriptive decomposition only. Predictors are structurally confounded '
                '(episodes watched chronologically: global_pos = f(season, within_pos)). '
                'Unique variance estimates reflect what survives after partialing, not '
                'true causal contributions. Large shared + small unique = predictors are '
                'indistinguishable without real scanning dates or cross-stimulus control.'
            ),
            'n_episodes': n_episodes,
            'inference': 'permutation (predictor-specific shuffle)',
            'n_sig_fdr05_global': sig_s3_g,
            'n_sig_fdr05_season': sig_s3_s,
            'n_sig_fdr05_within': sig_s3_w,
            'r2_full': _arr(s3_results['r2_full']),
            'dr2_global': _arr(s3_results['dr2_global']),
            'dr2_season': _arr(s3_results['dr2_season']),
            'dr2_within': _arr(s3_results['dr2_within']),
            'shared_r2': _arr(s3_results['shared_r2']),
            'p_global': _arr(s3_results['p_global']),
            'q_global': _arr(q_s3_global),
            'p_season': _arr(s3_results['p_season']),
            'q_season': _arr(q_s3_season),
            'p_within': _arr(s3_results['p_within']),
            'q_within': _arr(q_s3_within),
        },
        'rho_global_simple': _arr(rho_simple_global),
    }

    # Add diagnostics
    if motion_results:
        results['motion_confound'] = {
            'fd_metric': 'median_fd (robust to outlier spikes; Parkes et al. 2018)',
            'n_runs_with_fd': motion_results['n_runs_with_fd'],
            'fd_trend_rho': motion_results['fd_trend_rho'],
            'fd_trend_p': motion_results['fd_trend_p'],
            'rho_uncorrected': _arr(motion_results['rho_uncorrected']),
            'rho_corrected': _arr(motion_results['rho_corrected']),
        }

    if pair_results:
        # Don't save full emission_corr_matrix (large), just summary
        results['state_pair_analysis'] = {
            'n_anti_correlated_pairs': pair_results['n_pairs'],
            'n_opposite_trend': pair_results['n_opposite_trend'],
            'trend_vs_emission_r': pair_results.get('trend_vs_emission_r'),
            'pairs': pair_results['pairs'][:20],  # Top 20 for JSON size
        }

    json_path = os.path.join(out_dir, 'temporal_trend_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info("Saved JSON results: %s", json_path)

    # ── Plots ─────────────────────────────────────────────────────────────────
    logger.info("Generating plots...")

    plot_scale1_catalog(per_season_fo, recurrence_scores, tau_s1, q_s1,
                        sub_id, out_dir)

    plot_scale2_per_state(episode_fo, broadcast_meta, per_season_rho_s2,
                          mean_rho_s2, q_s2, recurrence_scores, sub_id, out_dir)

    plot_scale3_variance(s3_results, recurrence_scores, sub_id, out_dir, n_states,
                         acq_time_source=acq_time_source)

    # Compute mean FO per state for trend plot x-axis
    bcast_ids_all = list(episode_fo.keys())
    mean_fo = np.mean([episode_fo[b] for b in bcast_ids_all], axis=0)  # (n_states,)

    # Multi-scale trend scatter (x = mean FO, color = recurrence)
    results_per_scale = [
        {'rho': tau_s1, 'q': q_s1,
         'label': 'Scale 1: Cross-season\n(Kendall τ, n=6, exploratory)',
         'ylabel': 'Kendall τ'},
        {'rho': mean_rho_s2, 'q': q_s2,
         'label': 'Scale 2: Within-season position\n(mean ρ, permutation)',
         'ylabel': 'Spearman ρ'},
        {'rho': rho_simple_global, 'q': None,
         'label': 'Global episode trend\n(simple Spearman ρ)',
         'ylabel': 'Spearman ρ'},
    ]
    plot_trend_vs_mean_fo(mean_fo, recurrence_scores, results_per_scale, sub_id, out_dir)

    plot_motion_diagnostic(motion_results, recurrence_scores, sub_id, out_dir, n_states)
    plot_state_pairs(pair_results, recurrence_scores, sub_id, out_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Cross-Episode Temporal Trend Analysis v2 Complete")
    logger.info("=" * 60)
    logger.info("Scale 1 (cross-season, Mann-Kendall)     sig FDR<0.05: %d / %d", sig_s1, n_states)
    logger.info("Scale 2 (within-season, permutation)     sig FDR<0.05: %d / %d", sig_s2, n_states)
    logger.info("Scale 3 unique global_pos                sig FDR<0.05: %d / %d", sig_s3_g, n_states)
    logger.info("Scale 3 unique season                    sig FDR<0.05: %d / %d", sig_s3_s, n_states)
    logger.info("Scale 3 unique within-season             sig FDR<0.05: %d / %d", sig_s3_w, n_states)
    logger.info("Output directory: %s", out_dir)


if __name__ == '__main__':
    main()
