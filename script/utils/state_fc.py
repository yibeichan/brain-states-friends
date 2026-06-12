#!/usr/bin/env python3
"""
state_fc.py - Empirical within-state functional connectivity utilities.

Provides reusable functions for computing state-conditioned FC from BOLD
timeseries and decoded HMM state assignments.  Extracted from 05f_state_fc.py
so that the same logic can be reused by reliability analyses (04rc, 04rv).

Core workflow:
    1. load_matched_data()        – align parcel TS with decoded state sequences
    2. compute_empirical_state_fc() – Ledoit-Wolf shrinkage covariance → correlation
    3. compute_delta_correlation() – occupancy-weighted ΔR_k = R_k - R_grand
    4. compute_fc_similarity()     – pairwise FC similarity for matched state pairs
"""

import os
import re
import logging
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


# ── Data loading ──────────────────────────────────────────────────────────────


def load_matched_data(sub_id, parcellation, decoded_states, n_expected_parcels,
                      scratch_dir=None):
    """Load parcel timeseries matched to decoded_states runs.

    Args:
        sub_id:             Subject ID (e.g. 'sub-01').
        parcellation:       Full parcellation name (e.g. 'atlas-4S156Parcels').
        decoded_states:     dict  run_id → state sequence array.
        n_expected_parcels: Number of atlas parcels (for background column stripping).
        scratch_dir:        Root scratch directory.  Falls back to SCRATCH_DIR env var.

    Returns:
        parcel_ts_concat: (T_total, n_parcels)
        viterbi_concat:   (T_total,)
        n_runs:           number of matched runs
    """
    if scratch_dir is None:
        scratch_dir = os.environ.get("SCRATCH_DIR")
    if scratch_dir is None:
        raise ValueError("scratch_dir must be provided or SCRATCH_DIR set in env")

    ts_dir = os.path.join(
        scratch_dir, "output", "02_parcel_ts_avg", parcellation, sub_id,
    )

    # Index parcel TS files by task entity
    ts_files = sorted(Path(ts_dir).glob("*_parcel_avg.npy"))
    task_re = re.compile(r"task-([^\s_]+)")
    ts_by_task = {}
    for f in ts_files:
        m = task_re.search(f.name)
        if m:
            ts_by_task[m.group(1)] = f

    parcel_chunks = []
    viterbi_chunks = []
    n_matched = 0

    for run_id in sorted(decoded_states.keys()):
        ts_path = ts_by_task.get(run_id)
        if ts_path is None:
            logger.warning("No parcel TS for run %s", run_id)
            continue

        ts = np.load(ts_path)
        vit = np.asarray(decoded_states[run_id])

        n = min(len(ts), len(vit))
        if n == 0:
            continue

        parcel_chunks.append(ts[:n])
        viterbi_chunks.append(vit[:n])
        n_matched += 1

    if n_matched == 0:
        raise FileNotFoundError("No matched parcel/Viterbi run pairs found")

    parcel_ts = np.vstack(parcel_chunks)
    viterbi = np.concatenate(viterbi_chunks)

    # Strip background column if present
    if parcel_ts.shape[1] == n_expected_parcels + 1:
        logger.info(
            "Stripping background column 0: (%d, %d) -> (%d, %d)",
            parcel_ts.shape[0], parcel_ts.shape[1],
            parcel_ts.shape[0], n_expected_parcels,
        )
        parcel_ts = parcel_ts[:, 1:]
    elif parcel_ts.shape[1] != n_expected_parcels:
        raise ValueError(
            f"Parcel TS has {parcel_ts.shape[1]} columns but atlas has "
            f"{n_expected_parcels} parcels (expected {n_expected_parcels} or "
            f"{n_expected_parcels + 1})"
        )

    logger.info(
        "Loaded %d runs: %d TRs x %d parcels",
        n_matched, parcel_ts.shape[0], parcel_ts.shape[1],
    )
    return parcel_ts, viterbi, n_matched


# ── Empirical FC ──────────────────────────────────────────────────────────────


def compute_empirical_state_fc(parcel_ts, viterbi, K, min_trs=30):
    """Compute per-state correlation matrices using Ledoit-Wolf shrinkage.

    For each state k, all TRs assigned to that state are pooled and a
    shrinkage covariance is estimated, then converted to Pearson correlation.

    Args:
        parcel_ts: (T_total, n_parcels) concatenated timeseries.
        viterbi:   (T_total,) state assignments.
        K:         Number of HMM states (including inactive).
        min_trs:   Minimum TRs for reliable FC estimation.

    Returns:
        corr_parcel:      (K, n_parcels, n_parcels) correlation matrices.
        n_trs_per_state:  (K,) TR counts.
        reliable:         (K,) boolean - True if n_trs >= min_trs.
        shrinkage_alpha:  (K,) Ledoit-Wolf shrinkage intensity per state.
    """
    n_parcels = parcel_ts.shape[1]
    corr_parcel = np.zeros((K, n_parcels, n_parcels))
    n_trs_per_state = np.zeros(K, dtype=int)
    shrinkage_alpha = np.full(K, np.nan)

    for k in range(K):
        mask = viterbi == k
        n_k = mask.sum()
        n_trs_per_state[k] = n_k

        if n_k < 2:
            corr_parcel[k] = np.eye(n_parcels)
            continue

        X_k = parcel_ts[mask]

        if n_k < min_trs:
            logger.warning("State %d: only %d TRs (< %d), FC unreliable", k, n_k, min_trs)

        lw = LedoitWolf()
        lw.fit(X_k)
        cov_k = lw.covariance_
        shrinkage_alpha[k] = lw.shrinkage_

        # Convert to correlation
        diag_std = np.sqrt(np.diag(cov_k))
        diag_std[diag_std < 1e-10] = 1.0
        corr_parcel[k] = cov_k / np.outer(diag_std, diag_std)

    reliable = n_trs_per_state >= min_trs
    logger.info("Empirical FC: %d/%d states have >= %d TRs", reliable.sum(), K, min_trs)
    return corr_parcel, n_trs_per_state, reliable, shrinkage_alpha


def compute_delta_correlation(corr_parcel, occupancies):
    """Compute ΔR_k = R_k - R_grand (occupancy-weighted grand mean).

    Args:
        corr_parcel: (K, n_parcels, n_parcels) state correlation matrices.
        occupancies: (K,) weights summing to 1.

    Returns:
        delta_R: (K, n_parcels, n_parcels)
        R_grand: (n_parcels, n_parcels)
    """
    R_grand = np.tensordot(occupancies, corr_parcel, axes=([0], [0]))
    delta_R = corr_parcel - R_grand[np.newaxis, :, :]
    return delta_R, R_grand


# ── FC similarity for matched state pairs ─────────────────────────────────────


def compute_fc_similarity_pairs(corr_a, corr_b, pairs):
    """Compute FC similarity for Hungarian-matched state pairs using RV coefficient.

    The RV coefficient is the standard metric for comparing symmetric matrices
    in neuroimaging.  RV(A, B) = tr(A B) / sqrt(tr(A A) tr(B B)), giving a
    scale-invariant measure of structural similarity in [0, 1].

    Args:
        corr_a: (K_a, n_parcels, n_parcels) FC from model A.
        corr_b: (K_b, n_parcels, n_parcels) FC from model B.
        pairs:  list of dicts with 'state_A' and 'state_B' keys (indices).

    Returns:
        fc_similarities: list of float - per-pair RV coefficient.
    """
    from utils.stats import compute_rv_coefficient

    fc_similarities = []
    for pair in pairs:
        sa, sb = pair["state_A"], pair["state_B"]
        stacked = np.stack([corr_a[sa], corr_b[sb]])  # (2, p, p)
        rv_mat = compute_rv_coefficient(stacked)
        fc_similarities.append(float(rv_mat[0, 1]))

    return fc_similarities
