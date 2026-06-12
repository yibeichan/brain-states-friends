#!/usr/bin/env python3
"""
04ra_loso_struct_comp.py - Compare structural invariants across LOSO folds.

Each Leave-One-Season-Out fold fits an independent HMM on N-1 seasons.
This script loads the LOSO results and compares structural properties across
folds, establishing an initialization-sensitivity baseline from the primary
model's seed variability.

Note: sub-04 has only 4 seasons. Available folds are auto-detected from disk.

Prerequisites:
    - 04_combined_hdphmm.py (mode: loso_fit) completed for available seasons
    - 04_combined_hdphmm.py (mode: select) completed (primary model)

Outputs (saved to {SCRATCH_DIR}/output/04ra_loso_struct_comp/{parcellation}/{sub_id}/):
    - fold_invariants.json     per-fold scalar metrics
    - cross_fold_consistency.json  CV, range, pairwise KS stats
    - hungarian_matching.json  matched-pair correlation distribution
    - noise_floor.json         seed-to-seed variability of primary model
"""

import os
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy.stats import entropy, ks_2samp
from scipy.optimize import linear_sum_assignment

# Add script directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils.recurrence_utils import compute_fractional_occupancy, compute_recurrence_scores
from utils.common import normalize_parcellation_name
from utils.plot_style import NETWORK_ORDER, assign_network

from dotenv import load_dotenv

load_dotenv()
SCRATCH_DIR = os.getenv('SCRATCH_DIR')
if SCRATCH_DIR is None:
    raise ValueError("SCRATCH_DIR must be set in the .env file")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


# =============================================================================
# Path helpers
# =============================================================================

def get_hmm_output_base(sub_id, parcellation):
    """Root output directory for 04_combined_hdphmm results."""
    return os.path.join(
        SCRATCH_DIR, 'output', '04_combined_hdphmm', parcellation, sub_id
    )


def get_comparison_output_dir(sub_id, parcellation):
    """Output directory for this script's results."""
    return os.path.join(
        SCRATCH_DIR, 'output', '04ra_loso_struct_comp', parcellation, sub_id
    )


def get_final_dir(output_base, vt):
    """Final output directory for a given variance threshold."""
    return os.path.join(output_base, 'final', f'vt{vt}')


# =============================================================================
# Per-fold metrics
# =============================================================================

def identify_active_states(decoded_states, n_total_states, min_state_usage=0.01):
    """Identify active states from decoded sequences.

    Returns:
        active_mask: boolean array (n_total_states,)
        active_indices: integer array of active state indices
    """
    if not decoded_states:
        return np.zeros(n_total_states, dtype=bool), np.array([], dtype=int)

    non_empty = [np.asarray(seq) for seq in decoded_states.values() if len(seq) > 0]
    if not non_empty:
        return np.zeros(n_total_states, dtype=bool), np.array([], dtype=int)

    all_states = np.concatenate(non_empty)
    total_trs = len(all_states)
    counts = np.bincount(all_states, minlength=n_total_states)
    occupancy = counts / total_trs
    active_mask = occupancy > min_state_usage
    active_indices = np.where(active_mask)[0]
    return active_mask, active_indices


def compute_transition_entropy(transmat, active_indices):
    """Compute mean normalized transition entropy over active-state rows.

    For each active state, compute H(row) / log(K_active).
    Returns mean normalized entropy.

    Note: This computes entropy **conditional on the active subspace** - rows
    are subsetted to active states and re-normalized. This measures "how random
    are transitions among active states?" rather than full HMM entropy. The
    subspace restriction is intentional: inactive states have near-zero
    transition probability and would artificially inflate entropy.
    """
    k_active = len(active_indices)
    if k_active <= 1:
        return 0.0

    log_k = np.log(k_active)
    entropies = []
    for i in active_indices:
        row = transmat[i, active_indices]
        # Re-normalize within active subspace
        row = row / row.sum() if row.sum() > 0 else np.ones(k_active) / k_active
        h = entropy(row)
        entropies.append(h / log_k)

    return float(np.mean(entropies))


def compute_self_transition_prob(transmat, active_indices):
    """Mean diagonal of transmat_ for active states."""
    if len(active_indices) == 0:
        return 0.0
    diag_vals = [transmat[i, i] for i in active_indices]
    return float(np.mean(diag_vals))


def compute_dwell_times(decoded_states):
    """Compute dwell times (in TRs) from decoded state sequences.

    Returns:
        all_dwells: list of dwell durations across all runs
    """
    all_dwells = []
    for state_seq in decoded_states.values():
        if len(state_seq) == 0:
            continue
        # Find contiguous blocks
        diffs = np.diff(state_seq)
        change_points = np.where(diffs != 0)[0] + 1
        boundaries = np.concatenate([[0], change_points, [len(state_seq)]])
        for j in range(len(boundaries) - 1):
            all_dwells.append(boundaries[j + 1] - boundaries[j])
    return all_dwells


def compute_network_composition(state_means, active_indices, parcel_labels):
    """For each active state, assign to network of max-|activation| parcel.

    Args:
        state_means: (n_states, n_parcels) array
        active_indices: indices of active states
        parcel_labels: list of parcel label strings (length n_parcels)

    Returns:
        composition: dict network_name -> count of states assigned
        assignments: list of (state_idx, parcel_idx, network_name) tuples

    Note:
        This is a coarse descriptive summary based on the single largest-magnitude
        parcel. It does not preserve sign or distributed topography and should
        not be treated as a full biological equivalence test.
    """
    composition = {net: 0 for net in NETWORK_ORDER}
    assignments = []

    for s in active_indices:
        pattern = state_means[s]
        max_parcel_idx = int(np.argmax(np.abs(pattern)))
        label = parcel_labels[max_parcel_idx]

        # Try subcortical assignment first
        network = assign_network(label)
        if network is None:
            # Cortical: extract network from Schaefer label (e.g. "LH_Vis_1")
            parts = label.split('_')
            if len(parts) >= 2:
                network = parts[1]  # e.g. "Vis", "SomMot", "Default"
            else:
                network = 'Unknown'

        if network in composition:
            composition[network] += 1
        assignments.append((int(s), max_parcel_idx, network))

    return composition, assignments


def load_parcel_labels(parcellation):
    """Load parcel labels from the atlas TSV file via viz_yabplot.

    Returns:
        labels: list of parcel label strings
    """
    from utils.viz_yabplot import load_parcel_labels as _load_labels
    label_df = _load_labels(parcellation)
    return list(label_df['label'])


def compute_fold_invariants(fold_dir, n_total_states, fo_threshold, parcel_labels):
    """Compute all structural invariants for one LOSO fold.

    Args:
        fold_dir: path to loso/season_X/ directory
        n_total_states: total number of model states
        fo_threshold: threshold for fractional occupancy
        parcel_labels: list of parcel label strings

    Returns:
        dict of scalar and distributional metrics
    """
    # Load decoded states
    decoded_path = os.path.join(fold_dir, 'decoded_states.pkl')
    with open(decoded_path, 'rb') as f:
        decoded_states = pickle.load(f)

    # Load model
    model_path = os.path.join(fold_dir, 'best_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load state means
    means_path = os.path.join(fold_dir, 'state_means_parcel.npy')
    state_means = np.load(means_path)

    # Load loso_results.json for K_active
    results_path = os.path.join(fold_dir, 'loso_results.json')
    with open(results_path, 'r') as f:
        loso_results = json.load(f)
    k_active_reported = loso_results['loso_refit']['n_active_states']

    # Identify active states from decoded sequences
    active_mask, active_indices = identify_active_states(
        decoded_states, n_total_states
    )
    k_active = len(active_indices)

    # Recurrence scores
    fo = compute_fractional_occupancy(decoded_states, n_total_states)
    recurrence = compute_recurrence_scores(fo, n_total_states, fo_threshold)

    # Transition entropy (normalized)
    transmat = model.transmat_
    trans_entropy = compute_transition_entropy(transmat, active_indices)

    # Self-transition probability
    self_trans = compute_self_transition_prob(transmat, active_indices)

    # Dwell times
    dwells = compute_dwell_times(decoded_states)
    if dwells:
        dwell_arr = np.array(dwells)
        dwell_median = float(np.median(dwell_arr))
        dwell_q25 = float(np.percentile(dwell_arr, 25))
        dwell_q75 = float(np.percentile(dwell_arr, 75))
    else:
        dwell_median = float('nan')
        dwell_q25 = float('nan')
        dwell_q75 = float('nan')

    # Network composition
    composition, assignments = compute_network_composition(
        state_means, active_indices, parcel_labels
    )

    # Recurrence distribution for active states only
    recurrence_active = recurrence[active_indices]

    invariants = {
        'k_active': k_active,
        'k_active_reported': k_active_reported,
        'transition_entropy': trans_entropy,
        'self_transition_prob': self_trans,
        'dwell_median_tr': dwell_median,
        'dwell_iqr': (dwell_q25, dwell_q75),
        'network_composition': composition,
        'recurrence_scores_active': recurrence_active.tolist(),
        'recurrence_mean': float(np.mean(recurrence_active)) if len(recurrence_active) > 0 else float('nan'),
        'recurrence_median': float(np.median(recurrence_active)) if len(recurrence_active) > 0 else float('nan'),
        'n_decoded_runs': len(decoded_states),
    }
    return invariants, state_means, active_indices


# =============================================================================
# Initialization sensitivity: seed-to-seed variability of primary model
#
# This measures how much the EM solution varies due to random initialization
# (same data, different starting points). It is NOT a noise floor for
# fold-to-fold variability, which reflects different data subsets and may
# include genuine seasonal differences. Compare the two to assess whether
# fold variability exceeds initialization sensitivity.
# =============================================================================

def extract_active_indices_from_history(model, seed_idx=None, min_state_usage=0.01):
    """Extract active-state indices from stored training usage history.

    Returns array of active state indices, or empty array if history unavailable.
    """
    seed_label = f"seed {seed_idx}" if seed_idx is not None else "model"
    if not hasattr(model, 'history') or not model.history:
        logger.debug("%s: no history attribute, cannot extract active states", seed_label)
        return np.array([], dtype=int)
    usage_hist = model.history.get('state_usage')
    if not usage_hist:
        logger.debug("%s: no state_usage in history", seed_label)
        return np.array([], dtype=int)
    usage = np.asarray(usage_hist[-1], dtype=float)
    if usage.size == 0 or not np.all(np.isfinite(usage)):
        logger.debug("%s: state_usage is empty or contains non-finite values", seed_label)
        return np.array([], dtype=int)
    return np.where(usage > min_state_usage)[0]


def compute_noise_floor(seeds_dir, n_seeds=10, min_state_usage=0.01):
    """Compute K_active and transition entropy for each primary-model seed.

    Args:
        seeds_dir: path to final/vt{vt}/seeds/
        n_seeds: number of seeds to check
        min_state_usage: threshold for active state identification

    Returns:
        dict with seed-level K_active and entropy arrays
    """
    seed_k_active = []
    seed_entropy = []
    seed_self_trans = []
    seeds_loaded = []

    for i in range(n_seeds):
        seed_json = os.path.join(seeds_dir, f'seed_{i}.json')
        seed_model_path = os.path.join(seeds_dir, f'seed_{i}_model.pkl')

        if not os.path.exists(seed_json):
            continue
        with open(seed_json, 'r') as f:
            res = json.load(f)
        if res.get('status') != 'success':
            continue
        if not os.path.exists(seed_model_path):
            continue

        with open(seed_model_path, 'rb') as f:
            model = pickle.load(f)

        active_indices = extract_active_indices_from_history(
            model, seed_idx=i, min_state_usage=min_state_usage
        )
        if len(active_indices) == 0:
            logger.warning(
                "Skipping seed %d: no usable state_usage history.", i
            )
            continue

        k_active = len(active_indices)
        seed_k_active.append(k_active)

        transmat = model.transmat_
        h = compute_transition_entropy(transmat, active_indices)
        seed_entropy.append(h)

        st = compute_self_transition_prob(transmat, active_indices)
        seed_self_trans.append(st)

        seeds_loaded.append(i)

    if not seeds_loaded:
        logger.warning("No valid primary-model seeds found for noise floor.")
        return {
            'n_seeds_loaded': 0,
            'k_active': {'values': [], 'mean': None, 'std': None, 'range': None},
            'transition_entropy': {'values': [], 'mean': None, 'std': None, 'range': None},
            'self_transition_prob': {'values': [], 'mean': None, 'std': None, 'range': None},
        }

    k_arr = np.array(seed_k_active)
    h_arr = np.array(seed_entropy)
    st_arr = np.array(seed_self_trans)

    return {
        'n_seeds_loaded': len(seeds_loaded),
        'seeds_used': seeds_loaded,
        'k_active': {
            'values': seed_k_active,
            'mean': float(np.mean(k_arr)),
            'std': float(np.std(k_arr)),
            'range': [int(np.min(k_arr)), int(np.max(k_arr))],
        },
        'transition_entropy': {
            'values': [float(x) for x in seed_entropy],
            'mean': float(np.mean(h_arr)),
            'std': float(np.std(h_arr)),
            'range': [float(np.min(h_arr)), float(np.max(h_arr))],
        },
        'self_transition_prob': {
            'values': [float(x) for x in seed_self_trans],
            'mean': float(np.mean(st_arr)),
            'std': float(np.std(st_arr)),
            'range': [float(np.min(st_arr)), float(np.max(st_arr))],
        },
    }


# =============================================================================
# Cross-fold consistency
# =============================================================================

def compute_cross_fold_consistency(fold_invariants):
    """Compute cross-fold consistency metrics.

    Args:
        fold_invariants: dict season -> fold invariant dict

    Returns:
        dict with CV, range, and pairwise KS stats
    """
    seasons = sorted(fold_invariants.keys())
    n_folds = len(seasons)

    # Scalar invariants to compare
    # CV is only meaningful for unbounded positive metrics; for bounded
    # metrics (transition_entropy, self_transition_prob, recurrence_mean)
    # we report std and range instead.
    scalar_keys = ['k_active', 'transition_entropy', 'self_transition_prob',
                   'dwell_median_tr', 'recurrence_mean']
    bounded_keys = {'transition_entropy', 'self_transition_prob', 'recurrence_mean'}
    scalar_stats = {}

    for key in scalar_keys:
        values = [fold_invariants[s][key] for s in seasons]
        arr = np.array(values)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            scalar_stats[key] = {
                'per_fold': {s: float(v) for s, v in zip(seasons, values)},
                'mean': float('nan'),
                'std': float('nan'),
                'cv': float('nan'),
                'range': [float('nan'), float('nan')],
            }
            continue

        mean_val = float(np.mean(finite))
        std_val = float(np.std(finite))
        if key in bounded_keys:
            # CV is misleading for bounded [0,1] metrics; report None
            cv = None
        elif np.isnan(mean_val):
            cv = float('nan')
        elif mean_val != 0:
            cv = float(std_val / mean_val)
        else:
            cv = float('inf')
        scalar_stats[key] = {
            'per_fold': {s: float(v) for s, v in zip(seasons, values)},
            'mean': mean_val,
            'std': std_val,
            'cv': cv,
            'range': [float(np.nanmin(arr)), float(np.nanmax(arr))],
        }

    # Pairwise KS test on sorted recurrence distributions
    ks_results = []
    for (s1, s2) in combinations(seasons, 2):
        rec1 = np.sort(fold_invariants[s1]['recurrence_scores_active'])
        rec2 = np.sort(fold_invariants[s2]['recurrence_scores_active'])
        stat, pval = ks_2samp(rec1, rec2)
        ks_results.append({
            'fold_pair': [s1, s2],
            'ks_statistic': float(stat),
            'p_value': float(pval),
        })

    ks_stats = [r['ks_statistic'] for r in ks_results]
    ks_pvals = [r['p_value'] for r in ks_results]
    ks_pvals_fdr = benjamini_hochberg(ks_pvals)
    for result, p_fdr in zip(ks_results, ks_pvals_fdr):
        result['p_value_fdr_bh'] = float(p_fdr)

    return {
        'n_folds': n_folds,
        'scalar_invariants': scalar_stats,
        'recurrence_ks_tests': {
            'pairwise': ks_results,
            'n_pairs': len(ks_results),
            'mean_ks_statistic': float(np.mean(ks_stats)) if ks_stats else None,
            'median_p_value': float(np.median(ks_pvals)) if ks_pvals else None,
            'median_p_value_fdr_bh': float(np.median(ks_pvals_fdr)) if ks_pvals_fdr else None,
            'n_significant_005': sum(1 for p in ks_pvals if p < 0.05),
            'n_significant_fdr_bh_005': sum(1 for p in ks_pvals_fdr if p < 0.05),
        },
    }


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg FDR correction."""
    if not p_values:
        return []

    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)

    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for idx in range(n - 1, -1, -1):
        rank = idx + 1
        value = (ranked[idx] * n) / rank
        prev = min(prev, value)
        adjusted[idx] = min(prev, 1.0)

    corrected = np.empty(n, dtype=float)
    corrected[order] = adjusted
    return corrected.tolist()


# =============================================================================
# Hungarian matching
# =============================================================================

def hungarian_match_fold_to_primary(fold_means, primary_means,
                                    fold_active, primary_active,
                                    match_threshold=0.3):
    """Match fold states to primary model states via Hungarian algorithm.

    Args:
        fold_means: (n_fold_states, n_parcels) state means
        primary_means: (n_primary_states, n_parcels) state means
        fold_active: indices of active states in fold
        primary_active: indices of active states in primary
        match_threshold: minimum correlation for a good match

    Returns:
        dict with matching results
    """
    if len(fold_active) == 0 or len(primary_active) == 0:
        return {
            'n_fold_active': len(fold_active),
            'n_primary_active': len(primary_active),
            'n_matched': 0,
            'n_well_matched': 0,
            'fraction_well_matched': 0.0,
            'matched_correlations': [],
            'correlation_mean': None,
            'correlation_median': None,
            'correlation_std': None,
            'matches': [],
            'match_threshold': match_threshold,
        }

    # Extract active-state means
    fold_sub = fold_means[fold_active]
    primary_sub = primary_means[primary_active]

    # Compute correlation matrix between fold active states and primary active states
    # Shape: (n_fold_active, n_primary_active)
    n_f = len(fold_active)
    n_p = len(primary_active)
    fold_centered = fold_sub - fold_sub.mean(axis=1, keepdims=True)
    fold_std = fold_sub.std(axis=1, keepdims=True)
    fold_std = np.where(fold_std > 0, fold_std, 1.0)
    fold_z = fold_centered / fold_std

    primary_centered = primary_sub - primary_sub.mean(axis=1, keepdims=True)
    primary_std = primary_sub.std(axis=1, keepdims=True)
    primary_std = np.where(primary_std > 0, primary_std, 1.0)
    primary_z = primary_centered / primary_std

    corr_matrix = (fold_z @ primary_z.T) / fold_z.shape[1]
    corr_matrix = np.nan_to_num(corr_matrix, nan=-1.0, posinf=-1.0, neginf=-1.0)

    # Cost matrix: 1 - correlation (lower = better match)
    cost_matrix = 1.0 - corr_matrix

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Evaluate matches
    matches = []
    matched_corrs = []
    for r, c in zip(row_ind, col_ind):
        corr_val = float(corr_matrix[r, c])
        matched_corrs.append(corr_val)
        matches.append({
            'fold_state': int(fold_active[r]),
            'primary_state': int(primary_active[c]),
            'correlation': corr_val,
            'well_matched': corr_val >= match_threshold,
        })

    n_well = sum(1 for m in matches if m['well_matched'])
    n_matched = len(matches)

    return {
        'n_fold_active': n_f,
        'n_primary_active': n_p,
        'n_matched': n_matched,
        'n_well_matched': n_well,
        'fraction_well_matched': float(n_well / n_matched) if n_matched > 0 else 0.0,
        'matched_correlations': matched_corrs,
        'correlation_mean': float(np.mean(matched_corrs)) if matched_corrs else None,
        'correlation_median': float(np.median(matched_corrs)) if matched_corrs else None,
        'correlation_std': float(np.std(matched_corrs)) if matched_corrs else None,
        'matches': matches,
        'match_threshold': match_threshold,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare structural invariants across LOSO HMM folds.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--sub_id', required=True,
                        help='Subject ID (e.g., sub-01)')
    parser.add_argument('--parcellation', default='atlas-4S156Parcels',
                        help='Parcellation name (default: atlas-4S156Parcels)')
    parser.add_argument('--vt', type=str, default='0.95',
                        help='Variance threshold subdirectory under final/ '
                             '(default: 0.95)')
    parser.add_argument('--fo_threshold', type=float, default=0.01,
                        help='Fractional occupancy threshold for recurrence (default: 0.01)')
    parser.add_argument('--match_threshold', type=float, default=0.3,
                        help='Minimum correlation for well-matched states (default: 0.3)')
    args = parser.parse_args()

    sub_id = args.sub_id
    parcellation = normalize_parcellation_name(args.parcellation)
    vt = args.vt
    fo_threshold = args.fo_threshold
    match_threshold = args.match_threshold

    logger.info("=" * 60)
    logger.info(f"LOSO Structural Comparison: {sub_id}, {parcellation}")
    logger.info("=" * 60)

    output_base = get_hmm_output_base(sub_id, parcellation)
    final_dir = get_final_dir(output_base, vt)
    logger.info(f"Variance threshold: vt={vt}")
    logger.info(f"Primary final dir: {final_dir}")

    if not os.path.isdir(final_dir):
        logger.error(f"Primary final dir not found: {final_dir}")
        logger.error("Run 04_combined_hdphmm.py --mode select first.")
        sys.exit(1)

    # Output directory
    out_dir = get_comparison_output_dir(sub_id, parcellation)
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")

    # Load parcel labels for network composition
    parcel_labels = load_parcel_labels(parcellation)
    logger.info(f"Loaded {len(parcel_labels)} parcel labels")

    # ── Load primary model for Hungarian matching ────────────────────────
    primary_results_path = os.path.join(final_dir, 'final_results.json')
    with open(primary_results_path, 'r') as f:
        primary_results = json.load(f)

    primary_means_path = os.path.join(final_dir, 'state_means_parcel.npy')
    primary_means = np.load(primary_means_path)

    primary_decoded_path = os.path.join(final_dir, 'decoded_states.pkl')
    with open(primary_decoded_path, 'rb') as f:
        primary_decoded = pickle.load(f)

    n_primary_total = primary_means.shape[0]
    primary_active_mask, primary_active_indices = identify_active_states(
        primary_decoded, n_primary_total
    )
    logger.info(
        f"Primary model: {n_primary_total} total states, "
        f"{len(primary_active_indices)} active"
    )

    # ── Detect available LOSO folds ─────────────────────────────────────
    loso_base = os.path.join(output_base, 'loso')
    available_seasons = sorted([
        int(d.name.split('_')[1])
        for d in Path(loso_base).glob('season_*')
        if (d / 'loso_results.json').exists()
    ]) if os.path.isdir(loso_base) else []

    if not available_seasons:
        logger.error(f"No completed LOSO folds found in {loso_base}")
        logger.error("Run 04_combined_hdphmm.py --mode loso_fit first.")
        sys.exit(1)

    logger.info(f"Found {len(available_seasons)} completed LOSO folds: "
                f"seasons {available_seasons}")

    # ── Per-fold invariants ──────────────────────────────────────────────
    fold_invariants = {}
    fold_means_cache = {}  # for Hungarian matching
    fold_active_cache = {}

    for season in available_seasons:
        fold_dir = os.path.join(output_base, 'loso', f'season_{season}')
        results_path = os.path.join(fold_dir, 'loso_results.json')

        logger.info(f"\n--- Season {season} (held out) ---")

        with open(results_path, 'r') as f:
            loso_json = json.load(f)
        n_total_states = loso_json['loso_refit']['n_total_states']

        inv, fold_means, fold_active_idx = compute_fold_invariants(
            fold_dir, n_total_states, fo_threshold, parcel_labels
        )
        inv['held_out_season'] = season
        fold_invariants[season] = inv

        # Cache for Hungarian matching (returned from compute_fold_invariants)
        fold_means_cache[season] = fold_means
        fold_active_cache[season] = fold_active_idx

        logger.info(
            f"  K_active={inv['k_active']}, "
            f"H_trans={inv['transition_entropy']:.3f}, "
            f"P_self={inv['self_transition_prob']:.3f}, "
            f"dwell_median={inv['dwell_median_tr']:.1f} TR"
        )

    if not fold_invariants:
        logger.error("No LOSO folds found. Run loso_fit first.")
        sys.exit(1)

    # Save fold invariants
    # Convert tuples to lists for JSON serialization
    fold_inv_serializable = {}
    for s, inv in fold_invariants.items():
        inv_copy = dict(inv)
        inv_copy['dwell_iqr'] = list(inv_copy['dwell_iqr'])
        fold_inv_serializable[str(s)] = inv_copy

    fold_inv_path = os.path.join(out_dir, 'fold_invariants.json')
    with open(fold_inv_path, 'w') as f:
        json.dump(fold_inv_serializable, f, indent=2)
    logger.info(f"\nSaved fold invariants to {fold_inv_path}")

    # ── Initialization sensitivity (seed-to-seed variability) ───────────
    logger.info("\n--- Initialization Sensitivity (primary model seeds) ---")
    seeds_dir = os.path.join(final_dir, 'seeds')
    noise_floor = compute_noise_floor(seeds_dir, n_seeds=10)

    noise_floor['caveat'] = (
        'Seed-to-seed variability measures initialization sensitivity (same data, '
        'different EM starting points). Fold-to-fold variability reflects different '
        'data subsets and may include genuine seasonal differences. Fold variance '
        'exceeding seed variance does not necessarily indicate instability.'
    )
    noise_path = os.path.join(out_dir, 'noise_floor.json')
    with open(noise_path, 'w') as f:
        json.dump(noise_floor, f, indent=2)
    logger.info(
        f"  Seeds loaded: {noise_floor['n_seeds_loaded']}"
    )
    if noise_floor['k_active']['mean'] is not None:
        logger.info(
            f"  K_active: {noise_floor['k_active']['mean']:.1f} "
            f"+/- {noise_floor['k_active']['std']:.1f} "
            f"(range {noise_floor['k_active']['range']})"
        )
        logger.info(
            f"  H_trans:  {noise_floor['transition_entropy']['mean']:.3f} "
            f"+/- {noise_floor['transition_entropy']['std']:.3f}"
        )
    logger.info(f"Saved noise floor to {noise_path}")

    # ── Cross-fold consistency ───────────────────────────────────────────
    logger.info("\n--- Cross-Fold Consistency ---")
    consistency = compute_cross_fold_consistency(fold_invariants)

    for key in ['k_active', 'transition_entropy', 'self_transition_prob',
                'dwell_median_tr', 'recurrence_mean']:
        stats = consistency['scalar_invariants'][key]
        cv_str = f"CV={stats['cv']:.3f}" if stats['cv'] is not None else "CV=N/A (bounded)"
        logger.info(
            f"  {key}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
            f"{cv_str}, "
            f"range=[{stats['range'][0]:.3f}, {stats['range'][1]:.3f}]"
        )

    ks_info = consistency['recurrence_ks_tests']
    logger.info(
        f"  Recurrence KS: mean_stat={ks_info['mean_ks_statistic']:.3f}, "
        f"median_p={ks_info['median_p_value']:.3f}, "
        f"n_sig(0.05)={ks_info['n_significant_005']}/{ks_info['n_pairs']}"
    )

    consistency_path = os.path.join(out_dir, 'cross_fold_consistency.json')
    with open(consistency_path, 'w') as f:
        json.dump(consistency, f, indent=2)
    logger.info(f"Saved cross-fold consistency to {consistency_path}")

    # ── Hungarian matching ───────────────────────────────────────────────
    logger.info("\n--- Hungarian Matching (fold -> primary) ---")
    matching_results = {}

    for season in sorted(fold_invariants.keys()):
        match = hungarian_match_fold_to_primary(
            fold_means=fold_means_cache[season],
            primary_means=primary_means,
            fold_active=fold_active_cache[season],
            primary_active=primary_active_indices,
            match_threshold=match_threshold,
        )
        matching_results[str(season)] = match
        logger.info(
            f"  Season {season}: {match['n_well_matched']}/{match['n_matched']} "
            f"well-matched (r >= {match_threshold}), "
            f"mean r={match['correlation_mean']:.3f}"
            if match['correlation_mean'] is not None else
            f"mean r=NA"
        )

    # Summary across folds
    all_fracs = [
        matching_results[s]['fraction_well_matched']
        for s in matching_results
    ]
    all_mean_corrs = [
        matching_results[s]['correlation_mean']
        for s in matching_results
        if matching_results[s]['correlation_mean'] is not None
    ]
    matching_summary = {
        'per_fold': matching_results,
        'summary': {
            'mean_fraction_well_matched': float(np.mean(all_fracs)) if all_fracs else None,
            'mean_correlation_across_folds': float(np.mean(all_mean_corrs)) if all_mean_corrs else None,
            'match_threshold': match_threshold,
            'caveats': [
                'Matching uses parcel-space state means only; diagonal covariance '
                'structure is not compared. Covariance-based matching is in 05d/05f.',
                'Each LOSO fold has its own PCA (different n_pcs). Back-projected '
                'parcel-space means come from different reconstruction bases, '
                'introducing systematic differences unrelated to state content.',
                'Network composition is descriptive only (single argmax of '
                '|activation|) and does not preserve sign or distributed topography.',
                'Dwell times are from Viterbi decoding, which upper-bounds true '
                'dwell durations (blockier state assignments than posterior marginals).',
            ],
        },
    }

    matching_path = os.path.join(out_dir, 'hungarian_matching.json')
    with open(matching_path, 'w') as f:
        json.dump(matching_summary, f, indent=2)
    logger.info(f"Saved Hungarian matching to {matching_path}")

    # ── Final summary ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("LOSO Structural Comparison Complete")
    logger.info(f"  Folds analyzed: {len(fold_invariants)}")
    logger.info(f"  Output: {out_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
