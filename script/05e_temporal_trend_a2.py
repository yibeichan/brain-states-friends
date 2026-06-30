#!/usr/bin/env python3
"""
05e_temporal_trend_a2.py - Detect temporally-anchored brain states.

Investigates whether states cluster at specific temporal positions within
scanner runs, using the a/b suffix design to disentangle a-specific from
shared run-onset effects.

Episode structure reminder
--------------------------
a/b/c/d suffixes are SCANNER SESSION BOUNDARIES, not narrative splits.
Most episodes are split into two ~11-min scans:
  a = first half of episode (contains theme song at variable position)
  b = second half of episode

Only 4 episodes have 4-part splits (c/d).  Analyses are run separately for
each suffix type; "a" and "b" runs are the primary comparison.

Approach
--------
1. Re-extract contiguous state blocks for ALL states (not just recurring/
   partial) from decoded_states.pkl.
2. For each state, compute normalized block-onset position within each run
   (start_tr / episode_length_tr).
3. Compute per-state position metrics: mean position, IQR, early-fraction
   (fraction of blocks in the first 20% of the run), split by suffix type.
4. Permutation test (block-level within-run shuffle, 2000 permutations):
   tests whether early_fraction is significantly higher than chance,
   separately for "a" and "b" suffixes.
5. Separate BH FDR correction per suffix family.
6. Conjunction classification with neutral labels:
   a_start_specific (sig a, not b), ab_start_common (sig both),
   b_start_specific (sig b, not a), none.

This script reports structural observations only.  Causal interpretation
(theme song? arousal reset? context reinstatement?) is deferred to
downstream analyses (08 series).

Prerequisites:
    - 04_combined_hdphmm.py (mode: select) completed for this subject
    - 05a_recurrence_analysis.py completed for this subject

Outputs (saved to {SCRATCH_DIR}/output/05e_temporal_trend_a2/{parc}/{sub}/):
    - temporal_position_metrics.csv         per-state summary table
    - temporal_position_analysis.json       full results + FDR + flagged states
    - ab_early_fraction_scatter.png/pdf
    - position_cdf_flagged_states.png/pdf
    - early_fraction_bar_chart.png/pdf
"""

import argparse
import json
import logging
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.stats import benjamini_hochberg, fdr_with_nan as _fdr_with_nan
from utils.plot_style import recurrence_color, make_recurrence_colorbar, apply_publication_style
from utils.common import normalize_parcellation_name, _get_season
from utils.state_blocks import (
    TR_SECONDS,
    extract_state_block_records,
    load_eligible_states,
    write_records_csv,
)

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

# Default position threshold for "early" block classification (first 20% of run)
DEFAULT_EARLY_THRESHOLD = 0.20

# Minimum blocks required to compute position statistics per suffix
MIN_BLOCKS_FOR_STATS = 3

POSITION_METRIC_FIELDNAMES = [
    'state',
    'recurrence_score',
    'n_blocks_all',
    'n_blocks_a',
    'n_blocks_b',
    'mean_position_a',
    'mean_position_b',
    'position_iqr_a',
    'position_iqr_b',
    'early_fraction_a',
    'early_fraction_b',
    'p_early_a_perm',
    'q_early_a_fdr',
    'p_early_b_perm',
    'q_early_b_fdr',
    'n_runs_a',
    'n_runs_b',
    'position_locked_a',
    'position_locked_b',
    'theme_fraction_a',
    'delta_early',
    'anchoring_type',
]


# =============================================================================
# Core temporal position functions
# =============================================================================

def get_run_suffix(run_id):
    """Extract the suffix character from a run_id (e.g. 's01e01a' -> 'a')."""
    return run_id[-1]


def compute_position_metrics(block_records, early_threshold=DEFAULT_EARLY_THRESHOLD):
    """Compute per-state normalized position metrics from block records.

    Args:
        block_records: list of dicts from extract_state_block_records()
        early_threshold: fraction of run to consider "early" (default 0.20)

    Returns:
        metrics: dict state_id -> dict of position statistics
    """
    # Collect normalized positions and run info per state, split by suffix
    positions_by_state = defaultdict(lambda: {
        'all': [],
        'a': [],
        'b': [],
        'other': [],
        'runs_a': set(),
        'runs_b': set(),
    })

    for record in block_records:
        state_id = int(record['state'])
        ep_len = int(record['episode_length_tr'])
        start_tr = int(record['start_tr'])
        if ep_len == 0:
            continue

        pos = start_tr / ep_len
        suffix = get_run_suffix(record['run_id'])

        positions_by_state[state_id]['all'].append(pos)
        if suffix == 'a':
            positions_by_state[state_id]['a'].append(pos)
            positions_by_state[state_id]['runs_a'].add(record['run_id'])
        elif suffix == 'b':
            positions_by_state[state_id]['b'].append(pos)
            positions_by_state[state_id]['runs_b'].add(record['run_id'])
        else:
            positions_by_state[state_id]['other'].append(pos)

    metrics = {}
    for state_id, data in positions_by_state.items():
        pos_all = np.array(data['all'], dtype=float)
        pos_a = np.array(data['a'], dtype=float)
        pos_b = np.array(data['b'], dtype=float)

        def _stats(arr):
            if len(arr) == 0 or np.all(np.isnan(arr)):
                return None, None, 0.0
            arr_clean = arr[np.isfinite(arr)]
            if len(arr_clean) == 0:
                return None, None, 0.0
            mean = float(np.mean(arr_clean))
            iqr = float(np.percentile(arr_clean, 75) - np.percentile(arr_clean, 25))
            early = float(np.mean(arr_clean < early_threshold))
            return mean, iqr, early

        mean_all, iqr_all, _ = _stats(pos_all)
        mean_a, iqr_a, early_a = _stats(pos_a)
        mean_b, iqr_b, early_b = _stats(pos_b)

        metrics[state_id] = {
            'n_blocks_all': len(pos_all),
            'n_blocks_a': len(pos_a),
            'n_blocks_b': len(pos_b),
            'mean_position_all': mean_all,
            'mean_position_a': mean_a,
            'mean_position_b': mean_b,
            'position_iqr_all': iqr_all,
            'position_iqr_a': iqr_a,
            'position_iqr_b': iqr_b,
            'early_fraction_a': early_a,
            'early_fraction_b': early_b,
            'n_runs_a': len(data['runs_a']),
            'n_runs_b': len(data['runs_b']),
        }

    return metrics


def build_run_block_arrays(decoded_states):
    """Build per-run arrays of (block_durations, block_states) for permutation.

    Returns:
        run_data: dict run_id -> {'durations': np.array, 'states': np.array,
                                  'episode_length_tr': int, 'suffix': str}
    """
    run_data = {}
    for run_id, state_seq in decoded_states.items():
        state_seq = np.asarray(state_seq, dtype=int)
        ep_len = len(state_seq)
        if ep_len == 0:
            continue

        change_points = np.flatnonzero(state_seq[1:] != state_seq[:-1]) + 1
        starts = np.concatenate(([0], change_points))
        ends = np.concatenate((change_points, [ep_len]))
        states = state_seq[starts].astype(int)
        durations = (ends - starts).astype(int)

        run_data[run_id] = {
            'durations': durations,
            'states': states,
            'episode_length_tr': ep_len,
            'suffix': get_run_suffix(run_id),
        }

    return run_data


def compute_observed_early_fractions(run_data, n_states, suffix='a',
                                     early_threshold=DEFAULT_EARLY_THRESHOLD):
    """Compute observed early_fraction per state for a given run suffix.

    Args:
        run_data: dict from build_run_block_arrays()
        n_states: total number of states
        suffix: run suffix to filter ('a' or 'b')
        early_threshold: normalized position cutoff (default 0.20)

    Returns:
        early_frac: np.array(n_states,) - fraction of blocks in early window
        block_counts: np.array(n_states,) - total block count for this suffix
    """
    early_counts = np.zeros(n_states, dtype=float)
    total_counts = np.zeros(n_states, dtype=float)

    for run_id, data in run_data.items():
        if data['suffix'] != suffix:
            continue
        ep_len = data['episode_length_tr']
        starts = np.concatenate(([0], np.cumsum(data['durations'][:-1])))
        positions = starts / ep_len
        states = data['states']

        np.add.at(total_counts, states, 1)
        early_mask = positions < early_threshold
        np.add.at(early_counts, states[early_mask], 1)

    early_frac = np.where(
        total_counts > 0, early_counts / total_counts, np.nan
    )
    return early_frac, total_counts


def permutation_test_early_fraction(
    run_data,
    n_states,
    observed_early_frac,
    observed_block_counts,
    suffix='a',
    n_permutations=2000,
    seed=42,
    min_blocks=MIN_BLOCKS_FOR_STATS,
    early_threshold=DEFAULT_EARLY_THRESHOLD,
):
    """Block-level within-run permutation test for early_fraction.

    Null: shuffle the order of blocks within each run independently (preserves
    block counts and dwell times, randomises temporal position).

    p-value: fraction of permutations where permuted early_fraction >= observed
    (one-sided, testing for early anchoring).
    Uses (count + 1) / (n_perm + 1) correction (Phipson & Smyth 2010).

    Args:
        run_data: dict from build_run_block_arrays()
        n_states: total number of states
        observed_early_frac: observed early fractions from compute_observed_early_fractions()
        observed_block_counts: block counts from compute_observed_early_fractions()
        suffix: run suffix to test ('a' or 'b')
        n_permutations: number of permutations (default 2000)
        seed: random seed
        min_blocks: minimum blocks required (default 3)
        early_threshold: normalized position cutoff (default 0.20)

    Returns:
        p_values: np.array(n_states,) - NaN for states with too few blocks
    """
    rng = np.random.default_rng(seed)
    perm_greater = np.zeros(n_states, dtype=float)

    logger.info(
        "Permutation test (suffix=%s): %d permutations...", suffix, n_permutations
    )

    # Cache the runs of interest
    suffix_runs = {
        run_id: data
        for run_id, data in run_data.items()
        if data['suffix'] == suffix
    }

    for i in range(n_permutations):
        if i > 0 and i % 500 == 0:
            logger.info("  %d/%d permutations done", i, n_permutations)

        perm_early = np.zeros(n_states, dtype=float)
        perm_total = np.zeros(n_states, dtype=float)

        for run_id, data in suffix_runs.items():
            ep_len = data['episode_length_tr']
            n_blocks = len(data['durations'])
            if n_blocks <= 1:
                # Single-block run: position is always 0; skip permuting
                state_id = int(data['states'][0])
                perm_total[state_id] += 1
                # position = 0 / ep_len = 0 < early_threshold
                perm_early[state_id] += 1
                continue

            perm_order = rng.permutation(n_blocks)
            perm_durations = data['durations'][perm_order]
            perm_states = data['states'][perm_order]
            perm_starts = np.concatenate(([0], np.cumsum(perm_durations[:-1])))
            perm_positions = perm_starts / ep_len

            np.add.at(perm_total, perm_states, 1)
            early_mask = perm_positions < early_threshold
            np.add.at(perm_early, perm_states[early_mask], 1)

        perm_ef = np.where(
            perm_total > 0, perm_early / perm_total, np.nan
        )
        # One-sided: permuted >= observed (testing for early anchoring)
        perm_greater += np.where(
            np.isnan(perm_ef) | np.isnan(observed_early_frac),
            0.0,
            (perm_ef >= observed_early_frac).astype(float),
        )

    # Phipson & Smyth (2010) finite-sampling correction
    p_values = (perm_greater + 1) / (n_permutations + 1)

    # Set NaN for states with insufficient blocks
    insufficient = (observed_block_counts < min_blocks) | np.isnan(observed_early_frac)
    p_values[insufficient] = np.nan

    return p_values


def compute_theme_fraction(block_records, threshold_tr=33):
    """Compute fraction of 'a'-run blocks starting before threshold_tr.

    This is a descriptive metric only - theme song position varies across
    episodes due to cold opens, so this fixed window is an approximate
    lower bound.

    Args:
        block_records: list of dicts from extract_state_block_records()
        threshold_tr: absolute TR cutoff (default 33 ≈ 49s at TR=1.49s)

    Returns:
        theme_frac: dict state_id -> float (fraction of a-run blocks
                    with start_tr < threshold_tr)
    """
    early_counts = defaultdict(int)
    total_counts = defaultdict(int)

    for record in block_records:
        suffix = get_run_suffix(record['run_id'])
        if suffix != 'a':
            continue
        state_id = int(record['state'])
        total_counts[state_id] += 1
        if int(record['start_tr']) < threshold_tr:
            early_counts[state_id] += 1

    theme_frac = {}
    for state_id in total_counts:
        theme_frac[state_id] = early_counts[state_id] / total_counts[state_id]

    return theme_frac


def classify_anchoring_type(q_early_a, q_early_b, fdr_threshold=0.10):
    """Classify states by conjunction of per-suffix significance.

    Labels are neutral/descriptive - no causal mechanism assumed.

    Args:
        q_early_a: np.array of FDR-corrected q-values for suffix a
        q_early_b: np.array of FDR-corrected q-values for suffix b
        fdr_threshold: significance threshold (default 0.10)

    Returns:
        anchoring_types: list of str, one per state
    """
    n_states = len(q_early_a)
    anchoring_types = []
    for i in range(n_states):
        sig_a = (not np.isnan(q_early_a[i])) and (q_early_a[i] < fdr_threshold)
        sig_b = (not np.isnan(q_early_b[i])) and (q_early_b[i] < fdr_threshold)

        if sig_a and sig_b:
            anchoring_types.append('ab_start_common')
        elif sig_a and not sig_b:
            anchoring_types.append('a_start_specific')
        elif sig_b and not sig_a:
            anchoring_types.append('b_start_specific')
        else:
            anchoring_types.append('none')

    return anchoring_types


def transition_confound_check(model_path, position_locked_states, n_states,
                               transmat_threshold=0.15, excluded_sub_hrf=None):
    """Flag states whose position-locking may be inherited via transitions.

    Two checks:
    1. **Secondary anchoring:** Non-locked states that may appear early because
       a locked state has high transition probability to them.
    2. **Sub-HRF feeders:** Sub-HRF states (excluded from permutation testing)
       that have high transition probability to anchored states. These short-
       lived states may be the "trigger" that initiates anchored-state activation,
       meaning the anchored state's early position is partly inherited.

    Args:
        model_path: path to best_model.pkl (HMM with transmat_ attribute)
        position_locked_states: set of state IDs flagged as position-locked
        n_states: total number of states
        transmat_threshold: minimum transition probability to flag (default 0.15)
        excluded_sub_hrf: set of sub-HRF state IDs (default None)

    Returns:
        dict with 'secondary_states', 'sub_hrf_feeders', and metadata, or None.
    """
    if not os.path.exists(model_path):
        return None

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        transmat = model.transmat_[:n_states, :n_states]
    except Exception as e:
        logger.warning("Cannot load model for transition confound check: %s", e)
        return None

    if excluded_sub_hrf is None:
        excluded_sub_hrf = set()

    # 1. Secondary anchoring: locked → non-locked
    secondary = []
    if position_locked_states:
        for target in range(n_states):
            if target in position_locked_states:
                continue
            for source in position_locked_states:
                if source >= transmat.shape[0] or target >= transmat.shape[1]:
                    continue
                p_trans = float(transmat[source, target])
                if p_trans >= transmat_threshold:
                    secondary.append({
                        'state': int(target),
                        'source_locked_state': int(source),
                        'transition_prob': round(p_trans, 4),
                    })

    # 2. Sub-HRF feeders: sub-HRF → anchored
    sub_hrf_feeders = []
    if excluded_sub_hrf and position_locked_states:
        for target in position_locked_states:
            if target >= transmat.shape[1]:
                continue
            for source in excluded_sub_hrf:
                if source >= transmat.shape[0]:
                    continue
                p_trans = float(transmat[source, target])
                if p_trans >= transmat_threshold:
                    sub_hrf_feeders.append({
                        'anchored_state': int(target),
                        'sub_hrf_source': int(source),
                        'transition_prob': round(p_trans, 4),
                    })

    return {
        'n_secondary': len(secondary),
        'transmat_threshold': transmat_threshold,
        'secondary_states': secondary,
        'n_sub_hrf_feeders': len(sub_hrf_feeders),
        'sub_hrf_feeders': sub_hrf_feeders,
        'note': (
            'secondary_states: non-locked states that may appear early via '
            'transitions from locked states. sub_hrf_feeders: excluded sub-HRF '
            'states with high transition probability to anchored states - these '
            'short-lived states may trigger anchored-state activation.'
        ),
    }


# =============================================================================
# Plots
# =============================================================================

def plot_ab_early_fraction_scatter(
    state_rows,
    out_dir,
    sub_id,
    early_threshold=DEFAULT_EARLY_THRESHOLD,
    fdr_threshold=0.10,
):
    """Scatter: early_fraction_a vs early_fraction_b.

    Diagonal = equal anchoring (ab_start_common zone).
    Below diagonal = a-start-specific candidates.
    Above diagonal = b-start-specific candidates.
    """
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    rows_with_data = [
        r for r in state_rows
        if r['early_fraction_a'] is not None and r['early_fraction_b'] is not None
    ]
    if not rows_with_data:
        logger.warning("No data for A/B scatter; skipping.")
        plt.close(fig)
        return None

    ef_a = [r['early_fraction_a'] for r in rows_with_data]
    ef_b = [r['early_fraction_b'] for r in rows_with_data]
    scores = [r['recurrence_score'] for r in rows_with_data]
    colors = [recurrence_color(s) for s in scores]

    ax.scatter(
        ef_a, ef_b,
        s=50, alpha=0.8, c=colors, edgecolors='black', linewidths=0.4,
    )
    make_recurrence_colorbar(ax)

    # Diagonal (equal anchoring)
    ax.plot([0, 1], [0, 1], color='grey', linestyle='-', linewidth=0.8, alpha=0.5,
            label='Equal anchoring')

    # Reference lines at uniform null
    ax.axvline(early_threshold, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(early_threshold, color='grey', linestyle='--', linewidth=0.8, alpha=0.5,
               label=f'Uniform null ({early_threshold:.0%})')

    # Mark significant states
    sig_rows = [
        r for r in rows_with_data
        if r.get('anchoring_type', 'none') != 'none'
    ]
    if sig_rows:
        ax.scatter(
            [r['early_fraction_a'] for r in sig_rows],
            [r['early_fraction_b'] for r in sig_rows],
            s=100, facecolors='none', edgecolors='red', linewidths=1.5, zorder=5,
            label=f'Position-anchored (FDR q<{fdr_threshold}, n={len(sig_rows)})',
        )
        for r in sig_rows:
            ax.annotate(
                f"  s{r['state']}",
                (r['early_fraction_a'], r['early_fraction_b']),
                fontsize=7, alpha=0.85,
                xytext=(4, 0), textcoords='offset points',
            )

    ax.set_xlim(-0.02, max(0.5, max(ef_a) * 1.1))
    ax.set_ylim(-0.02, max(0.5, max(ef_b) * 1.1))
    ax.set_xlabel('Early fraction in "a" runs\n(fraction of blocks in first 20%)')
    ax.set_ylabel('Early fraction in "b" runs\n(fraction of blocks in first 20%)')
    ax.set_title(f'Position anchoring: "a" vs "b" runs\n{sub_id}')
    ax.grid(True, alpha=0.20)
    ax.legend(loc='upper left', fontsize=8, ncol=1)

    out_png = os.path.join(out_dir, 'ab_early_fraction_scatter.png')
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    fig.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved A/B early fraction scatter: %s", out_png)
    return out_png


def plot_position_cdfs(
    block_records,
    state_rows,
    out_dir,
    sub_id,
    max_states=10,
):
    """Empirical CDF of block-onset positions for flagged states.

    One subplot per position-anchored state. Lines for "a" (blue) and
    "b" (orange) runs. Diagonal = uniform null.
    """
    flagged = [
        r for r in state_rows
        if r.get('anchoring_type', 'none') != 'none'
    ]
    if not flagged:
        logger.warning("No flagged states for CDF plot; skipping.")
        return None

    # Sort by delta_early descending, take top N
    flagged = sorted(flagged, key=lambda r: abs(r.get('delta_early', 0) or 0),
                     reverse=True)[:max_states]

    # Collect positions per state per suffix
    positions = defaultdict(lambda: {'a': [], 'b': []})
    for record in block_records:
        state_id = int(record['state'])
        ep_len = int(record['episode_length_tr'])
        if ep_len == 0:
            continue
        pos = int(record['start_tr']) / ep_len
        suffix = get_run_suffix(record['run_id'])
        if suffix in ('a', 'b'):
            positions[state_id][suffix].append(pos)

    n_plots = len(flagged)
    ncols = min(4, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.5 * nrows),
                             squeeze=False)

    for idx, row in enumerate(flagged):
        ax = axes[idx // ncols, idx % ncols]
        state_id = row['state']

        for suffix, color, label in [('a', '#0072B2', '"a" runs'),
                                      ('b', '#E69F00', '"b" runs')]:
            pos_arr = np.sort(positions[state_id][suffix])
            if len(pos_arr) == 0:
                continue
            ecdf_y = np.arange(1, len(pos_arr) + 1) / len(pos_arr)
            ax.step(pos_arr, ecdf_y, where='post', color=color, linewidth=1.5,
                    label=f'{label} (n={len(pos_arr)})')

        # Uniform null diagonal
        ax.plot([0, 1], [0, 1], color='grey', linestyle=':', linewidth=0.8,
                alpha=0.5)

        atype = row.get('anchoring_type', 'none')
        ax.set_title(f's{state_id} [{atype}]', fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.15)
        if idx // ncols == nrows - 1:
            ax.set_xlabel('Normalized position')
        if idx % ncols == 0:
            ax.set_ylabel('Cumulative fraction')

    # Hide unused axes
    for idx in range(n_plots, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(f'Position CDFs - flagged states\n{sub_id}', fontsize=11)
    fig.tight_layout()

    out_png = os.path.join(out_dir, 'position_cdf_flagged_states.png')
    fig.savefig(out_png, bbox_inches='tight')
    fig.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved position CDF plot: %s", out_png)
    return out_png


def plot_early_fraction_bars(
    state_rows,
    out_dir,
    sub_id,
    early_threshold=DEFAULT_EARLY_THRESHOLD,
    fdr_threshold=0.10,
    min_blocks=5,
):
    """Horizontal grouped bars: early_fraction_a and early_fraction_b per state.

    States with fewer than *min_blocks* total blocks are excluded (early
    fractions from 1–2 blocks are meaningless noise).  Remaining states are
    sorted by max(ef_a, ef_b) descending so the most position-anchored states
    appear at the top.
    """
    rows_with_data = [
        r for r in state_rows
        if (r['early_fraction_a'] is not None or r['early_fraction_b'] is not None)
        and (r.get('n_blocks_a', 0) or 0) + (r.get('n_blocks_b', 0) or 0) >= min_blocks
    ]
    if not rows_with_data:
        logger.warning("No data for early fraction bars; skipping.")
        return None

    # Sort by max(ef_a, ef_b) descending - most anchored at top
    rows_sorted = sorted(
        rows_with_data,
        key=lambda r: max(r['early_fraction_a'] or 0, r['early_fraction_b'] or 0),
        reverse=True,
    )

    n_rows = len(rows_sorted)
    bar_height = 0.35
    fig_height = max(4.0, 0.55 * n_rows + 2.0)
    fig, ax = plt.subplots(figsize=(7.0, fig_height))

    y_pos = np.arange(n_rows)

    ef_a = [r['early_fraction_a'] if r['early_fraction_a'] is not None else 0.0
            for r in rows_sorted]
    ef_b = [r['early_fraction_b'] if r['early_fraction_b'] is not None else 0.0
            for r in rows_sorted]

    ax.barh(y_pos + bar_height / 2, ef_a, bar_height, label='"a" runs',
            color='#0072B2', alpha=0.85)
    ax.barh(y_pos - bar_height / 2, ef_b, bar_height, label='"b" runs',
            color='#E69F00', alpha=0.85)

    # Reference line
    ax.axvline(early_threshold, color='grey', linestyle='--', linewidth=1.0,
               alpha=0.6, label=f'Uniform null ({early_threshold:.0%})')

    # Stars for significant states
    for idx, r in enumerate(rows_sorted):
        atype = r.get('anchoring_type', 'none')
        if atype != 'none':
            max_ef = max(ef_a[idx], ef_b[idx])
            ax.text(max_ef + 0.01, y_pos[idx], '*', fontsize=12, fontweight='bold',
                    va='center', color='red')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([
        f"s{r['state']} (r={r['recurrence_score']:.2f})"
        for r in rows_sorted
    ], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f'Early fraction (first {early_threshold:.0%} of run)')
    ax.set_title(
        f'Early fraction by suffix - {sub_id}\n'
        f'(states with < {min_blocks} blocks excluded)',
        fontsize=10)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, axis='x', alpha=0.20)

    fig.tight_layout()
    out_png = os.path.join(out_dir, 'early_fraction_bar_chart.png')
    fig.savefig(out_png, bbox_inches='tight')
    fig.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved early fraction bar chart: %s", out_png)
    return out_png


def plot_anchored_transition_chains(
    model_path,
    state_rows,
    excluded_sub_hrf,
    n_states,
    out_dir,
    sub_id,
    top_feeders=3,
    min_prob=0.02,
):
    """Layered flow diagram of transitions into/between anchored states.

    Left column: feeder states (non-anchored + sub-HRF).
    Right column: anchored states, color-coded by anchoring type.
    Arrows from feeders → anchored with width ∝ transition probability.
    Inter-anchored transitions shown as curved arrows on the right side.

    Args:
        model_path: path to best_model.pkl
        state_rows: list of per-state dicts (with anchoring_type)
        excluded_sub_hrf: set of sub-HRF state IDs
        n_states: total number of states
        out_dir: output directory
        sub_id: subject ID for title
        top_feeders: number of top feeders per anchored state (default 3)
        min_prob: minimum transition probability to show (default 0.02)
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    if not os.path.exists(model_path):
        logger.warning("Model not found for transition chain plot; skipping.")
        return None

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        transmat = model.transmat_[:n_states, :n_states]
    except Exception as e:
        logger.warning("Cannot load model for transition chain plot: %s", e)
        return None

    anchored = {r['state']: r['anchoring_type']
                for r in state_rows if r['anchoring_type'] != 'none'}
    if not anchored:
        logger.warning("No anchored states for transition chain plot; skipping.")
        return None

    # Anchoring type colors (Wong palette)
    type_colors = {
        'a_start_specific': '#D55E00',   # vermillion
        'ab_start_common': '#E69F00',    # orange
        'b_start_specific': '#0072B2',   # blue
        'sub_hrf': '#999999',            # grey
        'none': '#BBBBBB',              # light grey
    }

    # ── Collect edges: feeder→anchored and inter-anchored ──
    # feeder_edges: list of (source, target, prob, source_type)
    feeder_edges = []
    inter_anchored_edges = []
    feeder_nodes = {}  # sid → node_type

    for target in anchored:
        if target >= transmat.shape[1]:
            continue
        col = transmat[:, target].copy()
        col[target] = 0  # exclude self-transition
        top_idx = np.argsort(-col)[:top_feeders]
        for source in top_idx:
            source = int(source)
            p = float(col[source])
            if p < min_prob:
                continue
            if source in anchored:
                continue  # inter-anchored handled separately
            ntype = 'sub_hrf' if source in excluded_sub_hrf else 'feeder'
            feeder_nodes[source] = ntype
            feeder_edges.append((source, target, p, ntype))

    for source in anchored:
        for target in anchored:
            if source == target:
                continue
            if source >= transmat.shape[0] or target >= transmat.shape[1]:
                continue
            p = float(transmat[source, target])
            if p >= min_prob:
                inter_anchored_edges.append((source, target, p))

    if not feeder_edges and not inter_anchored_edges:
        logger.warning("No edges above threshold for transition chain plot.")
        return None

    # ── Layout positions ──
    # Sort anchored states by type then ID for visual grouping
    type_order = {'ab_start_common': 0, 'a_start_specific': 1, 'b_start_specific': 2}
    anchored_sorted = sorted(anchored.keys(),
                             key=lambda s: (type_order.get(anchored[s], 9), s))
    n_anch = len(anchored_sorted)

    # Assign feeders close to their primary target to minimize crossing
    feeder_primary = {}  # feeder_sid → primary anchored target (highest prob)
    for src, tgt, p, _ in feeder_edges:
        if src not in feeder_primary or p > feeder_primary[src][1]:
            feeder_primary[src] = (tgt, p)

    feeder_sorted = sorted(feeder_nodes.keys(),
                           key=lambda s: (anchored_sorted.index(feeder_primary[s][0])
                                          if s in feeder_primary else 999, s))
    n_feed = len(feeder_sorted)

    # Coordinate system: x in [0, 1], y in [0, 1]
    x_feed, x_anch = 0.18, 0.72
    node_h, node_w = 0.06, 0.16  # node box dimensions in axes coords

    n_slots = max(n_anch, n_feed, 1)
    fig_height = max(4.0, 1.2 * n_slots + 1.5)
    fig, ax = plt.subplots(figsize=(8.0, fig_height))
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.05, 1.05)
    ax.axis('off')

    def y_positions(n, total_slots):
        """Evenly space *n* items vertically within [0.05, 0.95]."""
        if n == 1:
            return [0.5]
        margin = 0.08
        return [margin + i * (1.0 - 2 * margin) / (n - 1) for i in range(n)]

    anch_ys = y_positions(n_anch, n_slots)
    feed_ys = y_positions(n_feed, n_slots) if n_feed else []

    anch_pos = {sid: (x_anch, anch_ys[i]) for i, sid in enumerate(anchored_sorted)}
    feed_pos = {sid: (x_feed, feed_ys[i]) for i, sid in enumerate(feeder_sorted)}

    # ── Draw nodes as rounded rectangles ──
    def draw_node(ax, cx, cy, label, color, is_sub_hrf=False):
        x0 = cx - node_w / 2
        y0 = cy - node_h / 2
        box = FancyBboxPatch(
            (x0, y0), node_w, node_h,
            boxstyle="round,pad=0.01",
            facecolor=color, edgecolor='black', linewidth=1.2,
            alpha=0.9, transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(box)
        suffix = ' \u25C7' if is_sub_hrf else ''
        ax.text(cx, cy, f'{label}{suffix}', transform=ax.transAxes,
                ha='center', va='center', fontsize=9, fontweight='bold',
                clip_on=False)

    # Draw anchored nodes
    for sid in anchored_sorted:
        cx, cy = anch_pos[sid]
        color = type_colors.get(anchored[sid], type_colors['none'])
        draw_node(ax, cx, cy, f's{sid}', color)

    # Draw feeder nodes
    for sid in feeder_sorted:
        cx, cy = feed_pos[sid]
        is_sub = feeder_nodes[sid] == 'sub_hrf'
        color = type_colors['sub_hrf'] if is_sub else type_colors['none']
        draw_node(ax, cx, cy, f's{sid}', color, is_sub_hrf=is_sub)

    # ── Draw feeder → anchored arrows ──
    max_p = max((p for _, _, p, _ in feeder_edges), default=0.1)

    for src, tgt, p, _ in feeder_edges:
        sx, sy = feed_pos[src]
        tx, ty = anch_pos[tgt]
        # Arrow from right edge of feeder to left edge of anchored
        arrow = FancyArrowPatch(
            (sx + node_w / 2, sy), (tx - node_w / 2, ty),
            arrowstyle='-|>', mutation_scale=14,
            linewidth=0.8 + 3.5 * (p / max_p),
            color='#555555', alpha=0.7,
            transform=ax.transAxes, clip_on=False,
            connectionstyle='arc3,rad=0.0',
        )
        ax.add_patch(arrow)
        # Probability label at midpoint
        mx = (sx + node_w / 2 + tx - node_w / 2) / 2
        my = (sy + ty) / 2
        ax.text(mx, my + 0.015, f'{p:.2f}', transform=ax.transAxes,
                ha='center', va='bottom', fontsize=7, color='#333333',
                clip_on=False)

    # ── Draw inter-anchored arrows (curved, on right side) ──
    if inter_anchored_edges:
        max_ia = max(p for _, _, p in inter_anchored_edges)
        for src, tgt, p in inter_anchored_edges:
            sx, sy = anch_pos[src]
            tx, ty = anch_pos[tgt]
            # Curve to the right to avoid overlapping with nodes
            rad = 0.3 if sy > ty else -0.3
            arrow = FancyArrowPatch(
                (sx + node_w / 2 + 0.01, sy),
                (tx + node_w / 2 + 0.01, ty),
                arrowstyle='-|>', mutation_scale=12,
                linewidth=0.6 + 3.0 * (p / max_ia),
                color='#888888', alpha=0.6,
                transform=ax.transAxes, clip_on=False,
                connectionstyle=f'arc3,rad={rad}',
            )
            ax.add_patch(arrow)
            # Label on the curve
            arc_x = max(sx, tx) + node_w / 2 + 0.06
            arc_y = (sy + ty) / 2
            ax.text(arc_x, arc_y, f'{p:.2f}', transform=ax.transAxes,
                    ha='left', va='center', fontsize=7, color='#666666',
                    clip_on=False)

    # ── Column headers ──
    if n_feed:
        ax.text(x_feed, 1.01, 'Feeders', transform=ax.transAxes,
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(x_anch, 1.01, 'Anchored', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=10, fontweight='bold')

    # ── Legend ──
    from matplotlib.lines import Line2D
    legend_elements = []
    # Only include anchoring types that are present
    for atype, label in [('ab_start_common', 'ab_start_common'),
                          ('a_start_specific', 'a_start_specific'),
                          ('b_start_specific', 'b_start_specific')]:
        if any(anchored[s] == atype for s in anchored):
            legend_elements.append(
                Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=type_colors[atype],
                       markersize=10, markeredgecolor='black', label=label))
    if any(feeder_nodes[s] == 'sub_hrf' for s in feeder_nodes):
        legend_elements.append(
            Line2D([0], [0], marker='s', color='w',
                   markerfacecolor=type_colors['sub_hrf'],
                   markersize=9, markeredgecolor='black',
                   label='sub-HRF feeder (\u25C7)'))
    if any(feeder_nodes[s] == 'feeder' for s in feeder_nodes):
        legend_elements.append(
            Line2D([0], [0], marker='s', color='w',
                   markerfacecolor=type_colors['none'],
                   markersize=9, markeredgecolor='black',
                   label='non-anchored feeder'))

    if legend_elements:
        ax.legend(handles=legend_elements, loc='lower left', fontsize=7,
                  framealpha=0.9, bbox_to_anchor=(0.0, -0.02))

    ax.set_title(f'Transition chains into anchored states\n{sub_id}', fontsize=10)

    fig.tight_layout()
    out_png = os.path.join(out_dir, 'anchored_transition_chains.png')
    fig.savefig(out_png, bbox_inches='tight')
    fig.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved transition chain plot: %s", out_png)
    return out_png


# =============================================================================
# Main pipeline
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect temporally-anchored brain states."
    )
    parser.add_argument('--sub_id', type=str, required=True)
    parser.add_argument('--parcellation', type=str, default='atlas-4S156Parcels')
    parser.add_argument(
        '--n_permutations', type=int, default=2000,
        help="Number of block-shuffle permutations per suffix (default: 2000)"
    )
    parser.add_argument(
        '--fdr_threshold', type=float, default=0.10,
        help="FDR q-value threshold for flagging position-locked states (default: 0.10)"
    )
    parser.add_argument(
        '--early_threshold', type=float, default=DEFAULT_EARLY_THRESHOLD,
        help=f"Normalized position cutoff for 'early' blocks (default: {DEFAULT_EARLY_THRESHOLD})"
    )
    parser.add_argument('--vt', type=str, default=None,
                        help="Variance threshold subdirectory under final/ (e.g., 0.99). "
                             "Reads from final/vt{VT}/. If omitted, reads from final/ directly "
                             "(legacy path).")
    parser.add_argument('--exclude_sub_hrf', action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Exclude sub-HRF states from position analysis "
                             "(default: True). Use --no-exclude-sub-hrf to include all.")
    return parser.parse_args()


def main():
    args = parse_args()
    sub_id = args.sub_id
    parc = normalize_parcellation_name(args.parcellation)
    early_threshold = args.early_threshold

    # Input paths
    if args.vt is not None:
        hmm_base = os.path.join(
            SCRATCH_DIR, 'output', '04_combined_hdphmm', parc, sub_id,
            'final', f'vt{args.vt}'
        )
    else:
        hmm_base = os.path.join(
            SCRATCH_DIR, 'output', '04_combined_hdphmm', parc, sub_id, 'final'
        )
    decoded_path = os.path.join(hmm_base, 'decoded_states.pkl')
    model_path = os.path.join(hmm_base, 'best_model.pkl')
    recurrence_base = os.path.join(
        SCRATCH_DIR, 'output', '05a_recurrence_analysis', parc, sub_id
    )
    if args.vt is not None:
        recurrence_base = os.path.join(recurrence_base, f'vt{args.vt}')
    recurrence_summary_path = os.path.join(recurrence_base, 'recurrence_summary.json')

    for p in (decoded_path, recurrence_summary_path):
        if not os.path.exists(p):
            logger.error("Missing required input: %s", p)
            sys.exit(1)

    # Output dir (vt-aware) - no all_states/ subfolder
    out_dir = os.path.join(
        SCRATCH_DIR, 'output', '05e_temporal_trend_a2', parc, sub_id
    )
    if args.vt is not None:
        out_dir = os.path.join(out_dir, f'vt{args.vt}')
    os.makedirs(out_dir, exist_ok=True)

    # Load inputs
    logger.info("Loading decoded states from %s", decoded_path)
    with open(decoded_path, 'rb') as f:
        decoded_states = pickle.load(f)

    with open(recurrence_summary_path, 'r') as f:
        recurrence_summary = json.load(f)

    n_states = recurrence_summary['n_states']
    recurrence_scores = np.array(recurrence_summary['recurrence_scores'], dtype=float)

    # Load eligible states (sub-HRF filter)
    excluded_sub_hrf = set()
    if args.exclude_sub_hrf:
        try:
            eligible_ids, excluded_ids, _ = load_eligible_states(recurrence_base)
            excluded_sub_hrf = set(excluded_ids)
            logger.info(
                "Sub-HRF exclusion ON: %d states excluded from position analysis",
                len(excluded_sub_hrf),
            )
        except FileNotFoundError:
            logger.warning(
                "eligible_states.json not found in %s; sub-HRF filtering skipped. "
                "Re-run 05a to generate it.",
                recurrence_base,
            )

    logger.info(
        "Loaded: %d states, %d runs | suffix counts: %s",
        n_states,
        len(decoded_states),
        {s: sum(1 for r in decoded_states if r.endswith(s)) for s in ('a', 'b', 'c', 'd')},
    )

    # 1. Extract block records for ALL states
    block_records = extract_state_block_records(
        decoded_states,
        recurrence_scores,
        include_states=None,  # all states
        tr_seconds=TR_SECONDS,
    )
    # Filter out sub-HRF states if requested
    if excluded_sub_hrf:
        block_records = [
            r for r in block_records if int(r['state']) not in excluded_sub_hrf
        ]
    logger.info("Extracted %d block records (after sub-HRF filter)", len(block_records))

    # 2. Compute per-state position metrics
    pos_metrics = compute_position_metrics(block_records, early_threshold=early_threshold)

    # 3. Build run-level block arrays for permutation
    run_data = build_run_block_arrays(decoded_states)

    # 4. Observed early fractions
    obs_ef_a, obs_counts_a = compute_observed_early_fractions(
        run_data, n_states, suffix='a', early_threshold=early_threshold)
    obs_ef_b, obs_counts_b = compute_observed_early_fractions(
        run_data, n_states, suffix='b', early_threshold=early_threshold)

    # 5. Permutation tests
    p_early_a = permutation_test_early_fraction(
        run_data, n_states, obs_ef_a, obs_counts_a,
        suffix='a', n_permutations=args.n_permutations, seed=42,
        early_threshold=early_threshold,
    )
    p_early_b = permutation_test_early_fraction(
        run_data, n_states, obs_ef_b, obs_counts_b,
        suffix='b', n_permutations=args.n_permutations, seed=43,
        early_threshold=early_threshold,
    )

    # 5b. Null out p-values for excluded sub-HRF states so they don't
    #     inflate the FDR correction denominator.
    if excluded_sub_hrf:
        for sid in excluded_sub_hrf:
            if sid < n_states:
                p_early_a[sid] = np.nan
                p_early_b[sid] = np.nan
                obs_ef_a[sid] = np.nan
                obs_ef_b[sid] = np.nan
        logger.info(
            "Nulled p-values for %d excluded sub-HRF states", len(excluded_sub_hrf)
        )

    # 6. Separate FDR correction per suffix family
    q_early_a = _fdr_with_nan(p_early_a)
    q_early_b = _fdr_with_nan(p_early_b)

    # 7. Conjunction classification (neutral labels)
    anchoring_types = classify_anchoring_type(q_early_a, q_early_b, args.fdr_threshold)

    # 8. Descriptive theme_fraction_a
    theme_frac = compute_theme_fraction(block_records, threshold_tr=33)

    # 9. Build output rows
    state_rows = []
    for state_id in range(n_states):
        pm = pos_metrics.get(state_id, {})
        rec = float(recurrence_scores[state_id])

        ef_a = pm.get('early_fraction_a')
        ef_b = pm.get('early_fraction_b')
        p_a = float(p_early_a[state_id]) if not np.isnan(p_early_a[state_id]) else None
        q_a = float(q_early_a[state_id]) if not np.isnan(q_early_a[state_id]) else None
        p_b = float(p_early_b[state_id]) if not np.isnan(p_early_b[state_id]) else None
        q_b = float(q_early_b[state_id]) if not np.isnan(q_early_b[state_id]) else None

        position_locked_a = (q_a is not None and q_a < args.fdr_threshold)
        position_locked_b = (q_b is not None and q_b < args.fdr_threshold)

        tf_a = theme_frac.get(state_id)
        delta = None
        if ef_a is not None and ef_b is not None:
            delta = round(ef_a - ef_b, 6)

        state_rows.append({
            'state': state_id,
            'recurrence_score': rec,
            'n_blocks_all': pm.get('n_blocks_all', 0),
            'n_blocks_a': pm.get('n_blocks_a', 0),
            'n_blocks_b': pm.get('n_blocks_b', 0),
            'mean_position_a': pm.get('mean_position_a'),
            'mean_position_b': pm.get('mean_position_b'),
            'position_iqr_a': pm.get('position_iqr_a'),
            'position_iqr_b': pm.get('position_iqr_b'),
            'early_fraction_a': ef_a,
            'early_fraction_b': ef_b,
            'p_early_a_perm': p_a,
            'q_early_a_fdr': q_a,
            'p_early_b_perm': p_b,
            'q_early_b_fdr': q_b,
            'n_runs_a': pm.get('n_runs_a', 0),
            'n_runs_b': pm.get('n_runs_b', 0),
            'position_locked_a': position_locked_a,
            'position_locked_b': position_locked_b,
            'theme_fraction_a': tf_a,
            'delta_early': delta,
            'anchoring_type': anchoring_types[state_id],
            # Kept in JSON only (not in CSV) for downstream plotting
            'mean_position_all': pm.get('mean_position_all'),
            'position_iqr_all': pm.get('position_iqr_all'),
        })

    # 10. Summaries
    position_locked_a_states = [r['state'] for r in state_rows if r['position_locked_a']]
    position_locked_b_states = [r['state'] for r in state_rows if r['position_locked_b']]

    anchoring_counts = defaultdict(int)
    for at in anchoring_types:
        anchoring_counts[at] += 1

    logger.info(
        "\nPosition-locked states (FDR q < %.2f):", args.fdr_threshold
    )
    logger.info(
        "  Early in 'a' runs: %d states %s",
        len(position_locked_a_states),
        position_locked_a_states,
    )
    logger.info(
        "  Early in 'b' runs: %d states %s",
        len(position_locked_b_states),
        position_locked_b_states,
    )
    logger.info(
        "  Anchoring classification: %s",
        dict(anchoring_counts),
    )

    # 10b. Transition-structure confound check (includes sub-HRF feeders)
    locked_set = set(position_locked_a_states) | set(position_locked_b_states)
    trans_confound = transition_confound_check(
        model_path, locked_set, n_states, excluded_sub_hrf=excluded_sub_hrf)
    if trans_confound:
        if trans_confound['n_secondary'] > 0:
            logger.info(
                "  Transition confound: %d states may be secondarily anchored: %s",
                trans_confound['n_secondary'],
                [(s['state'], f"<-s{s['source_locked_state']} p={s['transition_prob']}")
                 for s in trans_confound['secondary_states']],
            )
        if trans_confound['n_sub_hrf_feeders'] > 0:
            logger.info(
                "  Sub-HRF feeders: %d sub-HRF states feed anchored states: %s",
                trans_confound['n_sub_hrf_feeders'],
                [(f"s{s['sub_hrf_source']}->s{s['anchored_state']} p={s['transition_prob']}")
                 for s in trans_confound['sub_hrf_feeders']],
            )

    # 11. Save CSV (excludes mean_position_all / position_iqr_all kept for JSON)
    metrics_path = os.path.join(out_dir, 'temporal_position_metrics.csv')
    fieldname_set = set(POSITION_METRIC_FIELDNAMES)
    csv_rows = [
        {k: v for k, v in r.items() if k in fieldname_set}
        for r in state_rows
    ]
    write_records_csv(metrics_path, POSITION_METRIC_FIELDNAMES, csv_rows)
    logger.info("Saved metrics CSV: %s", metrics_path)

    # 12. Plots
    plot_path_scatter = plot_ab_early_fraction_scatter(
        state_rows, out_dir, sub_id,
        early_threshold=early_threshold, fdr_threshold=args.fdr_threshold)
    plot_path_cdf = plot_position_cdfs(
        block_records, state_rows, out_dir, sub_id)
    plot_path_bars = plot_early_fraction_bars(
        state_rows, out_dir, sub_id,
        early_threshold=early_threshold, fdr_threshold=args.fdr_threshold)
    plot_path_chains = plot_anchored_transition_chains(
        model_path, state_rows, excluded_sub_hrf, n_states, out_dir, sub_id)

    # 13. JSON summary
    summary = {
        'sub_id': sub_id,
        'parcellation': parc,
        'n_states': n_states,
        'n_runs': len(decoded_states),
        'n_blocks_total': len(block_records),
        'early_position_threshold': early_threshold,
        'fdr_threshold': args.fdr_threshold,
        'n_permutations': args.n_permutations,
        'permutation_correction': '(count + 1) / (n_perm + 1)',
        'fdr_strategy': 'separate_per_family',
        'fdr_families': ['early_a', 'early_b'],
        'note': (
            'a/b/c/d suffixes are scanner session boundaries, not narrative splits. '
            '"a" runs = first half of episode; "b" runs = second half. '
            'Position-locked states may reflect temporal structure, not episode content. '
            'Causal interpretation deferred to downstream analyses (08 series).'
        ),
        'theme_fraction_caveat': (
            'Theme song position varies across episodes due to cold opens; '
            'theme_fraction_a uses a fixed 33-TR window as an approximate lower '
            'bound. Not used for inference.'
        ),
        'exclude_sub_hrf': args.exclude_sub_hrf,
        'n_excluded_sub_hrf_states': len(excluded_sub_hrf),
        'excluded_sub_hrf_states': sorted(excluded_sub_hrf),
        'permutation_note': (
            'Block-level permutation shuffles block order within each run, '
            'preserving block counts and dwell times but destroying temporal '
            'autocorrelation. P-values may be anti-conservative if position '
            'effects are driven by state-persistence rather than true anchoring.'
        ),
        'anchoring_summary': dict(anchoring_counts),
        'anchoring_labels': {
            'a_start_specific': 'Significantly early in "a" runs only (q_early_a < FDR, not q_early_b)',
            'ab_start_common': 'Significantly early in both "a" and "b" runs',
            'b_start_specific': 'Significantly early in "b" runs only (q_early_b < FDR, not q_early_a)',
            'none': 'Not position-anchored',
        },
        'position_locked_a': {
            'n': len(position_locked_a_states),
            'states': position_locked_a_states,
        },
        'position_locked_b': {
            'n': len(position_locked_b_states),
            'states': position_locked_b_states,
        },
        'transition_confound': trans_confound,
        'plots': {
            'ab_early_fraction_scatter': os.path.basename(plot_path_scatter) if plot_path_scatter else None,
            'position_cdf_flagged_states': os.path.basename(plot_path_cdf) if plot_path_cdf else None,
            'early_fraction_bar_chart': os.path.basename(plot_path_bars) if plot_path_bars else None,
            'anchored_transition_chains': os.path.basename(plot_path_chains) if plot_path_chains else None,
        },
        'per_state': [
            {k: v for k, v in r.items()}
            for r in state_rows
        ],
    }

    json_path = os.path.join(out_dir, 'temporal_position_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved JSON summary: %s", json_path)
    logger.info("\nOutputs saved to %s", out_dir)


if __name__ == '__main__':
    main()
