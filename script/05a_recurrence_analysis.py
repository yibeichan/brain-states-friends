#!/usr/bin/env python3
"""
05a_recurrence_analysis.py - Compute continuous recurrence scores for brain states.

This script operates on the decoded states produced by 04_combined_hdphmm.py
and computes fractional occupancy, recurrence scores, and season specificity indices.

Recurrence is a continuous gradient: each state's score is the fraction of runs
in which it is active (FO > threshold).  Season-specificity significance is
tested separately via a permutation test with FDR correction.

Recurrence and specificity are computed at the run level — each scan run
(e.g. s01e01a, s01e01b) is treated as an independent unit.  No multipart
episode aggregation is performed.  Dwell time and revisitation metrics are
model-derived temporal diagnostics, not direct neural timescale estimates.

Prerequisites:
    - 04_combined_hdphmm.py (mode: select) completed for this subject
    - Output available at {SCRATCH_DIR}/output/04_combined_hdphmm/{parcellation}/{sub_id}/final/

Outputs:
    Saves to {SCRATCH_DIR}/output/05a_recurrence_analysis/{parcellation}/{sub_id}/:
    - fractional_occupancy.pkl (run-level)
    - recurrence_scores.npy, specificity_index.npy
    - recurrence_summary.json (includes analysis_scope, threshold_method, season_specificity)
    - permutation_pvalues.json (uncorrected + FDR, with +1 correction)
    - per-state and per-run CSV tables, diagnostic plots
    - eligible_states.json (states passing the sub-HRF filter for downstream use)
"""

import os
import sys
import json
import pickle
import logging
import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import numpy.core as np_core


# Add script directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils.stats import benjamini_hochberg
from utils.plot_style import (
    RECURRENCE_CMAP,
    recurrence_color,
    make_recurrence_colorbar,
    apply_publication_style,
)
from utils.common import (
    normalize_parcellation_name,
    _get_season,
    parse_episode_order_key,
)
from utils.state_blocks import (
    TR_SECONDS,
    BLOCK_FIELDNAMES,
    EPISODE_STATE_FIELDNAMES,
    extract_state_block_records,
    summarize_block_records,
    write_records_csv,
)
from utils.recurrence_utils import (
    compute_fractional_occupancy,
    compute_recurrence_scores,
    compute_pooled_decoded_occupancy,
    compute_per_season_recurrence,
    compute_specificity_index,
)

from dotenv import load_dotenv

load_dotenv()
SCRATCH_DIR = os.getenv('SCRATCH_DIR')
if SCRATCH_DIR is None:
    raise ValueError("SCRATCH_DIR must be set in the .env file")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# States with median block dwell below this threshold are flagged as sub-HRF
# resolution.  At TR = 1.49 s this equals ~4.5 s, just below the canonical HRF
# peak (~5-6 s).  The flag indicates that BOLD evidence for these state
# assignments is limited — it does not imply the underlying neural event was
# shorter than the HRF.
SUB_HRF_DWELL_THRESHOLD_TR = 3.0

STATE_METRIC_FIELDNAMES = [
    'state',
    'decoded_usage_status',
    'recurrence_ratio',
    'recurrence_score',
    'n_active_runs',
    'n_blocks',
    'total_dwell_tr',
    'total_dwell_s',
    'mean_dwell_tr',
    'mean_dwell_s',
    'median_dwell_tr',
    'median_dwell_s',
    'frac_blocks_sub_hrf',
    'sub_hrf_resolution',
]
RUN_METRIC_FIELDNAMES = [
    'run_id',
    'season',
    'run_length_tr',
    'run_length_s',
    'n_unique_states_decoded',
    'n_unique_states_active_for_recurrence',
    'n_state_blocks',
    'n_revisited_states',
    'within_run_recurrence_ratio',
    'total_reentry_count',
    'mean_blocks_per_unique_state',
]
apply_publication_style()

# Pickled numpy objects may reference numpy._core depending on the source env.
sys.modules.setdefault('numpy._core', np_core)


# =============================================================================
# Core Recurrence Functions
# =============================================================================

# compute_fractional_occupancy, compute_recurrence_scores,
# compute_pooled_decoded_occupancy: moved to utils/recurrence_utils.py


def load_model_history_usage(best_model_path):
    """Load final model-history state usage if available."""
    if not os.path.exists(best_model_path):
        logger.warning("Best model missing; activity comparison will skip model-history usage.")
        return None

    try:
        with open(best_model_path, 'rb') as f:
            best_model = pickle.load(f)
    except Exception as exc:
        logger.warning("Failed to load best_model.pkl for activity comparison: %s", exc)
        return None

    if not hasattr(best_model, 'history') or not best_model.history:
        return None
    usage_hist = best_model.history.get('state_usage')
    if not usage_hist:
        return None
    usage = np.asarray(usage_hist[-1], dtype=float)
    if usage.size == 0 or not np.all(np.isfinite(usage)):
        return None
    return usage


# compute_per_season_recurrence: moved to utils/recurrence_utils.py


def plot_recurrence_histogram(recurrence, out_dir, sub_id):
    """Save recurrence-score histogram."""
    recurrence = np.asarray(recurrence, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0.0, 1.0, 21)
    ax.hist(recurrence, bins=bins, color='#4C78A8', edgecolor='black',
            linewidth=0.8, alpha=0.85)

    active = recurrence[recurrence > 0]
    if active.size:
        ax.axvline(np.median(active), color='red', linestyle='--', linewidth=1.2,
                    alpha=0.7, label=f'Median (active) = {np.median(active):.2f}')

    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel('Recurrence score')
    ax.set_ylabel('State count')
    ax.set_title(f'Recurrence Score Distribution\n{sub_id}')
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(loc='upper center', fontsize=9)

    out_png = os.path.join(out_dir, 'recurrence_score_histogram.png')
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    fig.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved recurrence histogram: %s", out_png)


def build_state_recurrence_dwell_metrics(
    decoded_states,
    fo,
    recurrence,
    n_states,
    fo_threshold,
    tr_seconds=TR_SECONDS,
):
    """Summarize recurrence ratio and dwell time statistics for every state.

    The sub_hrf_resolution flag uses the **median** block dwell time rather than
    the mean, because dwell distributions are right-skewed (geometric by
    construction for Markov models) and the mean is inflated by a few long
    blocks.  The flag indicates that BOLD evidence for a state's assignments is
    limited — it does not imply the underlying neural event was shorter than the
    HRF.
    """
    block_records = extract_state_block_records(
        decoded_states,
        recurrence,
        include_states=None,
        tr_seconds=tr_seconds,
    )

    # Collect per-block durations grouped by state
    block_durations = [[] for _ in range(n_states)]
    for record in block_records:
        state_id = int(record['state'])
        block_durations[state_id].append(int(record['duration_tr']))

    if fo:
        fo_matrix = np.stack(list(fo.values()))
        n_active_runs = (fo_matrix > fo_threshold).sum(axis=0).astype(int)
    else:
        n_active_runs = np.zeros(n_states, dtype=int)

    state_rows = []
    for state_id in range(n_states):
        durations = block_durations[state_id]
        block_count = len(durations)
        total_tr = float(sum(durations))

        mean_dwell_tr = None if block_count == 0 else total_tr / block_count
        mean_dwell_s = None if mean_dwell_tr is None else mean_dwell_tr * tr_seconds

        if block_count == 0:
            median_dwell_tr = None
            frac_sub_hrf = None
        else:
            median_dwell_tr = float(np.median(durations))
            frac_sub_hrf = float(
                sum(1 for d in durations if d < SUB_HRF_DWELL_THRESHOLD_TR)
                / block_count
            )
        median_dwell_s = (
            None if median_dwell_tr is None else median_dwell_tr * tr_seconds
        )

        if block_count == 0:
            decoded_usage_status = 'never_decoded'
        elif recurrence[state_id] == 0:
            decoded_usage_status = 'below_fo_threshold'
        else:
            decoded_usage_status = 'active_for_recurrence'

        sub_hrf = (
            median_dwell_tr is not None
            and median_dwell_tr < SUB_HRF_DWELL_THRESHOLD_TR
        )

        state_rows.append({
            'state': state_id,
            'decoded_usage_status': decoded_usage_status,
            'recurrence_ratio': float(recurrence[state_id]),
            'recurrence_score': float(recurrence[state_id]),
            'n_active_runs': int(n_active_runs[state_id]),
            'n_blocks': block_count,
            'total_dwell_tr': total_tr,
            'total_dwell_s': total_tr * tr_seconds,
            'mean_dwell_tr': mean_dwell_tr,
            'mean_dwell_s': mean_dwell_s,
            'median_dwell_tr': median_dwell_tr,
            'median_dwell_s': median_dwell_s,
            'frac_blocks_sub_hrf': frac_sub_hrf,
            'sub_hrf_resolution': sub_hrf,
        })

    return state_rows


def build_run_metrics(
    decoded_states,
    fo,
    fo_threshold,
    tr_seconds=TR_SECONDS,
):
    """Summarize per-run state diversity and within-run revisitation."""
    rows = []
    for run_id, state_seq in decoded_states.items():
        state_seq = np.asarray(state_seq, dtype=int)
        run_length_tr = int(len(state_seq))
        try:
            season = _get_season(run_id)
        except ValueError:
            season = None

        if run_length_tr == 0:
            rows.append({
                'run_id': run_id,
                'season': season,
                'run_length_tr': 0,
                'run_length_s': 0.0,
                'n_unique_states_decoded': 0,
                'n_unique_states_active_for_recurrence': 0,
                'n_state_blocks': 0,
                'n_revisited_states': 0,
                'within_run_recurrence_ratio': 0.0,
                'total_reentry_count': 0,
                'mean_blocks_per_unique_state': 0.0,
            })
            continue

        unique_states = np.unique(state_seq).astype(int)
        active_state_ids = np.flatnonzero(fo[run_id] > fo_threshold).astype(int)

        change_points = np.flatnonzero(state_seq[1:] != state_seq[:-1]) + 1
        starts = np.concatenate(([0], change_points))
        block_states = state_seq[starts].astype(int)
        block_counts = Counter(block_states.tolist())

        n_unique_states = int(unique_states.size)
        n_state_blocks = int(block_states.size)
        n_revisited_states = int(sum(count > 1 for count in block_counts.values()))
        total_reentry_count = int(
            sum(count - 1 for count in block_counts.values() if count > 1)
        )
        within_run_recurrence_ratio = (
            float(n_revisited_states / n_unique_states) if n_unique_states else 0.0
        )
        mean_blocks_per_unique_state = (
            float(n_state_blocks / n_unique_states) if n_unique_states else 0.0
        )

        rows.append({
            'run_id': run_id,
            'season': season,
            'run_length_tr': run_length_tr,
            'run_length_s': run_length_tr * tr_seconds,
            'n_unique_states_decoded': n_unique_states,
            'n_unique_states_active_for_recurrence': int(active_state_ids.size),
            'n_state_blocks': n_state_blocks,
            'n_revisited_states': n_revisited_states,
            'within_run_recurrence_ratio': within_run_recurrence_ratio,
            'total_reentry_count': total_reentry_count,
            'mean_blocks_per_unique_state': mean_blocks_per_unique_state,
        })

    return rows


def plot_run_state_diversity(
    run_rows,
    out_dir,
    sub_id,
):
    """Scatter unique decoded states vs within-run revisitation."""
    fig, ax = plt.subplots(figsize=(8, 6))
    seasons = sorted({row['season'] for row in run_rows if row['season'] is not None})

    if seasons:
        cmap = plt.get_cmap('viridis', len(seasons))
        for idx, season in enumerate(seasons):
            season_rows = [row for row in run_rows if row['season'] == season]
            ax.scatter(
                [row['n_unique_states_decoded'] for row in season_rows],
                [row['within_run_recurrence_ratio'] for row in season_rows],
                s=55,
                alpha=0.8,
                color=cmap(idx),
                edgecolors='black',
                linewidths=0.4,
                label=f'Season {season} (n={len(season_rows)})',
            )
    else:
        ax.scatter(
            [row['n_unique_states_decoded'] for row in run_rows],
            [row['within_run_recurrence_ratio'] for row in run_rows],
            s=55,
            alpha=0.8,
            color='#4C78A8',
            edgecolors='black',
            linewidths=0.4,
        )

    ax.set_xlabel(
        'Unique decoded states per run\n'
        'more different states visited ->'
    )
    ax.set_ylabel(
        'Within-run recurrence ratio\n'
        'more states came back after leaving ->'
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f'Run State Diversity vs. Revisitation\n{sub_id}')
    ax.grid(True, alpha=0.25)
    if seasons:
        ax.legend(loc='best', fontsize=8)

    out_png = os.path.join(out_dir, 'run_state_diversity_vs_revisitation.png')
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    fig.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved run diversity/revisitation plot: %s", out_png)
    return out_png


def plot_state_dwell_vs_recurrence(
    state_rows,
    out_dir,
    sub_id,
):
    """Plot per-state mean dwell time against recurrence ratio."""
    recurrence_ratio = np.array(
        [row['recurrence_ratio'] for row in state_rows],
        dtype=float,
    )
    mean_dwell_s = np.array(
        [
            np.nan if row['mean_dwell_s'] is None else row['mean_dwell_s']
            for row in state_rows
        ],
        dtype=float,
    )
    usage_status = np.array(
        [row['decoded_usage_status'] for row in state_rows],
        dtype=object,
    )

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    # Active states: coloured by recurrence score
    active_mask = (recurrence_ratio > 0) & np.isfinite(mean_dwell_s)
    if np.any(active_mask):
        sc = ax.scatter(
            recurrence_ratio[active_mask],
            mean_dwell_s[active_mask],
            c=recurrence_ratio[active_mask],
            cmap=RECURRENCE_CMAP, vmin=0, vmax=1,
            s=60, alpha=0.8, edgecolors='black', linewidths=0.4,
        )
        fig.colorbar(sc, ax=ax, label='Recurrence score', shrink=0.8)

    # Below-FO-threshold states (decoded but recurrence=0)
    inactive_decoded_mask = (
        (recurrence_ratio == 0) & (usage_status == 'below_fo_threshold')
    )
    if np.any(inactive_decoded_mask):
        ax.scatter(
            recurrence_ratio[inactive_decoded_mask],
            mean_dwell_s[inactive_decoded_mask],
            s=45, alpha=0.6, color='#999999', marker='x', linewidths=1.2,
            label=f'Below FO threshold (n={int(inactive_decoded_mask.sum())})',
        )
    never_decoded_mask = usage_status == 'never_decoded'
    if np.any(never_decoded_mask):
        ax.scatter(
            recurrence_ratio[never_decoded_mask],
            np.zeros(int(never_decoded_mask.sum())),
            s=40, alpha=0.6, color='#999999', marker='+', linewidths=1.2,
            label=f'Never decoded (n={int(never_decoded_mask.sum())})',
        )

    # Annotate top-5 states by recurrence
    top5 = sorted(
        (row for row in state_rows if row['recurrence_score'] > 0 and row['mean_dwell_s'] is not None),
        key=lambda row: row['recurrence_ratio'],
        reverse=True,
    )[:5]
    for row in top5:
        ax.annotate(
            f"  {row['state']}",
            (row['recurrence_ratio'], row['mean_dwell_s']),
            fontsize=8, alpha=0.85, xytext=(4, 0), textcoords='offset points',
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=-0.05)
    ax.set_xlabel('Recurrence ratio\n(fraction of runs where state is active)')
    ax.set_ylabel('Mean dwell time per state block (s)')
    ax.set_title(f'State Mean Dwell vs. Recurrence Ratio\n{sub_id}')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper left', fontsize=8)

    out_png = os.path.join(out_dir, 'state_dwell_vs_recurrence.png')
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    fig.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved dwell-vs-recurrence scatter: %s", out_png)
    return out_png


def plot_state_dwell_recurrence_heatmap(
    state_rows,
    out_dir,
    sub_id,
    n_recurrence_bins=20,
    n_dwell_bins=20,
):
    """Plot a 2D histogram heatmap of mean dwell time versus recurrence ratio."""
    recurrence_ratio = np.array(
        [row['recurrence_ratio'] for row in state_rows],
        dtype=float,
    )
    mean_dwell_s = np.array(
        [
            np.nan if row['mean_dwell_s'] is None else row['mean_dwell_s']
            for row in state_rows
        ],
        dtype=float,
    )

    valid = np.isfinite(mean_dwell_s)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    if np.any(valid):
        x = recurrence_ratio[valid]
        y = mean_dwell_s[valid]
        dwell_max = float(np.nanmax(y))
        dwell_upper = max(dwell_max, 0.1)
        heatmap, xedges, yedges = np.histogram2d(
            x,
            y,
            bins=[
                np.linspace(0.0, 1.0, n_recurrence_bins + 1),
                np.linspace(0.0, dwell_upper, n_dwell_bins + 1),
            ],
        )
        image = ax.pcolormesh(
            xedges,
            yedges,
            heatmap.T,
            cmap='YlOrRd',
            shading='auto',
        )
        colorbar = fig.colorbar(image, ax=ax, pad=0.02)
        colorbar.set_label('Number of states')
    else:
        ax.text(
            0.5, 0.5,
            'No states with defined mean dwell time',
            ha='center', va='center', transform=ax.transAxes,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel('Recurrence ratio\n(fraction of runs where state is active)')
    ax.set_ylabel('Mean dwell time per state block (s)')
    ax.set_title(f'State Dwell vs. Recurrence Heatmap\n{sub_id}')
    ax.grid(False)

    out_png = os.path.join(out_dir, 'state_dwell_vs_recurrence_heatmap.png')
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    fig.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved dwell-vs-recurrence heatmap: %s", out_png)
    return out_png


def build_seasonal_recurrence_breakdown(
    season_recurrence,
    recurrence_scores,
    available_seasons,
    within_season_threshold=0.70,
):
    """Break active states into per-season recurrence details.

    Parameters
    ----------
    season_recurrence : dict
        season -> np.array(n_states,) of within-season recurrence.
    recurrence_scores : np.ndarray
        Global recurrence scores, shape (n_states,).
    available_seasons : list
        Season identifiers.
    within_season_threshold : float
        Threshold for counting a state as 'recurrent within a season'.
    """
    active_states = sorted(int(k) for k in range(len(recurrence_scores)) if recurrence_scores[k] > 0)
    season_fieldnames = [
        f'season_{int(season):02d}_recurrence'
        for season in available_seasons
    ]
    fieldnames = [
        'state',
        'recurrence_score',
        'n_active_seasons',
        'n_recurrent_seasons',
        'active_seasons',
        'recurrent_seasons',
        'across_season_recurrent',
        'seasonal_recurrent',
        'max_season_recurrence',
        'min_season_recurrence',
    ] + season_fieldnames

    rows = []
    for state_id in active_states:
        scores = [float(season_recurrence[season][state_id]) for season in available_seasons]
        active_seasons = [
            int(season)
            for season, score in zip(available_seasons, scores)
            if score > 0
        ]
        recurrent_seasons = [
            int(season)
            for season, score in zip(available_seasons, scores)
            if score >= within_season_threshold
        ]
        across_season_recurrent = (
            len(available_seasons) > 0 and len(recurrent_seasons) == len(available_seasons)
        )
        seasonal_recurrent = 0 < len(recurrent_seasons) < len(available_seasons)

        row = {
            'state': state_id,
            'recurrence_score': float(recurrence_scores[state_id]),
            'n_active_seasons': len(active_seasons),
            'n_recurrent_seasons': len(recurrent_seasons),
            'active_seasons': ';'.join(str(season) for season in active_seasons),
            'recurrent_seasons': ';'.join(str(season) for season in recurrent_seasons),
            'across_season_recurrent': across_season_recurrent,
            'seasonal_recurrent': seasonal_recurrent,
            'max_season_recurrence': max(scores) if scores else 0.0,
            'min_season_recurrence': min(scores) if scores else 0.0,
        }
        for fieldname, score in zip(season_fieldnames, scores):
            row[fieldname] = score
        rows.append(row)

    rows.sort(
        key=lambda row: (
            not row['across_season_recurrent'],
            -row['n_recurrent_seasons'],
            -row['max_season_recurrence'],
            row['state'],
        )
    )

    by_season = {}
    for season in available_seasons:
        season_state_ids = [
            state_id
            for state_id in active_states
            if season_recurrence[season][state_id] >= within_season_threshold
        ]
        by_season[str(season)] = {
            'recurrent_states': season_state_ids,
        }

    summary = {
        'n_active_states': len(active_states),
        'across_season_recurrent_states': [
            row['state'] for row in rows if row['across_season_recurrent']
        ],
        'seasonal_recurrent_states': [
            row['state'] for row in rows if row['seasonal_recurrent']
        ],
        'by_season': by_season,
    }
    return rows, fieldnames, summary


def plot_seasonal_recurrence_heatmap(
    seasonal_rows,
    available_seasons,
    out_dir,
    sub_id,
):
    """Heatmap of per-season recurrence for globally recurring and partial states."""
    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.3 * max(len(seasonal_rows), 1) + 2.0)))

    if seasonal_rows and available_seasons:
        matrix = np.array([
            [row[f'season_{int(season):02d}_recurrence'] for season in available_seasons]
            for row in seasonal_rows
        ], dtype=float)
        image = ax.imshow(
            matrix,
            aspect='auto',
            interpolation='nearest',
            cmap='YlGnBu',
            vmin=0.0,
            vmax=1.0,
        )
        colorbar = fig.colorbar(image, ax=ax, pad=0.02)
        colorbar.set_label('Per-season recurrence')
        ax.set_xticks(np.arange(len(available_seasons)))
        ax.set_xticklabels([f'S{int(season)}' for season in available_seasons])
        ax.set_yticks(np.arange(len(seasonal_rows)))
        ax.set_yticklabels([
            f"s{row['state']} ({row['recurrence_score']:.2f})"
            for row in seasonal_rows
        ], fontsize=8)
    else:
        ax.text(
            0.5, 0.5,
            'No active states available for season breakdown',
            ha='center',
            va='center',
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_xlabel('Season')
    ax.set_ylabel('State')
    ax.set_title(f'Seasonal Recurrence of Active States\n{sub_id}')

    out_png = os.path.join(out_dir, 'seasonal_recurrence_breakdown_heatmap.png')
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    fig.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved seasonal recurrence heatmap: %s", out_png)
    return out_png


def build_activity_definition_summary(
    decoded_states,
    recurrence,
    n_states,
    fo_threshold,
    best_model_path,
    model_usage_threshold,
):
    """Compare model-history, pooled decoded, and recurrence activity definitions."""
    pooled_occupancy, total_trs = compute_pooled_decoded_occupancy(decoded_states, n_states)
    model_usage = load_model_history_usage(best_model_path)

    if model_usage is None:
        model_active_states = []
        model_usage_serializable = None
    else:
        model_active_states = np.flatnonzero(
            model_usage > model_usage_threshold
        ).astype(int).tolist()
        model_usage_serializable = model_usage.tolist()

    pooled_active_states = np.flatnonzero(
        pooled_occupancy > fo_threshold
    ).astype(int).tolist()
    recurrence_noninactive_states = np.flatnonzero(
        recurrence > 0
    ).astype(int).tolist()

    episodic_vs_model = sorted(
        set(recurrence_noninactive_states) - set(model_active_states)
    )
    episodic_vs_pooled = sorted(
        set(recurrence_noninactive_states) - set(pooled_active_states)
    )
    model_vs_recurrence = sorted(
        set(model_active_states) - set(recurrence_noninactive_states)
    )
    pooled_vs_recurrence = sorted(
        set(pooled_active_states) - set(recurrence_noninactive_states)
    )

    per_state = []
    for state_id in range(n_states):
        per_state.append({
            'state': state_id,
            'model_history_usage': (
                None if model_usage is None else float(model_usage[state_id])
            ),
            'pooled_decoded_occupancy': float(pooled_occupancy[state_id]),
            'recurrence_score': float(recurrence[state_id]),
            'model_history_active': state_id in model_active_states,
            'pooled_decoded_active': state_id in pooled_active_states,
            'recurrence_noninactive': state_id in recurrence_noninactive_states,
        })

    return {
        'n_total_trs': total_trs,
        'fo_threshold': float(fo_threshold),
        'model_usage_threshold': float(model_usage_threshold),
        'definitions': {
            'model_history_active': (
                'Final training-iteration usage > model_usage_threshold '
                '(train+valid fit history from best_model.pkl).'
            ),
            'pooled_decoded_active': (
                'Pooled occupancy across all decoded runs > fo_threshold.'
            ),
            'recurrence_noninactive': (
                'Recurrence score > 0, i.e. state exceeds fo_threshold in at least one decoded run.'
            ),
        },
        'counts': {
            'model_history_active': len(model_active_states),
            'pooled_decoded_active': len(pooled_active_states),
            'recurrence_noninactive': len(recurrence_noninactive_states),
        },
        'state_sets': {
            'model_history_active': model_active_states,
            'pooled_decoded_active': pooled_active_states,
            'recurrence_noninactive': recurrence_noninactive_states,
            'episodically_present_but_model_subthreshold': episodic_vs_model,
            'episodically_present_but_pooled_subthreshold': episodic_vs_pooled,
            'model_active_but_recurrence_inactive': model_vs_recurrence,
            'pooled_active_but_recurrence_inactive': pooled_vs_recurrence,
        },
        'model_history_usage': model_usage_serializable,
        'pooled_decoded_occupancy': pooled_occupancy.tolist(),
        'per_state': per_state,
    }


# compute_specificity_index: moved to utils/recurrence_utils.py


def permutation_test_specificity(fo, n_states, available_seasons, fo_threshold, n_permutations=5000, seed=42):
    """Permutation test for season-specificity of each state.

    Null: season labels are randomly assigned to runs (preserving per-season counts).
    Test statistic: season-specificity index (range of per-season recurrence).

    Uses the standard finite-sampling correction: p = (count + 1) / (n_perm + 1)
    (Phipson & Smyth, 2010) to avoid zero p-values.

    Season labels are shuffled and passed as a direct mapping to
    compute_per_season_recurrence() via season_override, avoiding the
    key collision bug from re-keying run IDs.

    Args:
        fo: dict run_id -> np.array(n_states,) (run-level FO)
        n_states: Number of HMM states
        available_seasons: List of season ints with data
        fo_threshold: Minimum FO to count as active
        n_permutations: Number of permutations
        seed: Random seed

    Returns:
        p_values: np.array(n_states,) — fraction of permutations where surrogate
                  specificity >= observed specificity.
    """
    # Observed
    _, season_rec_obs = compute_per_season_recurrence(fo, n_states, available_seasons, fo_threshold)
    observed_spec = compute_specificity_index(season_rec_obs)

    run_ids = list(fo.keys())
    season_labels = np.array([_get_season(r) for r in run_ids])

    rng = np.random.default_rng(seed)
    perm_greater_count = np.zeros(n_states)

    logger.info(f"Running {n_permutations} permutations for season-specificity...")
    for i in range(n_permutations):
        if i > 0 and i % 1000 == 0:
            logger.info(f"  {i}/{n_permutations} permutations done")

        shuffled_labels = rng.permutation(season_labels)
        season_override = dict(zip(run_ids, shuffled_labels.tolist()))

        _, season_rec_perm = compute_per_season_recurrence(
            fo, n_states, available_seasons, fo_threshold,
            season_override=season_override,
        )
        perm_spec = compute_specificity_index(season_rec_perm)
        perm_greater_count += (perm_spec >= observed_spec).astype(float)

    # Finite-sampling correction (Phipson & Smyth, 2010): avoids zero p-values
    return (perm_greater_count + 1) / (n_permutations + 1)


# =============================================================================
# Main Pipeline
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze recurrence of combined HMM states.")
    parser.add_argument('--sub_id', type=str, required=True, help="Subject ID (e.g., sub-01)")
    parser.add_argument('--parcellation', type=str, default='atlas-4S156Parcels')
    
    parser.add_argument('--fo_threshold', type=float, default=0.02,
                        help="Minimum FO to count a state as active in a run (default: 0.02)")
    parser.add_argument('--model_usage_threshold', type=float, default=0.01,
                        help="Threshold for model-history active states in the activity comparison (default: 0.01)")
    
    parser.add_argument('--n_permutations', type=int, default=5000,
                        help="Number of permutations for specificity test (default: 5000)")
    parser.add_argument('--threshold_sweep', action='store_true',
                        help="Compute recurrence at multiple FO thresholds to check stability")
    parser.add_argument('--vt', type=str, default=None,
                        help="Variance threshold subdirectory under final/ (e.g., 0.99). "
                             "Reads from final/vt{VT}/. If omitted, reads from final/ directly "
                             "(legacy path).")

    return parser.parse_args()


def main():
    args = parse_args()
    sub_id = args.sub_id
    parc = normalize_parcellation_name(args.parcellation)

    # Input paths (from 04 select mode)
    if args.vt is not None:
        hmm_base = os.path.join(SCRATCH_DIR, 'output', '04_combined_hdphmm', parc, sub_id,
                                'final', f'vt{args.vt}')
    else:
        hmm_base = os.path.join(SCRATCH_DIR, 'output', '04_combined_hdphmm', parc, sub_id, 'final')
    decoded_path = os.path.join(hmm_base, 'decoded_states.pkl')
    results_path = os.path.join(hmm_base, 'final_results.json')
    best_model_path = os.path.join(hmm_base, 'best_model.pkl')

    if not os.path.exists(decoded_path):
        logger.error(f"Missing decoded states: {decoded_path}")
        logger.error("Run 04_combined_hdphmm.py --mode select first.")
        sys.exit(1)
        
    if not os.path.exists(results_path):
        logger.error(f"Missing final results: {results_path}")
        sys.exit(1)

    # Output paths (vt-aware to avoid overwriting across variance thresholds)
    out_dir = os.path.join(SCRATCH_DIR, 'output', '05a_recurrence_analysis', parc, sub_id)
    if args.vt is not None:
        out_dir = os.path.join(out_dir, f'vt{args.vt}')
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Loading decoded states from {decoded_path}")
    with open(decoded_path, 'rb') as f:
        decoded_states = pickle.load(f)
        
    with open(results_path, 'r') as f:
        final_results = json.load(f)
    n_states = final_results['model_info']['n_states']
    logger.info(f"Loaded HMM with {n_states} states across {len(decoded_states)} runs")

    # 1. FO & Recurrence (run-level — no episode aggregation)
    fo = compute_fractional_occupancy(decoded_states, n_states)
    n_runs = len(fo)
    logger.info(f"Computed run-level FO for {n_runs} runs (no episode aggregation)")

    # Log FO threshold in hemodynamic terms
    run_lengths = {rid: len(seq) for rid, seq in decoded_states.items()}
    mean_run_trs = np.mean(list(run_lengths.values())) if run_lengths else 0
    fo_trs = args.fo_threshold * mean_run_trs
    fo_seconds = fo_trs * TR_SECONDS
    logger.info(
        f"FO threshold {args.fo_threshold} ≈ {fo_trs:.1f} TRs ≈ {fo_seconds:.1f}s "
        f"at TR={TR_SECONDS}s (mean run length {mean_run_trs:.0f} TRs)"
    )

    recurrence = compute_recurrence_scores(fo, n_states, args.fo_threshold)
    plot_recurrence_histogram(recurrence, out_dir, sub_id)

    active_states = [k for k in range(n_states) if recurrence[k] > 0]
    inactive_states = [k for k in range(n_states) if recurrence[k] == 0.0]
    n_active = len(active_states)
    logger.info(
        "Recurrence summary (FO threshold = %s): %d active states, %d inactive (never meets FO threshold)",
        args.fo_threshold, n_active, len(inactive_states),
    )

    # Low-confidence flagging: states with <10 TRs total across all runs
    total_trs_per_state = np.zeros(n_states)
    for run_id, seq in decoded_states.items():
        for s in seq:
            total_trs_per_state[int(s)] += 1
    low_confidence_states = [
        int(k) for k in range(n_states)
        if total_trs_per_state[k] > 0 and total_trs_per_state[k] < 10
    ]
    if low_confidence_states:
        logger.warning(
            "Low-confidence states (decoded but <10 TRs total): %s",
            low_confidence_states,
        )

    # 2. Season Specificity
    all_seasons = sorted(set(_get_season(r) for r in fo.keys()))
    # Filter to seasons with actual runs (guard against subjects with 0 runs in some seasons)
    seasons_with_data = [
        s for s in all_seasons
        if any(_get_season(ep) == s for ep in fo.keys())
    ]
    if len(seasons_with_data) < len(all_seasons):
        empty_seasons = sorted(set(all_seasons) - set(seasons_with_data))
        logger.warning(
            "Seasons with 0 runs skipped from specificity analysis: %s",
            empty_seasons,
        )
    all_seasons = seasons_with_data

    per_season_mean_fo, season_recurrence = compute_per_season_recurrence(
        fo, n_states, all_seasons, args.fo_threshold
    )
    specificity_index = compute_specificity_index(season_recurrence)

    # 3. Permutation Test + FDR correction
    # Exclude low-confidence states (<10 TRs) from permutation testing: their
    # sparse occupancy produces high specificity by chance, inflating false
    # positives.  Set their p-values to 1.0 (not significant) so they don't
    # consume FDR budget.
    low_conf_set = set(low_confidence_states)
    if len(all_seasons) >= 2:
        p_values = permutation_test_specificity(
            fo, n_states, all_seasons, args.fo_threshold, n_permutations=args.n_permutations
        )
        for k in low_conf_set:
            p_values[k] = 1.0
        fdr_pvalues = benjamini_hochberg(p_values)
    else:
        logger.warning(
            "Fewer than 2 seasons with data (%d); skipping permutation test.",
            len(all_seasons),
        )
        p_values = np.ones(n_states)
        fdr_pvalues = np.ones(n_states)

    significant_specific_uncorrected = [
        k for k in active_states if p_values[k] < 0.05
    ]
    significant_specific = [
        k for k in active_states if fdr_pvalues[k] < 0.05
    ]
    logger.info(f"  Season-specific states (uncorrected p<0.05): {len(significant_specific_uncorrected)}")
    logger.info(f"  Season-specific states (FDR q<0.05):         {len(significant_specific)}")

    # 4. Activity-definition comparison
    activity_definition_summary = build_activity_definition_summary(
        decoded_states,
        recurrence,
        n_states,
        args.fo_threshold,
        best_model_path,
        args.model_usage_threshold,
    )
    activity_definition_path = os.path.join(
        out_dir, 'activity_definition_comparison.json'
    )
    with open(activity_definition_path, 'w') as f:
        json.dump({
            'sub_id': sub_id,
            'parcellation': parc,
            'n_states': n_states,
            'n_runs': len(decoded_states),
            **activity_definition_summary,
        }, f, indent=2)
    logger.info(
        "  Activity definitions: model-history=%d pooled=%d recurrence=%d",
        activity_definition_summary['counts']['model_history_active'],
        activity_definition_summary['counts']['pooled_decoded_active'],
        activity_definition_summary['counts']['recurrence_noninactive'],
    )

    # 5. State-level dwell/recurrence summary for all states
    state_metric_rows = build_state_recurrence_dwell_metrics(
        decoded_states,
        fo,
        recurrence,
        n_states,
        args.fo_threshold,
        tr_seconds=TR_SECONDS,
    )
    state_metrics_path = os.path.join(out_dir, 'state_recurrence_dwell_metrics.csv')
    write_records_csv(state_metrics_path, STATE_METRIC_FIELDNAMES, state_metric_rows)
    dwell_recurrence_plot_path = plot_state_dwell_vs_recurrence(
        state_metric_rows,
        out_dir,
        sub_id,
    )
    dwell_recurrence_heatmap_path = plot_state_dwell_recurrence_heatmap(
        state_metric_rows,
        out_dir,
        sub_id,
    )

    # 6. Run-wise state diversity and revisitation summary
    run_metric_rows = build_run_metrics(
        decoded_states,
        fo,
        args.fo_threshold,
        tr_seconds=TR_SECONDS,
    )
    run_metrics_path = os.path.join(out_dir, 'run_state_metrics.csv')
    write_records_csv(run_metrics_path, RUN_METRIC_FIELDNAMES, run_metric_rows)
    run_plot_path = plot_run_state_diversity(
        run_metric_rows,
        out_dir,
        sub_id,
    )

    # 7. Season breakdown for global recurring and partial states
    seasonal_rows, seasonal_fieldnames, seasonal_summary = build_seasonal_recurrence_breakdown(
        season_recurrence,
        recurrence,
        all_seasons,
    )
    seasonal_breakdown_path = os.path.join(out_dir, 'seasonal_recurrence_breakdown.csv')
    write_records_csv(seasonal_breakdown_path, seasonal_fieldnames, seasonal_rows)
    seasonal_heatmap_path = plot_seasonal_recurrence_heatmap(
        seasonal_rows,
        all_seasons,
        out_dir,
        sub_id,
    )
    seasonal_breakdown_json_path = os.path.join(
        out_dir, 'seasonal_recurrence_breakdown.json'
    )
    with open(seasonal_breakdown_json_path, 'w') as f:
        json.dump({
            'sub_id': sub_id,
            'parcellation': parc,
            'seasons_analyzed': all_seasons,
            **seasonal_summary,
        }, f, indent=2)

    logger.info(
        "  Run-wise summary: %d runs | mean unique states %.2f | mean revisitation ratio %.2f",
        len(run_metric_rows),
        np.mean([row['n_unique_states_decoded'] for row in run_metric_rows]) if run_metric_rows else 0.0,
        np.mean([row['within_run_recurrence_ratio'] for row in run_metric_rows]) if run_metric_rows else 0.0,
    )
    logger.info(
        "  Season breakdown: across-season recurrent=%d | seasonal recurrent=%d",
        len(seasonal_summary['across_season_recurrent_states']),
        len(seasonal_summary['seasonal_recurrent_states']),
    )

    # 8. Export contiguous blocks for all active states
    active_state_set = set(active_states)
    block_records = extract_state_block_records(
        decoded_states,
        recurrence,
        include_states=active_state_set,
        tr_seconds=TR_SECONDS,
    )
    run_state_rows = summarize_block_records(
        block_records,
        recurrence,
        args.fo_threshold,
        tr_seconds=TR_SECONDS,
    )
    block_path = os.path.join(out_dir, 'state_blocks.csv.gz')
    run_state_path = os.path.join(out_dir, 'run_state_summary.csv.gz')
    write_records_csv(block_path, BLOCK_FIELDNAMES, block_records)
    write_records_csv(
        run_state_path,
        EPISODE_STATE_FIELDNAMES,
        run_state_rows,
    )
    logger.info(
        "  Saved %d contiguous state blocks across %d run-state rows",
        len(block_records),
        len(run_state_rows),
    )

    # 9. Save Outputs
    # Save run-level FO (used by downstream scripts 05b, 05c, 05d)
    with open(os.path.join(out_dir, 'fractional_occupancy.pkl'), 'wb') as f:
        pickle.dump(fo, f, protocol=4)
        
    np.save(os.path.join(out_dir, 'recurrence_scores.npy'), recurrence)
    np.save(os.path.join(out_dir, 'specificity_index.npy'), specificity_index)
    
    with open(os.path.join(out_dir, 'per_season_mean_fo.json'), 'w') as f:
        json.dump({str(s): arr.tolist() for s, arr in per_season_mean_fo.items()}, f, indent=2)
        
    with open(os.path.join(out_dir, 'per_season_recurrence.json'), 'w') as f:
        json.dump({str(s): arr.tolist() for s, arr in season_recurrence.items()}, f, indent=2)
        
    with open(os.path.join(out_dir, 'permutation_pvalues.json'), 'w') as f:
        json.dump({
            'uncorrected': p_values.tolist(),
            'fdr_corrected': fdr_pvalues.tolist(),
        }, f, indent=2)
    # Collect sub-HRF states from the already-computed flag (median-based)
    sub_hrf_states = [
        row['state'] for row in state_metric_rows if row['sub_hrf_resolution']
    ]
    if sub_hrf_states:
        logger.warning(
            "States with median dwell < %.1f TRs (sub-HRF resolution): %s",
            SUB_HRF_DWELL_THRESHOLD_TR,
            sub_hrf_states,
        )

    # Export eligible_states.json for downstream scripts
    eligible_state_ids = [
        row['state'] for row in state_metric_rows
        if not row['sub_hrf_resolution']
        and row['recurrence_score'] > 0
        and row['decoded_usage_status'] != 'never_decoded'
    ]
    eligible_states_payload = {
        'eligible_state_ids': eligible_state_ids,
        'excluded_sub_hrf_state_ids': sub_hrf_states,
        'criterion': 'median_dwell_tr',
        'threshold_tr': SUB_HRF_DWELL_THRESHOLD_TR,
        'threshold_s': round(SUB_HRF_DWELL_THRESHOLD_TR * TR_SECONDS, 2),
        'note': (
            'States whose median block dwell time is below the HRF peak '
            '(~4.5s). Excluded states may reflect hemodynamic transition '
            'artifacts or have insufficient BOLD evidence for reliable '
            'characterization.'
        ),
    }
    eligible_path = os.path.join(out_dir, 'eligible_states.json')
    with open(eligible_path, 'w') as f:
        json.dump(eligible_states_payload, f, indent=2)
    logger.info(
        "Exported eligible_states.json: %d eligible, %d sub-HRF excluded",
        len(eligible_state_ids),
        len(sub_hrf_states),
    )

    # Compute mean_fo_when_active for each state
    fo_matrix = np.stack(list(fo.values()))  # (n_runs, n_states)
    mean_fo_when_active = np.zeros(n_states)
    for k in range(n_states):
        active_mask = fo_matrix[:, k] > args.fo_threshold
        if active_mask.any():
            mean_fo_when_active[k] = float(fo_matrix[active_mask, k].mean())

    summary = {
        'analysis_scope': 'single_subject',
        'note': 'FDR correction does not replace cross-subject replication',
        'sub_id': sub_id,
        'parcellation': parc,
        'n_states': n_states,
        'n_runs': n_runs,
        'n_episodes': n_runs,  # backward compat alias (analysis unit is runs, not aggregated episodes)
        'n_active_states': n_active,
        'seasons_analyzed': all_seasons,
        'fo_active_threshold': args.fo_threshold,
        'fo_threshold_trs': round(fo_trs, 1),
        'fo_threshold_seconds': round(fo_seconds, 1),
        'tr_seconds': TR_SECONDS,
        'n_permutations': args.n_permutations,
        'permutation_pvalue_correction': '(count + 1) / (n_permutations + 1)',
        'recurrence_scores': recurrence.tolist(),
        'specificity_index': specificity_index.tolist(),
        'permutation_pvalues': p_values.tolist(),
        'fdr_corrected_pvalues': fdr_pvalues.tolist(),
        'significant_specific_states_uncorrected': significant_specific_uncorrected,
        'significant_specific_states': significant_specific,
        'activity_definition_comparison': {
            'path': os.path.basename(activity_definition_path),
            **activity_definition_summary['counts'],
            'episodically_present_but_model_subthreshold': activity_definition_summary['state_sets']['episodically_present_but_model_subthreshold'],
            'episodically_present_but_pooled_subthreshold': activity_definition_summary['state_sets']['episodically_present_but_pooled_subthreshold'],
        },
        'state_metric_summary': {
            'state_metrics_path': os.path.basename(state_metrics_path),
            'dwell_vs_recurrence_plot_path': os.path.basename(dwell_recurrence_plot_path),
            'dwell_vs_recurrence_heatmap_path': os.path.basename(
                dwell_recurrence_heatmap_path
            ),
            'n_states_with_blocks': int(
                sum(row['n_blocks'] > 0 for row in state_metric_rows)
            ),
        },
        'run_metric_summary': {
            'run_metrics_path': os.path.basename(run_metrics_path),
            'run_diversity_plot_path': os.path.basename(run_plot_path),
            'mean_unique_states_per_run': float(
                np.mean([row['n_unique_states_decoded'] for row in run_metric_rows])
            ) if run_metric_rows else 0.0,
            'mean_within_run_recurrence_ratio': float(
                np.mean([row['within_run_recurrence_ratio'] for row in run_metric_rows])
            ) if run_metric_rows else 0.0,
            'mean_revisited_states_per_run': float(
                np.mean([row['n_revisited_states'] for row in run_metric_rows])
            ) if run_metric_rows else 0.0,
        },
        'seasonal_recurrence_breakdown': {
            'csv_path': os.path.basename(seasonal_breakdown_path),
            'json_path': os.path.basename(seasonal_breakdown_json_path),
            'heatmap_path': os.path.basename(seasonal_heatmap_path),
            **seasonal_summary,
        },
        'block_export': {
            'tr_seconds': TR_SECONDS,
            'state_blocks_path': os.path.basename(block_path),
            'run_state_summary_path': os.path.basename(run_state_path),
            'n_state_blocks': len(block_records),
            'n_run_state_rows': len(run_state_rows),
        },
        'low_confidence_states': low_confidence_states,
        'sub_hrf_resolution_states': sub_hrf_states,
        'mean_fo_when_active': mean_fo_when_active.tolist(),
        'total_trs_per_state': total_trs_per_state.astype(int).tolist(),
        'season_specificity': {
            'specificity_index': specificity_index.tolist(),
            'permutation_pvalues': p_values.tolist(),
            'fdr_corrected_qvalues': fdr_pvalues.tolist(),
            'significant_states_uncorrected': significant_specific_uncorrected,
            'significant_states_fdr': significant_specific,
            'n_seasons_analyzed': len(all_seasons),
            'seasons_with_data': all_seasons,
            'low_confidence_excluded_from_test': sorted(low_conf_set),
            'note': (
                'Permutation test shuffles season labels to test whether '
                'specificity exceeds chance. This tests label-data association '
                'and cannot distinguish narrative-content effects from '
                'longitudinal confounds (session order, scanner drift). '
                'Low-confidence states (<10 TRs) are excluded from testing '
                '(p-value set to 1.0) to avoid inflated false positives from '
                'sparse occupancy.'),
        },
    }

    # Optional Threshold Sweep — shows n_active at different FO thresholds
    if args.threshold_sweep:
        logger.info("\nRunning threshold sweep (FO sensitivity)...")
        fo_thresholds = [0.005, 0.01, 0.02, 0.05]
        rec_levels = [0.25, 0.50, 0.75, 0.90]
        sweep_results = {}
        for fo_th in fo_thresholds:
            rec_at_th = compute_recurrence_scores(fo, n_states, fo_th)
            key = f"fo={fo_th}"
            sweep_results[key] = {
                'fo_threshold': fo_th,
                'n_active': int(np.sum(rec_at_th > 0)),
            }
            for level in rec_levels:
                sweep_results[key][f'n_above_{level}'] = int(np.sum(rec_at_th >= level))
        summary['threshold_sweep'] = sweep_results
        logger.info(
            "Sensitivity table: %d FO thresholds computed", len(fo_thresholds)
        )

    with open(os.path.join(out_dir, 'recurrence_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
        
    logger.info(f"\nOutputs saved to {out_dir}")

if __name__ == '__main__':
    main()
