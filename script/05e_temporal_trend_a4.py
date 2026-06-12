#!/usr/bin/env python3
"""
05e_temporal_trend_a4.py - State flag synthesis.

Synthesizes per-state temporal metrics from a1 (cross-episode trends),
a2 (within-run position anchoring), and a3 (within-session habituation)
into a unified state classification. Each state receives boolean tags
and a summary category for downstream filtering.

Tags (non-mutually-exclusive):
    sub_hrf             Median dwell < 3 TRs; BOLD evidence degraded
    unused              Recurrence == 0; never assigned by decoder
    run_onset           Locked to start of both a and b runs (ab_start_common)
    a_anchored          Locked to start of a-runs only (a_start_specific)
    b_anchored          Locked to start of b-runs only (b_start_specific)
    session_trend_down  FO decreases within session (LME slope < 0, q < alpha)
                        [INFORMATIONAL - handled via detrended FO, not exclusion]
    session_trend_up    FO increases within session (LME slope > 0, q < alpha)
                        [INFORMATIONAL - handled via detrended FO, not exclusion]
    season_structured   FO varies by season identity (q < alpha)
    global_trend        FO trends with global position (q < alpha, from a1 Scale 3)

Summary categories (mutually exclusive, priority order):
    1. unused           2. low_confidence    3. run_onset_anchored
    4. season_temporal (only if season_structured AND global_trend)
    5. eligible_for_content_analysis
    6. rare (recurrence < floor, no other tags)

Note: session trends no longer exclude states. Within-session FO drift is
addressed via detrended FO output from a3 (fractional_occupancy_detrended.pkl).
Season structure only excludes when co-occurring with a global temporal trend,
since season-only FO variation likely reflects content differences.

Prerequisites:
    - 05a_recurrence_analysis completed
    - 05e_temporal_trend_a1 completed (or at least a1 CSV present)
    - 05e_temporal_trend_a2 completed (or at least a2 CSV present)
    - 05e_temporal_trend_a3 completed (or at least a3 CSV present)

Outputs (saved to {SCRATCH_DIR}/output/05e_temporal_trend_a4/{parc}/{sub_id}/[vt{VT}/]):
    - state_flags.csv           per-state tags + summary category
    - state_flags_summary.json  counts, thresholds, source availability
    - state_flag_overview.png/pdf   binary heatmap

See also:
    05e_temporal_trend_a1.py - cross-episode temporal trends
    05e_temporal_trend_a2.py - within-run temporal position
    05e_temporal_trend_a3.py - within-session FO habituation (LME)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils.plot_style import (
    apply_publication_style,
    recurrence_color,
    load_parcel_networks, compute_dominant_networks,
)
from utils.common import normalize_parcellation_name
from utils.state_blocks import load_eligible_states
from utils.state_flags_io import (
    TAG_COLUMNS as _TAG_COLUMNS,
    TAG_COLORS as _TAG_COLORS,
    CATEGORY_PRIORITY as _CATEGORY_PRIORITY,
    CATEGORY_COLORS as _CATEGORY_COLORS,
)

from dotenv import load_dotenv

load_dotenv()
SCRATCH_DIR = os.getenv('SCRATCH_DIR')

logger = logging.getLogger(__name__)

# Shared constants (canonical definitions in utils/state_flags_io.py)
TAG_COLUMNS = _TAG_COLUMNS
CATEGORY_PRIORITY = _CATEGORY_PRIORITY
TAG_COLORS = _TAG_COLORS
CATEGORY_COLORS = _CATEGORY_COLORS


# =============================================================================
# Data loading
# =============================================================================

def load_upstream_data(sub_id, parcellation, vt, scratch_dir):
    """Load recurrence data + a1/a2/a3 CSVs. Returns merged DataFrame."""

    parc_short = normalize_parcellation_name(parcellation)
    parc_full = f'atlas-{parc_short}' if not parcellation.startswith('atlas-') else parcellation
    vt_dir = f'vt{vt}' if vt else ''

    # -- 05a: recurrence scores + eligible states --
    recurrence_dir = os.path.join(
        scratch_dir, 'output', '05a_recurrence_analysis', parc_full, sub_id,
        vt_dir)
    recurrence_scores = np.load(os.path.join(recurrence_dir, 'recurrence_scores.npy'))
    n_states = len(recurrence_scores)

    eligible_ids, excluded_ids, _ = load_eligible_states(recurrence_dir)
    is_sub_hrf = np.zeros(n_states, dtype=bool)
    for sid in excluded_ids:
        is_sub_hrf[sid] = True

    # Base DataFrame
    base_df = pd.DataFrame({
        'state': np.arange(n_states),
        'recurrence_score': recurrence_scores,
        'is_sub_hrf': is_sub_hrf,
    })
    sources = {}

    # a1
    a1_path = os.path.join(
        scratch_dir, 'output', '05e_temporal_trend_a1',
        parc_full, sub_id, vt_dir, 'temporal_trend_metrics.csv')
    a1_df = _try_load_csv(a1_path, 'a1')
    sources['a1'] = a1_df is not None

    # a2
    a2_path = os.path.join(
        scratch_dir, 'output', '05e_temporal_trend_a2',
        parc_full, sub_id, vt_dir, 'temporal_position_metrics.csv')
    a2_df = _try_load_csv(a2_path, 'a2')
    sources['a2'] = a2_df is not None

    # a3
    a3_path = os.path.join(
        scratch_dir, 'output', '05e_temporal_trend_a3',
        parc_full, sub_id, vt_dir, 'habituation_metrics.csv')
    a3_df = _try_load_csv(a3_path, 'a3')
    sources['a3'] = a3_df is not None

    # Merge
    df = base_df.copy()
    if a1_df is not None:
        a1_cols = ['state', 'q_s3_season', 'q_s3_global', 'dr2_season']
        available = [c for c in a1_cols if c in a1_df.columns]
        df = df.merge(a1_df[available], on='state', how='left')

    if a2_df is not None:
        a2_cols = ['state', 'anchoring_type', 'early_fraction_a', 'early_fraction_b']
        available = [c for c in a2_cols if c in a2_df.columns]
        df = df.merge(a2_df[available], on='state', how='left')

    if a3_df is not None:
        a3_cols = ['state', 'lme_slope', 'lme_se', 'lme_icc', 'q_fdr']
        available = [c for c in a3_cols if c in a3_df.columns]
        df = df.merge(a3_df[available], on='state', how='left')

    logger.info("Loaded data for %d states. Sources: %s", n_states, sources)
    return df, sources, n_states


def _try_load_csv(path, label):
    """Load a CSV, returning None if not found."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        logger.info("Loaded %s: %s (%d rows, %d cols)", label, path,
                     len(df), len(df.columns))
        return df
    else:
        logger.warning("%s CSV not found: %s", label, path)
        return None


# =============================================================================
# Tag computation
# =============================================================================

def compute_tags(df, alpha=0.05):
    """Add boolean tag columns to the DataFrame."""

    n = len(df)

    # sub_hrf (from 05a)
    df['sub_hrf'] = df['is_sub_hrf'].fillna(False).astype(bool)

    # unused
    df['unused'] = df['recurrence_score'] == 0.0

    # Position anchoring (from a2)
    anchoring = df.get('anchoring_type', pd.Series([''] * n))
    anchoring = anchoring.fillna('')
    df['run_onset'] = anchoring == 'ab_start_common'
    df['a_anchored'] = anchoring == 'a_start_specific'
    df['b_anchored'] = anchoring == 'b_start_specific'

    # Session trends (from a3)
    q_a3 = df.get('q_fdr', pd.Series([np.nan] * n))
    slope = df.get('lme_slope', pd.Series([np.nan] * n))
    sig_a3 = q_a3.lt(alpha) & q_a3.notna()
    df['session_trend_down'] = sig_a3 & slope.lt(0)
    df['session_trend_up'] = sig_a3 & slope.gt(0)

    # Season structured (from a1)
    q_season = df.get('q_s3_season', pd.Series([np.nan] * n))
    df['season_structured'] = q_season.lt(alpha) & q_season.notna()

    # Global trend (from a1 Scale 3 - global position predictor)
    q_global = df.get('q_s3_global', pd.Series([np.nan] * n))
    df['global_trend'] = q_global.lt(alpha) & q_global.notna()

    return df


# =============================================================================
# Summary category
# =============================================================================

def compute_summary_category(df, recurrence_floor=0.10):
    """Assign mutually exclusive summary category per state (priority order).

    Session trends (session_trend_down/up) are informational tags only - they
    do NOT exclude states.  Session-level FO drift is handled via detrended FO
    output from a3 rather than state exclusion.

    Season structure excludes only when co-occurring with a significant global
    temporal trend (global_trend), indicating genuine longitudinal drift rather
    than content-driven seasonal FO variation.
    """
    categories = []
    for _, row in df.iterrows():
        if row['unused']:
            categories.append('unused')
        elif row['sub_hrf']:
            categories.append('low_confidence')
        elif row['run_onset'] or row['a_anchored'] or row['b_anchored']:
            categories.append('run_onset_anchored')
        elif row['season_structured'] and row['global_trend']:
            categories.append('season_temporal')
        elif row['recurrence_score'] >= recurrence_floor:
            categories.append('eligible_for_content_analysis')
        else:
            categories.append('rare')

    df['summary_category'] = categories
    return df


# =============================================================================
# Network annotation
# =============================================================================

def annotate_networks(df, sub_id, parcellation, vt, scratch_dir):
    """Add dominant_network column using compute_dominant_networks()."""
    parc_short = normalize_parcellation_name(parcellation)
    parc_full = f'atlas-{parc_short}' if not parcellation.startswith('atlas-') else parcellation
    vt_dir = f'vt{vt}' if vt else ''

    # Use pre-computed state_means_parcel.npy (same as a3)
    means_path = os.path.join(
        scratch_dir, 'output', '04_combined_hdphmm',
        parc_full, sub_id, 'final', vt_dir, 'state_means_parcel.npy')

    if not os.path.exists(means_path):
        logger.warning("state_means_parcel.npy not found: %s", means_path)
        df['dominant_network'] = ''
        return df

    state_means = np.load(means_path)

    parcel_networks = load_parcel_networks(parcellation)
    if parcel_networks is None:
        logger.warning("Could not load parcel networks for %s", parcellation)
        df['dominant_network'] = ''
        return df

    active_states = df.loc[~df['unused'], 'state'].values
    dominant = compute_dominant_networks(
        state_means, active_states, parcel_networks, include_sign=False)

    df['dominant_network'] = df['state'].map(
        lambda s: dominant.get(int(s), ''))
    logger.info("Annotated %d states with dominant network", len(dominant))
    return df


# =============================================================================
# Visualization
# =============================================================================

def plot_flag_heatmap(df, sub_id, out_dir):
    """Binary heatmap: states (sorted by recurrence) × tags."""
    apply_publication_style()

    # Sort by descending recurrence
    df_sorted = df.sort_values('recurrence_score', ascending=False).reset_index(drop=True)
    n_states = len(df_sorted)

    # Build binary matrix
    tag_matrix = df_sorted[TAG_COLUMNS].values.astype(float)

    # Tag display names
    tag_labels = [
        'Sub-HRF', 'Unused', 'Run onset', 'A-anchor', 'B-anchor',
        'Session ↓', 'Session ↑', 'Season', 'Global trend',
    ]

    fig_height = max(4, n_states * 0.11)
    fig, axes = plt.subplots(1, 3, figsize=(7, fig_height),
                              gridspec_kw={'width_ratios': [0.8, 8, 2.5]})

    # -- Left: recurrence colorbar --
    ax_rec = axes[0]
    rec_vals = df_sorted['recurrence_score'].values
    for i, r in enumerate(rec_vals):
        ax_rec.barh(i, 1, height=1, color=recurrence_color(r), edgecolor='none')
    ax_rec.set_xlim(0, 1)
    ax_rec.set_ylim(-0.5, n_states - 0.5)
    ax_rec.invert_yaxis()
    ax_rec.set_yticks(range(n_states))
    ax_rec.set_yticklabels(df_sorted['state'].values)
    ax_rec.set_xticks([])
    ax_rec.set_ylabel('State (sorted by recurrence)')
    ax_rec.set_xlabel('Rec.')

    # -- Center: per-column colored heatmap --
    ax_heat = axes[1]
    ax_heat.set_facecolor('#F0F0F0')
    for j, tag in enumerate(TAG_COLUMNS):
        color = TAG_COLORS[tag]
        for i in range(n_states):
            if tag_matrix[i, j]:
                ax_heat.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor=color, edgecolor='none'))
    ax_heat.set_xticks(range(len(tag_labels)))
    ax_heat.set_xticklabels(tag_labels, ha='center', fontsize=7)
    ax_heat.xaxis.tick_top()
    ax_heat.xaxis.set_tick_params(pad=2)
    ax_heat.set_yticks([])
    ax_heat.set_xlim(-0.5, len(tag_labels) - 0.5)
    ax_heat.set_ylim(n_states - 0.5, -0.5)

    # Add grid lines
    for i in range(n_states + 1):
        ax_heat.axhline(i - 0.5, color='white', linewidth=0.3)
    for j in range(len(tag_labels) + 1):
        ax_heat.axvline(j - 0.5, color='white', linewidth=0.3)

    # Tag counts below heatmap
    tag_counts = tag_matrix.sum(axis=0).astype(int)
    for j, count in enumerate(tag_counts):
        ax_heat.text(j, n_states + 0.3, str(count), ha='center', va='top',
                     fontsize=7, color='#666666')

    # -- Right: summary category (merge adjacent same-category rows) --
    ax_cat = axes[2]
    categories = df_sorted['summary_category'].values
    # Find runs of identical categories
    blocks = []
    start = 0
    for i in range(1, len(categories)):
        if categories[i] != categories[start]:
            blocks.append((start, i - 1, categories[start]))
            start = i
    blocks.append((start, len(categories) - 1, categories[start]))

    light_cats = {'eligible_for_content_analysis', 'rare'}
    for row_start, row_end, cat in blocks:
        color = CATEGORY_COLORS.get(cat, '#CCCCCC')
        height = row_end - row_start + 1
        ax_cat.add_patch(Rectangle(
            (0, row_start - 0.5), 1, height,
            facecolor=color, edgecolor='white', linewidth=0.5))
        mid_y = (row_start + row_end) / 2
        text_color = '#333333' if cat in light_cats else 'white'
        label = cat.replace('_', ' ')
        ax_cat.text(0.5, mid_y, label, ha='center', va='center',
                    fontsize=4.5 if height == 1 else 5, color=text_color)
    ax_cat.set_xlim(0, 1)
    ax_cat.set_ylim(-0.5, n_states - 0.5)
    ax_cat.invert_yaxis()
    ax_cat.set_yticks([])
    ax_cat.set_xticks([])

    # Manually position axes: heatmap flush against category panel
    left_margin = 0.08
    rec_w = 0.025
    gap_rec = 0.04
    cat_w = 0.18
    right_margin = 0.01
    heat_x = left_margin + rec_w + gap_rec
    heat_w = 1.0 - heat_x - cat_w - right_margin
    cat_x = heat_x + heat_w

    top = 0.95
    bottom = 0.05
    h = top - bottom

    axes[0].set_position([left_margin, bottom, rec_w, h])
    axes[1].set_position([heat_x, bottom, heat_w, h])
    axes[2].set_position([cat_x, bottom, cat_w, h])

    for ext in ('png', 'pdf'):
        out_path = os.path.join(out_dir, f'state_flag_overview.{ext}')
        fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved state flag heatmap to %s", out_dir)


# =============================================================================
# Save outputs
# =============================================================================

def save_outputs(df, sources, alpha, recurrence_floor, sub_id, out_dir,
                 a3_dir=None):
    """Save CSV + JSON summary."""

    # -- CSV --
    csv_cols = ['state', 'recurrence_score'] + TAG_COLUMNS + [
        'summary_category', 'dominant_network',
        'lme_slope', 'early_fraction_a', 'early_fraction_b', 'dr2_season',
    ]
    # Only include columns that exist
    csv_cols = [c for c in csv_cols if c in df.columns]
    csv_path = os.path.join(out_dir, 'state_flags.csv')
    df[csv_cols].to_csv(csv_path, index=False)
    logger.info("Saved %s (%d states)", csv_path, len(df))

    # -- JSON summary --
    tag_counts = {tag: int(df[tag].sum()) for tag in TAG_COLUMNS}
    cat_counts = df['summary_category'].value_counts().to_dict()
    # Ensure all categories present
    for cat in CATEGORY_PRIORITY:
        cat_counts.setdefault(cat, 0)

    summary = {
        'sub_id': sub_id,
        'n_states': len(df),
        'alpha': alpha,
        'recurrence_floor': recurrence_floor,
        'sources_available': sources,
        'tag_counts': tag_counts,
        'category_counts': {k: cat_counts.get(k, 0) for k in CATEGORY_PRIORITY},
        'tag_definitions': {
            'sub_hrf': 'Median dwell < 3 TRs; BOLD evidence degraded',
            'unused': 'Recurrence == 0; never assigned by decoder',
            'run_onset': 'Locked to start of both a and b runs (from a2 anchoring_type)',
            'a_anchored': 'Locked to start of a-runs only (from a2)',
            'b_anchored': 'Locked to start of b-runs only (from a2)',
            'session_trend_down': f'FO decreases within session (a3 LME slope < 0, q < {alpha}) [informational]',
            'session_trend_up': f'FO increases within session (a3 LME slope > 0, q < {alpha}) [informational]',
            'season_structured': f'FO varies by season identity (a1 Scale 3 q < {alpha})',
            'global_trend': f'FO trends with global position (a1 Scale 3 q < {alpha})',
        },
        'category_definitions': {
            'unused': 'Recurrence == 0',
            'low_confidence': 'Sub-HRF (median dwell < 3 TRs)',
            'run_onset_anchored': 'Position-anchored to run boundaries',
            'season_temporal': 'Significant season effect AND global temporal trend (co-occurring drift)',
            'eligible_for_content_analysis': f'No structural/temporal tags, recurrence >= {recurrence_floor}',
            'rare': f'Recurrence < {recurrence_floor}, no other tags',
        },
    }

    # Detrended FO metadata
    detrended_fo_path = None
    if a3_dir:
        candidate = os.path.join(a3_dir, 'fractional_occupancy_detrended.pkl')
        if os.path.exists(candidate):
            detrended_fo_path = candidate
    summary['detrended_fo'] = {
        'available': detrended_fo_path is not None,
        'path': detrended_fo_path,
        'description': (
            'Session-detrended FO from a3 (LME slope removed). '
            'Use for content analysis when session trends are a concern.'
        ),
    }

    json_path = os.path.join(out_dir, 'state_flags_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved %s", json_path)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='05e_temporal_trend_a4: state flag synthesis')
    parser.add_argument('--sub_id', required=True)
    parser.add_argument('--parcellation', default='atlas-4S156Parcels')
    parser.add_argument('--vt', type=float, default=0.95)
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='FDR threshold for a1/a3 q-values (default: 0.05)')
    parser.add_argument('--recurrence_floor', type=float, default=0.10,
                        help='Recurrence below this → "rare" category (default: 0.10)')
    parser.add_argument('--no_network', action='store_true',
                        help='Skip network annotation (avoids loading model)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    parc_short = normalize_parcellation_name(args.parcellation)
    parc_full = f'atlas-{parc_short}' if not args.parcellation.startswith('atlas-') else args.parcellation
    vt_dir = f'vt{args.vt}' if args.vt else ''
    out_dir = os.path.join(
        SCRATCH_DIR, 'output', '05e_temporal_trend_a4',
        parc_full, args.sub_id, vt_dir)
    os.makedirs(out_dir, exist_ok=True)

    logger.info("=== 05e_a4 State Flag Synthesis ===")
    logger.info("Subject: %s, Parcellation: %s, VT: %s", args.sub_id, parc_full, args.vt)

    # Load and merge data
    df, sources, n_states = load_upstream_data(
        args.sub_id, args.parcellation, args.vt, SCRATCH_DIR)

    # Compute tags
    df = compute_tags(df, alpha=args.alpha)

    # Summary category
    df = compute_summary_category(df, recurrence_floor=args.recurrence_floor)

    # Network annotation
    if not args.no_network:
        df = annotate_networks(
            df, args.sub_id, args.parcellation, args.vt, SCRATCH_DIR)
    else:
        df['dominant_network'] = ''

    # Log summary
    for tag in TAG_COLUMNS:
        logger.info("  %-22s %3d states", tag, df[tag].sum())
    logger.info("---")
    for cat in CATEGORY_PRIORITY:
        count = (df['summary_category'] == cat).sum()
        if count > 0:
            logger.info("  %-22s %3d states", cat, count)

    # Save (pass a3 output dir for detrended FO path)
    a3_dir = os.path.join(
        SCRATCH_DIR, 'output', '05e_temporal_trend_a3',
        parc_full, args.sub_id, vt_dir)
    save_outputs(df, sources, args.alpha, args.recurrence_floor,
                 args.sub_id, out_dir, a3_dir=a3_dir)

    # Plot
    plot_flag_heatmap(df, args.sub_id, out_dir)

    logger.info("=== Done ===")


if __name__ == '__main__':
    main()
