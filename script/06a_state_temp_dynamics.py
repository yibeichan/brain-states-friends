#!/usr/bin/env python3
"""
06a_state_temp_dynamics.py - Analyze temporal dynamics of brain states.

This script loads the decoded brain state sequences and the classified state
categories (recurring vs. specific) to quantify temporal geometries using
threshold-free continuous measures.

Key Analyses:
1. Contiguous Block Extraction: start/end TRs, duration (TR and seconds).
2. State-Level Summary Table: continuous metrics per state (threshold-free).
3. Transition Dynamics: Transition probability matrices.
4. Continuous Scatters: Recurrence vs dwell, self-transition vs recurrence,
   transition entropy vs recurrence (all threshold-free).
5. Visualization: Sequence barcodes, transition heatmaps.

Prerequisites:
    - 04_combined_hdphmm.py (mode: select) completed.
    - 05a_recurrence_analysis.py completed.

Outputs:
    Saves DataFrames, statistics, and plots to:
    {SCRATCH_DIR}/output/06a_state_temp_dynamics/{parcellation}/{sub_id}/
"""

import os
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path
from itertools import groupby

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv

# Setup paths and logger
sys.path.insert(0, str(Path(__file__).parent))
from utils.plot_style import (
    RECURRENCE_CMAP, recurrence_color,
    apply_publication_style,
    NETWORK_ORDER, NETWORK_COLORS,
    load_parcel_networks, compute_dominant_networks,
)
from utils.state_blocks import TR_SECONDS
from utils.common import normalize_parcellation_name
from utils.state_flags_io import load_state_flags, CATEGORY_MARKERS
from utils.transition_utils import (
    compute_recurrence_assortativity as _shared_assortativity,
)

load_dotenv()
SCRATCH_DIR = os.getenv('SCRATCH_DIR')
if SCRATCH_DIR is None:
    raise ValueError("SCRATCH_DIR must be set in the .env file")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

apply_publication_style()

# Edge-weight threshold for assortativity graph construction.
# Removes near-zero transition probabilities (numerical noise).
_EDGE_THRESH_ASSORTATIVITY = 0.005


def extract_state_blocks(decoded_states, n_states, recurrence_scores):
    """
    Extract contiguous blocks of states from the decoded state sequences.

    Returns a pandas DataFrame where each row is a contiguous state appearance.
    """
    records = []

    for run_id, state_seq in decoded_states.items():
        if len(state_seq) == 0:
            continue

        current_tr = 0
        # Group adjacent identical elements
        for state_val, group in groupby(state_seq):
            duration_tr = sum(1 for _ in group)
            start_tr = current_tr
            end_tr = current_tr + duration_tr

            state_id = int(state_val)

            records.append({
                'run_id': run_id,
                'state': state_id,
                'recurrence_score': float(recurrence_scores[state_id]),
                'start_tr': start_tr,
                'end_tr': end_tr,
                'duration_tr': duration_tr,
                'start_time_s': start_tr * TR_SECONDS,
                'end_time_s': end_tr * TR_SECONDS,
                'duration_s': duration_tr * TR_SECONDS
            })

            current_tr = end_tr

    df = pd.DataFrame(records)
    return df


def calculate_transition_matrix(decoded_states, n_states):
    """
    Compute state-to-state transition probabilities.
    Rows = From State, Columns = To State.
    """
    # Count transitions
    P_counts = np.zeros((n_states, n_states))

    for run_id, state_seq in decoded_states.items():
        if len(state_seq) < 2:
            continue
        # Transitions are consecutive element pairs
        from_states = state_seq[:-1]
        to_states = state_seq[1:]

        # Add to transition matrix
        for i, j in zip(from_states, to_states):
            P_counts[i, j] += 1

    # Convert to probabilities (normalize by row)
    row_sums = P_counts.sum(axis=1, keepdims=True)

    # Avoid division by zero for inactive states
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.nan_to_num(P_counts / row_sums, nan=0.0)

    return P_counts, P


def build_state_summary_table(df_blocks, P, model_transmat, recurrence_scores,
                              n_states, out_dir):
    """
    Build a per-state summary table with continuous, threshold-free metrics.

    Columns: state_id, recurrence_score, n_blocks, median_dwell_s,
    mean_dwell_s, total_occupancy_s, self_transition_prob, transition_entropy.

    Only active states (recurrence_score > 0) are included.

    self_transition_prob comes from the fitted HMM's transmat_ (model parameter),
    NOT from empirical decoded-label counts. This allows the self-transition scatter
    to disentangle model mechanics from observed temporal dynamics.
    """
    from scipy.stats import entropy as shannon_entropy

    rows = []
    for state_id in range(n_states):
        if recurrence_scores[state_id] == 0:
            continue

        state_blocks = df_blocks[df_blocks['state'] == state_id]
        n_blocks = len(state_blocks)
        median_dwell = float(state_blocks['duration_s'].median()) if n_blocks > 0 else 0.0
        mean_dwell = float(state_blocks['duration_s'].mean()) if n_blocks > 0 else 0.0
        total_occ = float(state_blocks['duration_s'].sum()) if n_blocks > 0 else 0.0

        # Model-learned self-transition (a_kk), not empirical
        self_trans = float(model_transmat[state_id, state_id])

        # Transition entropy: Shannon entropy of outgoing transition distribution
        row_probs = P[state_id, :]
        # Use only non-zero entries for entropy (base 2)
        nonzero = row_probs[row_probs > 0]
        trans_ent = float(shannon_entropy(nonzero, base=2)) if len(nonzero) > 0 else 0.0

        rows.append({
            'state_id': state_id,
            'recurrence_score': float(recurrence_scores[state_id]),
            'n_blocks': n_blocks,
            'median_dwell_s': median_dwell,
            'mean_dwell_s': mean_dwell,
            'total_occupancy_s': total_occ,
            'self_transition_prob': self_trans,
            'transition_entropy': trans_ent,
        })

    df_summary = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, 'state_summary_table.csv')
    df_summary.to_csv(out_path, index=False)
    logger.info(f"Saved state summary table ({len(df_summary)} active states) to {out_path}")
    return df_summary


def compute_dwell_time_descriptives(df_blocks, out_dir):
    """
    Compute overall descriptive statistics for dwell times (no hypothesis tests).
    """
    durations = df_blocks['duration_s']
    stats_results = {
        'all_active_states': {
            'count': int(len(durations)),
            'mean_s': float(durations.mean()) if len(durations) > 0 else 0,
            'median_s': float(durations.median()) if len(durations) > 0 else 0,
            'std_s': float(durations.std()) if len(durations) > 0 else 0,
        }
    }

    with open(os.path.join(out_dir, 'dwell_time_statistics.json'), 'w') as f:
        json.dump(stats_results, f, indent=2)

    logger.info(f"All active states: {stats_results['all_active_states']['count']} blocks, "
                f"Median Dwell: {stats_results['all_active_states']['median_s']:.2f}s")


def plot_transition_matrix(P, n_states, recurrence_scores, out_dir):
    """Plot transition probability matrix as a heatmap with recurrence annotation.
    States ordered by descending recurrence score."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    active_indices = [i for i in range(n_states) if recurrence_scores[i] > 0]
    active_indices.sort(key=lambda s: recurrence_scores[s], reverse=True)

    if len(active_indices) == 0:
        return

    cleaned_indices = np.array(active_indices)
    P_cleaned = P[cleaned_indices, :][:, cleaned_indices]
    rec_sorted = np.array([recurrence_scores[s] for s in active_indices])

    fig, ax = plt.subplots(figsize=(10, 8))

    mask = P_cleaned == 0
    sns.heatmap(P_cleaned, mask=mask, cmap='viridis', robust=True,
                xticklabels=False, yticklabels=False, ax=ax)

    ax.set_title('Transition Probability Matrix\n(Sorted by descending recurrence score)')
    ax.set_xlabel('To State')
    ax.set_ylabel('From State')

    # Add recurrence color strip on the left
    divider = make_axes_locatable(ax)
    ax_rec = divider.append_axes("left", size="3%", pad=0.05)
    rec_strip = rec_sorted[:, np.newaxis]
    ax_rec.imshow(rec_strip, cmap=RECURRENCE_CMAP, aspect='auto',
                  vmin=0, vmax=1, interpolation='nearest')
    ax_rec.set_xticks([])
    ax_rec.set_yticks([])
    ax_rec.set_ylabel('high recurrence →', fontsize=8)

    # Add recurrence color strip on top
    ax_rec_top = divider.append_axes("top", size="3%", pad=0.15)
    rec_strip_h = rec_sorted[np.newaxis, :]
    ax_rec_top.imshow(rec_strip_h, cmap=RECURRENCE_CMAP, aspect='auto',
                      vmin=0, vmax=1, interpolation='nearest')
    ax_rec_top.set_xticks([])
    ax_rec_top.set_yticks([])

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'transition_probabilities.{ext}'),
                    dpi=300 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)


def plot_barcodes(decoded_states, out_dir, recurrence_scores,
                  dominant_networks=None):
    """Plot timeline barcodes for one representative episode per season.
    Color encodes dominant network (or recurrence as fallback)."""
    import matplotlib.patches as mpatches
    import re

    run_ids = list(decoded_states.keys())
    if len(run_ids) == 0:
        return

    # Select one run per season for diversity
    season_runs = {}
    for rid in run_ids:
        m = re.search(r's(\d+)', rid)
        if m:
            season = int(m.group(1))
            if season not in season_runs:
                season_runs[season] = rid
    selected_runs = [season_runs[s] for s in sorted(season_runs.keys())]
    if not selected_runs:
        selected_runs = run_ids[:6]

    n_runs = len(selected_runs)
    fig, axes = plt.subplots(n_runs, 1, figsize=(12, 1.8 * n_runs + 0.8),
                             sharex=True)
    if n_runs == 1:
        axes = [axes]

    present_nets = set()
    for i, run_id in enumerate(selected_runs):
        seq = decoded_states[run_id]
        ax = axes[i]

        if dominant_networks:
            color_rgb = []
            for s in seq:
                net = dominant_networks.get(int(s), 'Unknown')
                present_nets.add(net)
                c = NETWORK_COLORS.get(net, '#888888')
                # Convert hex to RGB tuple
                import matplotlib.colors as mcolors
                color_rgb.append(mcolors.to_rgb(c))
        else:
            color_rgb = [recurrence_color(recurrence_scores[s])[:3] for s in seq]

        color_array = np.array(color_rgb)[np.newaxis, :, :]
        ax.imshow(color_array, aspect='auto', extent=[0, len(seq) * TR_SECONDS, 0, 1])
        ax.set_yticks([])
        ax.set_title(f'{run_id}', fontsize=10, loc='left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    axes[-1].set_xlabel('Time (seconds)')

    if dominant_networks and present_nets:
        fig.suptitle('Brain State Barcodes (color = dominant network)',
                     fontsize=13, y=0.99)
        handles = [mpatches.Patch(color=NETWORK_COLORS.get(net, '#888888'), label=net)
                   for net in NETWORK_ORDER if net in present_nets]
        fig.legend(handles=handles, loc='lower center', ncol=min(6, len(handles)),
                   fontsize=7, framealpha=0.8,
                   bbox_to_anchor=(0.5, -0.02))
    else:
        fig.suptitle('Brain State Barcodes (color = recurrence score)',
                     fontsize=13, y=0.99)
        # Use figure-level colorbar to avoid stealing subplot space
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        sm = ScalarMappable(cmap=RECURRENCE_CMAP, norm=Normalize(0, 1))
        sm.set_array([])
        fig.colorbar(sm, ax=axes, label='Recurrence Score', shrink=0.6,
                     location='bottom', pad=0.08)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'state_sequence_barcodes.{ext}'),
                    dpi=300 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)


def _network_colors_for_states(states, dominant_networks):
    """Get colors + legend handles for states based on dominant network."""
    import matplotlib.patches as mpatches
    colors = []
    present_nets = set()
    for s in states:
        net = dominant_networks.get(int(s), 'Unknown')
        colors.append(NETWORK_COLORS.get(net, '#888888'))
        present_nets.add(net)
    handles = [mpatches.Patch(color=NETWORK_COLORS.get(net, '#888888'), label=net)
               for net in NETWORK_ORDER if net in present_nets]
    return colors, handles


# CATEGORY_MARKERS imported from utils.state_flags_io (canonical source)


def _scatter_dual_encoded(ax, x, y, states, dominant_networks, state_categories):
    """Plot scatter with dual encoding: color=network, shape=category."""
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    present_nets = set()
    # Group by category to plot with different markers
    by_cat = {}
    for i, sid in enumerate(states):
        cat = state_categories.get(int(sid), 'unknown')
        if cat not in by_cat:
            by_cat[cat] = {'x': [], 'y': [], 'colors': [], 'sids': []}
        net = dominant_networks.get(int(sid), 'Unknown') if dominant_networks else 'Unknown'
        present_nets.add(net)
        by_cat[cat]['x'].append(x[i])
        by_cat[cat]['y'].append(y[i])
        by_cat[cat]['colors'].append(NETWORK_COLORS.get(net, '#888888'))
        by_cat[cat]['sids'].append(sid)

    for cat, data in by_cat.items():
        marker = CATEGORY_MARKERS.get(cat, 'o')
        ax.scatter(data['x'], data['y'], c=data['colors'], s=55,
                   marker=marker, alpha=0.8, edgecolors='black', linewidth=0.5)
        for j, sid in enumerate(data['sids']):
            ax.annotate(str(sid), (data['x'][j], data['y'][j]),
                        fontsize=6, alpha=0.5, ha='center', va='bottom')

    # Network color legend
    net_handles = [mpatches.Patch(color=NETWORK_COLORS.get(n, '#888888'), label=n)
                   for n in NETWORK_ORDER if n in present_nets]
    # Category shape legend
    cat_handles = []
    for cat in by_cat:
        m = CATEGORY_MARKERS.get(cat, 'o')
        label = cat.replace('_', ' ')
        cat_handles.append(Line2D([0], [0], marker=m, color='gray', markerfacecolor='gray',
                                   markersize=7, linestyle='None', label=label))

    return net_handles, cat_handles


def plot_recurrence_vs_temporal(df_summary, out_dir, dominant_networks=None,
                                state_categories=None):
    """2-panel scatter: (a) recurrence vs median dwell, (b) recurrence vs total occupancy.
    Dual encoding: color=dominant network, shape=05e_a4 category."""
    from scipy.stats import spearmanr, theilslopes

    if df_summary.empty:
        return

    if state_categories is None:
        state_categories = {}

    sids = df_summary['state_id'].values
    rec = df_summary['recurrence_score'].values
    dwell = df_summary['median_dwell_s'].values
    occ = df_summary['total_occupancy_s'].values

    eligible_mask = np.array([state_categories.get(int(s), 'eligible_for_content_analysis')
                              == 'eligible_for_content_analysis' for s in sids])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for panel_idx, (y_data, ylabel, title_suffix, fname_suffix) in enumerate([
        (dwell, 'Median Dwell Time (s)', 'Median Dwell', 'dwell'),
        (occ, 'Total Occupancy (s)', 'Total Occupancy', 'occupancy'),
    ]):
        ax = axes[panel_idx]
        net_handles, cat_handles = _scatter_dual_encoded(
            ax, rec, y_data, sids, dominant_networks, state_categories)

        ax.set_xlabel('Recurrence Score')
        ax.set_ylabel(ylabel)
        ax.set_title(f'({chr(97 + panel_idx)}) Recurrence vs {title_suffix}')

        # Theil-Sen + Spearman on eligible states only
        if eligible_mask.sum() >= 5:
            rec_e = rec[eligible_mask]
            y_e = y_data[eligible_mask]
            rho_e, p_e = spearmanr(rec_e, y_e)
            slope, intercept, _, _ = theilslopes(y_e, rec_e)
            x_line = np.linspace(rec_e.min(), rec_e.max(), 50)
            ax.plot(x_line, intercept + slope * x_line, '--', color='grey', lw=1.5)

            # All-states for comparison
            rho_all, p_all = spearmanr(rec, y_data) if len(rec) >= 5 else (np.nan, np.nan)

            ax.text(0.02, 0.98,
                    f'eligible: \u03c1={rho_e:.2f}, p={p_e:.2e}\n'
                    f'all: \u03c1={rho_all:.2f}, p={p_all:.2e}',
                    transform=ax.transAxes, fontsize=8, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    # Combined legends
    if net_handles or cat_handles:
        all_handles = (net_handles or []) + (cat_handles or [])
        fig.legend(handles=all_handles, loc='lower center',
                   ncol=min(6, len(all_handles)), fontsize=7, framealpha=0.8,
                   bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'recurrence_vs_temporal.{ext}'),
                    dpi=300 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)


def plot_dwell_distribution(df_blocks, model_transmat, out_dir,
                            state_categories=None):
    """Histogram of dwell durations with geometric distribution overlay."""
    if df_blocks.empty:
        return

    if state_categories is None:
        state_categories = {}

    durations_tr = df_blocks['duration_tr'].values

    # Separate by category
    eligible_mask = df_blocks['state'].map(
        lambda s: state_categories.get(int(s), 'eligible_for_content_analysis')
        == 'eligible_for_content_analysis').values
    dur_eligible = durations_tr[eligible_mask]
    dur_excluded = durations_tr[~eligible_mask]

    fig, ax = plt.subplots(figsize=(8, 5))

    max_dur = min(int(np.percentile(durations_tr, 99)), 30)
    bins = np.arange(1, max_dur + 2) - 0.5

    if len(dur_eligible) > 0:
        ax.hist(dur_eligible, bins=bins, alpha=0.7, color='steelblue',
                label=f'eligible (n={len(dur_eligible):,})', density=True)
    if len(dur_excluded) > 0:
        ax.hist(dur_excluded, bins=bins, alpha=0.5, color='salmon',
                label=f'excluded (n={len(dur_excluded):,})', density=True)

    # Geometric distribution overlay from mean model a_kk
    diag = np.diag(model_transmat)
    active_diag = diag[diag > 0]
    if len(active_diag) > 0:
        mean_akk = float(np.mean(active_diag))
        p_exit = 1.0 - mean_akk
        k_vals = np.arange(1, max_dur + 1)
        geom_pmf = (mean_akk ** (k_vals - 1)) * p_exit
        ax.plot(k_vals, geom_pmf, 'k--', lw=1.5, alpha=0.7,
                label=f'Geometric (mean a_kk={mean_akk:.2f})')

    # HRF threshold
    ax.axvline(3, color='red', lw=1, ls=':', alpha=0.7, label='HRF threshold (3 TRs)')

    ax.set_xlabel(f'Dwell Duration (TRs, 1 TR = {TR_SECONDS:.2f}s)')
    ax.set_ylabel('Density')
    ax.set_title('Dwell Time Distribution')
    ax.legend(fontsize=8, framealpha=0.8)

    # Annotate modal dwell
    if len(durations_tr) > 0:
        mode_tr = int(pd.Series(durations_tr).mode().iloc[0])
        ax.text(0.98, 0.98, f'mode={mode_tr} TRs ({mode_tr * TR_SECONDS:.1f}s)\n'
                f'median={np.median(durations_tr):.0f} TRs\n'
                f'mean={np.mean(durations_tr):.1f} TRs',
                transform=ax.transAxes, fontsize=8, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'dwell_time_distribution.{ext}'),
                    dpi=300 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)


def plot_network_dwell_comparison(df_summary, out_dir, dominant_networks=None,
                                  state_categories=None):
    """Boxplot of median dwell per state grouped by dominant network."""
    from scipy.stats import kruskal

    if df_summary.empty or not dominant_networks:
        return

    if state_categories is None:
        state_categories = {}

    # Add network and category columns
    df = df_summary.copy()
    df['network'] = df['state_id'].map(
        lambda s: dominant_networks.get(int(s), 'Unknown'))
    df['category'] = df['state_id'].map(
        lambda s: state_categories.get(int(s), 'unknown'))
    df['eligible'] = df['category'] == 'eligible_for_content_analysis'

    # Filter to networks with enough states
    df_elig = df[df['eligible']]
    net_counts = df_elig['network'].value_counts()
    nets_with_data = [n for n in NETWORK_ORDER if net_counts.get(n, 0) >= 2]

    if len(nets_with_data) < 2:
        logger.warning("Too few networks with eligible states for comparison")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    positions = []
    box_data = []
    colors = []
    labels = []
    for i, net in enumerate(nets_with_data):
        vals = df_elig.loc[df_elig['network'] == net, 'median_dwell_s'].values
        box_data.append(vals)
        positions.append(i)
        colors.append(NETWORK_COLORS.get(net, '#888888'))
        labels.append(f'{net}\n(n={len(vals)})')

    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=True, medianprops=dict(color='black', lw=1.5))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Median Dwell Time (s)')
    ax.set_title('Dwell Time by Dominant Network (eligible states only)')

    # Kruskal-Wallis test
    if len(box_data) >= 2 and all(len(d) >= 1 for d in box_data):
        valid_groups = [d for d in box_data if len(d) >= 1]
        if len(valid_groups) >= 2:
            try:
                H, p_kw = kruskal(*valid_groups)
                ax.text(0.98, 0.98, f'Kruskal-Wallis H={H:.1f}, p={p_kw:.3f}',
                        transform=ax.transAxes, fontsize=9, va='top', ha='right',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                                  alpha=0.8))
            except ValueError:
                pass

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'network_dwell_comparison.{ext}'),
                    dpi=300 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)


def compute_recurrence_assortativity(P, recurrence_scores, active_states,
                                     decoded_states,
                                     n_perm=5000, n_bootstrap=1000, seed=42):
    """Weighted directed recurrence assortativity with permutation + bootstrap.

    Tests whether high-recurrence states preferentially transition to each other.
    Builds an empirical graph and delegates to the shared implementation in
    transition_utils.compute_recurrence_assortativity.

    Returns dict with point_estimate, perm_p_value, bootstrap_ci.
    """
    import networkx as nx

    # Build empirical directed graph
    G = nx.DiGraph()
    for s in active_states:
        G.add_node(s, recurrence_score=float(recurrence_scores[s]))
    for i in active_states:
        for j in active_states:
            if i != j and P[i, j] > _EDGE_THRESH_ASSORTATIVITY:
                G.add_edge(i, j, weight=float(P[i, j]))

    return _shared_assortativity(
        G, decoded_states, active_states, recurrence_scores,
        edge_thresh=_EDGE_THRESH_ASSORTATIVITY,
        n_perm=n_perm, n_bootstrap=n_bootstrap, seed=seed,
        logger=logger,
    )


def cross_subject_summary(parcellation, vt):
    """Generate cross-subject summary figures from per-subject 06a outputs.

    Creates a 2x3 multi-panel scatter (recurrence vs dwell) for all 6 subjects.
    """
    from scipy.stats import spearmanr, theilslopes

    parc = normalize_parcellation_name(parcellation)
    subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06']

    out_dir = os.path.join(SCRATCH_DIR, 'output', '06a_state_temp_dynamics',
                           parc, 'cross_subject_summary')
    if vt is not None:
        out_dir = os.path.join(out_dir, f'vt{vt}')
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    summary_stats = {}

    for idx, sub_id in enumerate(subjects):
        ax = axes[idx]
        sub_dir = os.path.join(SCRATCH_DIR, 'output', '06a_state_temp_dynamics',
                               parc, sub_id)
        if vt is not None:
            sub_dir = os.path.join(sub_dir, f'vt{vt}')
        csv_path = os.path.join(sub_dir, 'state_summary_table.csv')

        if not os.path.exists(csv_path):
            ax.text(0.5, 0.5, f'{sub_id}\n(no data)', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 30)
            continue

        df = pd.read_csv(csv_path)
        rec = df['recurrence_score'].values
        dwell = df['median_dwell_s'].values

        ax.scatter(rec, dwell, c='#4A90D9', s=50, alpha=0.8,
                   edgecolors='black', linewidth=0.4)

        # Theil-Sen regression
        if len(rec) >= 5:
            slope, intercept, _, _ = theilslopes(dwell, rec)
            x_line = np.linspace(rec.min(), rec.max(), 50)
            ax.plot(x_line, intercept + slope * x_line, '--', color='grey', lw=1.2)

            rho, p = spearmanr(rec, dwell)
            weight = 'bold' if p < 0.05 else 'normal'
            ax.text(0.02, 0.98, f'ρ={rho:.2f}, p={p:.2e}',
                    transform=ax.transAxes, fontsize=9, va='top',
                    fontweight=weight,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                              alpha=0.8))
            summary_stats[sub_id] = {'rho': float(rho), 'p': float(p),
                                     'n_states': len(rec)}
        else:
            summary_stats[sub_id] = {'rho': np.nan, 'p': np.nan, 'n_states': len(rec)}

        ax.set_title(sub_id, fontsize=12, fontweight='bold')
        ax.set_xlim(-0.02, 1.0)
        ax.set_ylim(0, max(dwell.max() * 1.1, 15) if len(dwell) > 0 else 15)

        if idx >= 3:
            ax.set_xlabel('Recurrence Score')
        if idx % 3 == 0:
            ax.set_ylabel('Median Dwell (s)')

    fig.suptitle('Recurrence vs Median Dwell Time - All Subjects', fontsize=14, y=1.01)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'cross_subject_recurrence_vs_dwell.{ext}'),
                    dpi=300 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)

    # Save summary statistics
    with open(os.path.join(out_dir, 'cross_subject_summary_stats.json'), 'w') as f:
        json.dump(summary_stats, f, indent=2)

    logger.info(f"Cross-subject summary saved to {out_dir}")
    for sub, stats in summary_stats.items():
        logger.info(f"  {sub}: rho={stats['rho']:.3f}, p={stats['p']:.2e}, "
                    f"n={stats['n_states']}")


def main():
    parser = argparse.ArgumentParser(description="Analyze temporal dynamics of brain states.")
    parser.add_argument('--sub_id', type=str, default=None,
                        help="Subject ID (e.g., sub-01). Required for per_subject mode.")
    parser.add_argument('--mode', type=str, default='per_subject',
                        choices=['per_subject', 'cross_subject_summary'],
                        help="per_subject: standard analysis. "
                             "cross_subject_summary: aggregate multi-panel figure.")
    parser.add_argument('--parcellation', type=str, default='atlas-4S156Parcels')
    parser.add_argument('--vt', type=str, default=None,
                        help="Variance threshold subdirectory under final/ (e.g., 0.99). "
                             "Reads from final/vt{VT}/. If omitted, reads from final/ directly "
                             "(legacy path).")
    args = parser.parse_args()

    if args.mode == 'cross_subject_summary':
        cross_subject_summary(args.parcellation, args.vt)
        return

    if args.sub_id is None:
        parser.error("--sub_id is required for per_subject mode")

    parc = normalize_parcellation_name(args.parcellation)
    sub_id = args.sub_id

    logger.info("==============================================")
    logger.info("06a - Temporal Dynamics Analysis")
    logger.info("==============================================")
    logger.info(f"Subject: {sub_id}, Parcellation: {parc}")

    if args.vt is not None:
        hmm_base = os.path.join(SCRATCH_DIR, 'output', '04_combined_hdphmm', parc, sub_id,
                                'final', f'vt{args.vt}')
    else:
        hmm_base = os.path.join(SCRATCH_DIR, 'output', '04_combined_hdphmm', parc, sub_id, 'final')
    decoded_path = os.path.join(hmm_base, 'decoded_states.pkl')
    model_path = os.path.join(hmm_base, 'best_model.pkl')

    recur_base = os.path.join(SCRATCH_DIR, 'output', '05a_recurrence_analysis', parc, sub_id)
    if args.vt is not None:
        recur_base = os.path.join(recur_base, f'vt{args.vt}')
    summary_path = os.path.join(recur_base, 'recurrence_summary.json')

    out_dir = os.path.join(SCRATCH_DIR, 'output', '06a_state_temp_dynamics', parc, sub_id)
    if args.vt is not None:
        out_dir = os.path.join(out_dir, f'vt{args.vt}')
    os.makedirs(out_dir, exist_ok=True)

    for fpath, label in [(decoded_path, 'decoded states'),
                         (model_path, 'best model'),
                         (summary_path, 'recurrence summary')]:
        if not os.path.exists(fpath):
            logger.error(f"Missing {label}: {fpath}")
            sys.exit(1)

    logger.info("Loading inputs...")
    with open(decoded_path, 'rb') as f:
        decoded_states = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    model_transmat = model.transmat_.copy()
    del model  # Free memory
    with open(summary_path, 'r') as f:
        recurrence_summary = json.load(f)

    n_states = recurrence_summary['n_states']
    recurrence_scores = np.array(recurrence_summary['recurrence_scores'])

    # Load dominant network assignments for scatter plot coloring
    dominant_networks = {}
    state_means_path = os.path.join(hmm_base, 'state_means_parcel.npy')
    if os.path.exists(state_means_path):
        state_means = np.load(state_means_path)
        parcel_networks = load_parcel_networks(parc)
        if parcel_networks is not None:
            active_for_nets = [i for i in range(n_states) if recurrence_scores[i] > 0]
            dominant_networks = compute_dominant_networks(
                state_means, np.array(active_for_nets), parcel_networks)
            logger.info(f"Loaded dominant networks for {len(dominant_networks)} states")
    if not dominant_networks:
        logger.warning("Could not load network data - scatter plots will use fallback colors")

    # Load 05e_a4 state flags for category annotations
    parc_full = parc  # already normalized by normalize_parcellation_name()
    state_flags_df = load_state_flags(sub_id, parc_full, SCRATCH_DIR, args.vt)
    state_categories = {}
    if state_flags_df is not None:
        for _, row in state_flags_df.iterrows():
            state_categories[int(row['state'])] = row.get('summary_category', 'unknown')
        logger.info(f"Loaded state flags for {len(state_categories)} states")
    else:
        logger.warning("No 05e_a4 state flags - all states treated as eligible")

    logger.info("Extracting contiguous blocks...")
    df_blocks = extract_state_blocks(decoded_states, n_states, recurrence_scores)

    # Save block data
    df_blocks.to_csv(os.path.join(out_dir, 'state_blocks.csv'), index=False)

    logger.info("Computing dwell time descriptives...")
    compute_dwell_time_descriptives(df_blocks, out_dir)

    logger.info("Computing transition matrices...")
    P_counts, P = calculate_transition_matrix(decoded_states, n_states)
    np.save(os.path.join(out_dir, 'transition_counts.npy'), P_counts)
    np.save(os.path.join(out_dir, 'transition_probabilities.npy'), P)

    plot_transition_matrix(P, n_states, recurrence_scores, out_dir)

    logger.info("Plotting timeline barcodes (one per season)...")
    plot_barcodes(decoded_states, out_dir, recurrence_scores, dominant_networks)

    logger.info("Building state summary table...")
    df_summary = build_state_summary_table(
        df_blocks, P, model_transmat, recurrence_scores,
        n_states, out_dir)

    # Annotate summary table with network and category
    if dominant_networks:
        df_summary['dominant_network'] = df_summary['state_id'].map(
            lambda s: dominant_networks.get(int(s), 'Unknown'))
    if state_categories:
        df_summary['summary_category'] = df_summary['state_id'].map(
            lambda s: state_categories.get(int(s), 'unknown'))
    # Re-save with annotations
    df_summary.to_csv(os.path.join(out_dir, 'state_summary_table.csv'), index=False)

    logger.info("Creating recurrence vs temporal dynamics scatter...")
    plot_recurrence_vs_temporal(df_summary, out_dir, dominant_networks, state_categories)

    logger.info("Creating dwell time distribution...")
    plot_dwell_distribution(df_blocks, model_transmat, out_dir, state_categories)

    logger.info("Creating network dwell comparison...")
    plot_network_dwell_comparison(df_summary, out_dir, dominant_networks, state_categories)

    logger.info("Computing recurrence assortativity of transition graph...")
    active_states = [i for i in range(n_states) if recurrence_scores[i] > 0]
    assort_result = compute_recurrence_assortativity(
        P, recurrence_scores, active_states, decoded_states)
    with open(os.path.join(out_dir, 'recurrence_assortativity.json'), 'w') as f:
        json.dump(assort_result, f, indent=2)

    logger.info(f"Done! Outputs saved to {out_dir}")
    logger.info("==============================================")

if __name__ == '__main__':
    main()
