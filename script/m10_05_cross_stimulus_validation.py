#!/usr/bin/env python3
"""
m10_05_cross_stimulus_validation.py - Cross-stimulus validation of brain state recurrence scores.

Tests whether Friends-recurring brain states generalize to movie10 data by correlating
Friends recurrence scores with movie fractional occupancy.

Analyses:
    A1. Recurrence-FO Correlation (Spearman: Friends recurrence vs mean movie FO)
    A2. Per-Movie-Type Breakdown (Spearman per type, FDR-corrected)
    A3. Log-Likelihood Comparison (Friends test vs movie vs baseline)
    A4. State Coverage (heatmap: state × movie run FO)
    A5. PCA Transfer Diagnostic (Friends vs movie R²)

Prerequisites:
    - m10_04_score_and_decode.py completed (decoded states, FO, LL summary)
    - 05a_recurrence_analysis.py completed (recurrence scores)
    - m10_03_project_movie_pca.py completed (PCA diagnostic)

Outputs:
    {SCRATCH_DIR}/output/m10_05_cross_validation/{parcellation}/{sub_id}/
        cross_stimulus_summary.json   - All test results, effect sizes, p-values
        A1_recurrence_fo_scatter.png  - Scatter: recurrence score vs movie FO
        A2_per_type_scatter.png       - 2×2 scatter per movie type
        A3_ll_comparison.png          - Bar chart: LL comparison
        A4_state_coverage_heatmap.png - Heatmap: state × movie run FO
        A5_pca_diagnostic.png         - Bar chart: Friends vs movie R²

Documentation: the design notes
"""

import os
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Add script directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils.stats import benjamini_hochberg
from utils.plot_style import recurrence_color, make_recurrence_colorbar, apply_publication_style
from utils.common import normalize_parcellation_name
from utils.state_blocks import load_eligible_states
from utils.transformer_analysis import load_content_eligibility

from dotenv import load_dotenv

load_dotenv()
SCRATCH_DIR = os.getenv('SCRATCH_DIR')
if SCRATCH_DIR is None:
    raise ValueError("SCRATCH_DIR must be set in the .env file")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

apply_publication_style()

MOVIE_TYPE_LABELS = {
    'bourne': 'Bourne (Action)',
    'wolf': 'Wolf of WS (Drama)',
    'figures': 'Hidden Figures (Drama)',
    'life': 'Life (Documentary)',
}


def load_recurrence_scores_from_summary(recurrence_summary):
    """Extract recurrence_scores array from recurrence summary.

    Returns:
        np.ndarray: Per-state recurrence scores.
    """
    return np.array(recurrence_summary["recurrence_scores"], dtype=float)


# =========================================================================
# Content-eligibility helpers (project-wide 05e_a4 convention)
# =========================================================================

def _spearman_recurrence_vs_fo(recurrence_scores, mean_fo, state_subset):
    """Spearman ρ between recurrence and mean FO restricted to *state_subset*.

    *state_subset* is a sequence of state IDs to keep (e.g. content-eligible
    states intersected with active states). Returns a dict with spearman_rho,
    spearman_p, n, state_ids. When fewer than 3 states survive, rho/p are None.
    """
    ids = sorted(int(s) for s in state_subset)
    result = {
        'n': len(ids),
        'state_ids': ids,
    }
    if len(ids) >= 3:
        rec_vec = recurrence_scores[ids]
        fo_vec = mean_fo[ids]
        rho, p_val = stats.spearmanr(rec_vec, fo_vec)
        result['spearman_rho'] = float(rho)
        result['spearman_p'] = float(p_val)
    else:
        result['spearman_rho'] = None
        result['spearman_p'] = None
    return result


def compute_eligible_recurrence_correlations(
    stim_fo, stim_run_ids, recurrence_scores, n_states,
    eligibility,
):
    """Compute eligible-subset Spearman ρ for overall and per-type FO.

    Parameters
    ----------
    stim_fo : dict[run_id -> np.ndarray(n_states,)]
        Per-run fractional occupancy.
    stim_run_ids : dict[type -> list[run_id]]
        Per-type grouping of run IDs.
    recurrence_scores : np.ndarray
        Per-state Friends recurrence scores.
    n_states : int
    eligibility : dict
        Output of :func:`utils.transformer_analysis.load_content_eligibility`.

    Returns
    -------
    dict with keys:
        - ``eligibility_source`` (str)
        - ``n_content_eligible`` (int)
        - ``n_eligible_active`` (int) - eligible ∩ (recurrence > 0)
        - ``eligible_state_ids`` (list[int])
        - ``A1_overall`` - eligible-subset Spearman for mean FO across all runs
        - ``A2_per_type`` - dict[type -> eligible-subset Spearman] (+ FDR across types)
    """
    eligible_ids = set(int(s) for s in eligibility.get('content_eligible', []))
    active_mask = recurrence_scores > 0
    active_ids = set(int(i) for i in np.where(active_mask)[0])
    eligible_active = sorted(eligible_ids & active_ids)

    # Overall (A1-equivalent, mean FO across all runs)
    if stim_fo:
        overall_mean_fo = np.mean(np.array(list(stim_fo.values())), axis=0)
    else:
        overall_mean_fo = np.zeros(n_states)
    a1_overall = _spearman_recurrence_vs_fo(
        recurrence_scores, overall_mean_fo, eligible_active,
    )

    # Per-type (A2-equivalent)
    per_type = {}
    raw_pvals = []
    pval_keys = []
    for stype, run_ids in stim_run_ids.items():
        type_fo_arrays = [stim_fo[rid] for rid in run_ids if rid in stim_fo]
        if not type_fo_arrays:
            per_type[stype] = {
                'spearman_rho': None, 'spearman_p': None,
                'n': 0, 'state_ids': [], 'n_runs': 0,
            }
            continue
        type_mean_fo = np.mean(np.array(type_fo_arrays), axis=0)
        tr = _spearman_recurrence_vs_fo(recurrence_scores, type_mean_fo, eligible_active)
        tr['n_runs'] = len(type_fo_arrays)
        per_type[stype] = tr
        if tr['spearman_p'] is not None:
            raw_pvals.append(tr['spearman_p'])
            pval_keys.append(stype)

    if len(raw_pvals) >= 2:
        fdr_q = benjamini_hochberg(np.array(raw_pvals))
        for stype, q in zip(pval_keys, fdr_q):
            per_type[stype]['fdr_q'] = float(q)

    return {
        'eligibility_source': eligibility.get('eligibility_source', 'unknown'),
        'n_content_eligible': len(eligible_ids),
        'n_eligible_active': len(eligible_active),
        'eligible_state_ids': eligible_active,
        'A1_overall': a1_overall,
        'A2_per_type': per_type,
    }


# =========================================================================
# Analysis functions (A1-A5)
# =========================================================================

def analysis_a1_recurrence_fo_correlation(movie_fo, recurrence_scores, n_states, out_dir):
    """A1: Recurrence-FO Correlation.

    Spearman rank correlation between Friends recurrence score and mean movie FO.
    All active states (recurrence > 0) are included.
    """
    logger.info("A1: Recurrence-FO Correlation")

    # Compute mean movie FO per state
    fo_matrix = np.array(list(movie_fo.values()))  # (n_runs, n_states)
    mean_fo = np.mean(fo_matrix, axis=0)

    # Active states: recurrence > 0
    active_mask = np.array([recurrence_scores[i] > 0 for i in range(n_states)])
    active_indices = np.where(active_mask)[0]

    rec_active = recurrence_scores[active_mask]
    fo_active = mean_fo[active_mask]

    result = {'n_active_states': int(len(active_indices))}

    if len(rec_active) >= 3:
        rho, p_val = stats.spearmanr(rec_active, fo_active)
        result['spearman_rho'] = float(rho)
        result['spearman_p'] = float(p_val)
        result['spearman_p_note'] = (
            'Approximate: movie FO is compositional (sums to 1 per run), '
            'inducing weak negative dependence among state-level values. '
            'Effect is attenuated by cross-dataset design (recurrence from '
            'Friends, FO from Movie10).')
        result['positive_correlation'] = bool(rho > 0 and p_val < 0.05)
    else:
        result['spearman_rho'] = None
        result['spearman_p'] = None
        result['positive_correlation'] = None
        logger.warning("A1: Too few active states for correlation")

    # --- Plot: scatter colored by recurrence score ---
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = [recurrence_color(recurrence_scores[i]) for i in active_indices]
    ax.scatter(rec_active, fo_active,
               c=colors, alpha=0.7, s=30, edgecolors='white', linewidths=0.5)

    ax.set_xlabel('Friends Recurrence Score')
    ax.set_ylabel('Mean Movie FO')
    ax.set_title('A1: Friends Recurrence vs Movie FO')
    make_recurrence_colorbar(ax)

    if result['spearman_rho'] is not None:
        rho_str = f"ρ = {result['spearman_rho']:.3f}"
        p_str = f"p = {result['spearman_p']:.2e}" if result['spearman_p'] < 0.001 \
                else f"p = {result['spearman_p']:.3f}"
        ax.annotate(f"Spearman {rho_str}, {p_str}",
                    xy=(0.02, 0.98), xycoords='axes fraction',
                    ha='left', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'A1_recurrence_fo_scatter.png'), dpi=150)
    plt.close(fig)

    return result


def analysis_a2_per_type(movie_fo, movie_run_ids, recurrence_scores, n_states, out_dir):
    """A2: Per-Movie-Type Breakdown.

    Spearman correlation per movie type, FDR-corrected.
    All active states (recurrence > 0) included.
    """
    logger.info("A2: Per-Movie-Type Breakdown")

    active_mask = np.array([recurrence_scores[i] > 0 for i in range(n_states)])
    rec_active = recurrence_scores[active_mask]
    active_indices = np.where(active_mask)[0]

    type_results = {}
    p_values = []

    for mtype, run_ids in movie_run_ids.items():
        # Compute mean FO for this movie type
        type_fo_arrays = [movie_fo[rid] for rid in run_ids if rid in movie_fo]
        if not type_fo_arrays:
            type_results[mtype] = {'spearman_rho': None, 'spearman_p': None, 'n_runs': 0}
            continue

        type_mean_fo = np.mean(np.array(type_fo_arrays), axis=0)
        fo_active = type_mean_fo[active_mask]

        if len(rec_active) >= 3:
            rho, p_val = stats.spearmanr(rec_active, fo_active)
            type_results[mtype] = {
                'spearman_rho': float(rho),
                'spearman_p': float(p_val),
                'spearman_p_note': 'Approximate due to FO compositionality; see A1 note.',
                'n_runs': len(type_fo_arrays),
            }
            p_values.append(p_val)
        else:
            type_results[mtype] = {'spearman_rho': None, 'spearman_p': None, 'n_runs': len(type_fo_arrays)}

    # FDR correction across types
    if p_values:
        fdr_q = benjamini_hochberg(np.array(p_values))
        p_idx = 0
        for mtype in movie_run_ids:
            if type_results[mtype]['spearman_p'] is not None:
                type_results[mtype]['fdr_q'] = float(fdr_q[p_idx])
                p_idx += 1

    # --- Plot: 2x2 scatter colored by recurrence score ---
    movie_types = list(movie_run_ids.keys())
    n_types = len(movie_types)
    ncols = min(2, n_types)
    nrows = (n_types + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    for i, mtype in enumerate(movie_types):
        row, col = divmod(i, ncols)
        ax = axes[row][col]

        run_ids = movie_run_ids[mtype]
        type_fo_arrays = [movie_fo[rid] for rid in run_ids if rid in movie_fo]
        if type_fo_arrays:
            type_mean_fo = np.mean(np.array(type_fo_arrays), axis=0)
        else:
            type_mean_fo = np.zeros(n_states)

        colors = [recurrence_color(recurrence_scores[idx]) for idx in active_indices]
        ax.scatter(rec_active, type_mean_fo[active_mask],
                   c=colors, alpha=0.7, s=20, edgecolors='white', linewidths=0.3)

        ax.set_xlabel('Friends Recurrence Score')
        ax.set_ylabel('Mean Movie FO')
        label = MOVIE_TYPE_LABELS.get(mtype, mtype)
        ax.set_title(label)

        tr = type_results[mtype]
        if tr['spearman_rho'] is not None:
            q_str = f"q={tr.get('fdr_q', tr['spearman_p']):.3f}"
            ax.annotate(f"ρ={tr['spearman_rho']:.3f}, {q_str}",
                        xy=(0.02, 0.98), xycoords='axes fraction',
                        ha='left', va='top', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        if i == 0:
            make_recurrence_colorbar(ax)

    # Hide unused subplots
    for i in range(n_types, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle('A2: Per-Movie-Type Recurrence-FO Correlation', fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'A2_per_type_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    return type_results


def analysis_a3_ll_comparison(ll_summary, out_dir):
    """A3: Log-Likelihood Comparison."""
    logger.info("A3: Log-Likelihood Comparison")

    friends_ll = ll_summary['friends_test_ll_per_sample']
    movie_ll = ll_summary['movie_overall_ll_per_sample']
    baseline_ll = ll_summary['baseline_ll_per_sample']

    result = {
        'friends_test_ll': friends_ll,
        'movie_overall_ll': movie_ll,
        'baseline_ll': baseline_ll,
        'baseline_note': (
            'Heuristic reference point only: log(1/n_active_states) is a '
            'uniform state-assignment baseline, not on the same scale as '
            'Gaussian-emission HMM log-likelihood. A single-state Gaussian '
            'baseline would be a principled null.'),
        'll_gap': ll_summary['ll_gap_friends_minus_movie'],
        'movie_above_baseline': ll_summary['movie_above_baseline'],
    }

    # Per-type LL
    per_type = {}
    for mtype, info in ll_summary.get('per_type', {}).items():
        per_type[mtype] = info.get('ll_per_sample')
    result['per_type_ll'] = per_type

    # --- Plot: bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ['Friends\nTest']
    values = [friends_ll]
    colors = ['#4477AA']

    # Per-type bars
    type_colors = {'bourne': '#CC6677', 'wolf': '#DDCC77', 'figures': '#88CCEE', 'life': '#44AA99'}
    for mtype in ['bourne', 'wolf', 'figures', 'life']:
        if mtype in per_type and per_type[mtype] is not None:
            labels.append(MOVIE_TYPE_LABELS.get(mtype, mtype).split('(')[0].strip())
            values.append(per_type[mtype])
            colors.append(type_colors.get(mtype, '#AABBCC'))

    labels.append('Movie\nOverall')
    values.append(movie_ll)
    colors.append('#882255')

    labels.append('Baseline\n(uniform)')
    values.append(baseline_ll)
    colors.append('#999999')

    bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.8, edgecolor='white')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('LL / sample')
    ax.set_title('A3: Log-Likelihood Comparison')
    ax.axhline(y=baseline_ll, color='#999999', linestyle='--', alpha=0.5, label='Baseline')

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'A3_ll_comparison.png'), dpi=150)
    plt.close(fig)

    return result


def analysis_a4_state_coverage(movie_fo, movie_run_ids, recurrence_scores, n_states,
                                fo_threshold, out_dir):
    """A4: State Coverage Analysis.

    Heatmap of state x movie run FO, ordered by recurrence score.
    Spearman: Friends recurrence vs movie coverage (continuous metric).
    """
    logger.info("A4: State Coverage Analysis")

    # Active states: recurrence > 0
    active_states = [i for i in range(n_states) if recurrence_scores[i] > 0]
    active_states_sorted = sorted(active_states, key=lambda s: recurrence_scores[s], reverse=True)

    # Build FO matrix: state x run
    all_run_ids = []
    run_types = []
    for mtype in ['bourne', 'wolf', 'figures', 'life']:
        for rid in movie_run_ids.get(mtype, []):
            if rid in movie_fo:
                all_run_ids.append(rid)
                run_types.append(mtype)

    fo_matrix = np.zeros((len(active_states_sorted), len(all_run_ids)))
    for j, rid in enumerate(all_run_ids):
        for i, state_id in enumerate(active_states_sorted):
            fo_matrix[i, j] = movie_fo[rid][state_id]

    # State coverage: fraction of movie runs where each state is active
    coverage = {}
    for state_id in active_states_sorted:
        n_active_runs = sum(1 for rid in all_run_ids if movie_fo[rid][state_id] > fo_threshold)
        frac = n_active_runs / len(all_run_ids) if all_run_ids else 0
        coverage[int(state_id)] = {
            'movie_coverage': float(frac),
            'friends_recurrence': float(recurrence_scores[state_id]),
        }

    # Spearman: Friends recurrence vs movie coverage (continuous replacement for A4 categorical count)
    rec_vals = np.array([recurrence_scores[s] for s in active_states_sorted])
    cov_vals = np.array([coverage[s]['movie_coverage'] for s in active_states_sorted])
    coverage_corr = {}
    if len(rec_vals) >= 5:
        rho, p = stats.spearmanr(rec_vals, cov_vals)
        coverage_corr = {'rho': float(rho), 'p': float(p), 'n': len(rec_vals)}

    # Check for Friends-inactive states that become active in movies
    inactive_states = [i for i in range(n_states) if recurrence_scores[i] == 0]
    inactive_activated = {}
    for state_id in inactive_states:
        movie_mean_fo = float(np.mean([movie_fo[rid][state_id] for rid in all_run_ids]))
        if movie_mean_fo > fo_threshold:
            inactive_activated[int(state_id)] = float(movie_mean_fo)
            logger.warning(f"A4: Friends-inactive state {state_id} has mean movie FO={movie_mean_fo:.4f}")

    result = {
        'state_coverage': coverage,
        'recurrence_vs_coverage_spearman': coverage_corr,
        'n_active_states_plotted': len(active_states_sorted),
        'inactive_states_activated_in_movies': inactive_activated,
    }

    # --- Plot: heatmap ---
    if len(active_states_sorted) > 0 and len(all_run_ids) > 0:
        fig_height = max(4, len(active_states_sorted) * 0.15 + 2)
        fig_width = max(8, len(all_run_ids) * 0.12 + 3)
        fig, ax = plt.subplots(figsize=(min(fig_width, 20), min(fig_height, 30)))

        im = ax.imshow(fo_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        plt.colorbar(im, ax=ax, label='Fractional Occupancy', shrink=0.8)

        # Y-axis: state labels colored by recurrence score
        y_labels = []
        y_colors = []
        for state_id in active_states_sorted:
            y_labels.append(f"S{state_id}")
            y_colors.append(recurrence_color(recurrence_scores[state_id]))

        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=5)
        for tick, color in zip(ax.get_yticklabels(), y_colors):
            tick.set_color(color)

        # X-axis: movie type boundaries
        type_boundaries = []
        current_type = run_types[0] if run_types else ''
        for j, rt in enumerate(run_types):
            if rt != current_type:
                type_boundaries.append(j)
                current_type = rt
        for b in type_boundaries:
            ax.axvline(x=b - 0.5, color='white', linewidth=1.5)

        # X-axis labels: movie types at midpoints
        prev_b = 0
        for b in type_boundaries + [len(run_types)]:
            mid = (prev_b + b) / 2
            mtype = run_types[prev_b] if prev_b < len(run_types) else ''
            ax.text(mid, -0.5, MOVIE_TYPE_LABELS.get(mtype, mtype).split('(')[0].strip(),
                    ha='center', va='bottom', fontsize=8, transform=ax.get_xaxis_transform())
            prev_b = b

        ax.set_xticks([])
        ax.set_xlabel('Movie Runs (grouped by type)')
        ax.set_ylabel('States (sorted by Friends recurrence, descending)')
        ax.set_title('A4: State Coverage Across Movie Runs')

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'A4_state_coverage_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    return result


def analysis_a5_pca_diagnostic(pca_diagnostic, out_dir):
    """A5: PCA Transfer Diagnostic."""
    logger.info("A5: PCA Transfer Diagnostic")

    result = {
        'friends_r2': pca_diagnostic['friends_r2_n_pcs'],
        'movie_r2': pca_diagnostic['movie_r2_n_pcs'],
        'transfer_gap': pca_diagnostic['transfer_gap'],
        'flag_low_variance': pca_diagnostic['flag_low_variance'],
        'n_pcs': pca_diagnostic['n_pcs'],
    }

    # --- Plot: bar chart ---
    fig, ax = plt.subplots(figsize=(5, 4))

    labels = ['Friends\n(training)', 'Movie10']
    values = [pca_diagnostic['friends_r2_n_pcs'], pca_diagnostic['movie_r2_n_pcs']]
    colors = ['#4477AA', '#882255']

    bars = ax.bar(labels, values, color=colors, alpha=0.8, width=0.5)
    ax.set_ylabel(f'Variance Explained (R², {pca_diagnostic["n_pcs"]} PCs)')
    ax.set_title('A5: PCA Transfer Diagnostic')
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.70, color='red', linestyle='--', alpha=0.5, label='70% threshold')
    ax.legend(fontsize=8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'A5_pca_diagnostic.png'), dpi=150)
    plt.close(fig)

    return result


# =========================================================================
# Main
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Cross-stimulus validation: test Friends brain state generalization to movies.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script/m10_05_cross_stimulus_validation.py --sub_id sub-01
  python script/m10_05_cross_stimulus_validation.py --sub_id sub-01 --parcellation atlas-4S456Parcels
        """
    )
    parser.add_argument('--sub_id', type=str, required=True,
                        help='Subject ID (e.g., "sub-01")')
    parser.add_argument('--parcellation', type=str, default='atlas-4S156Parcels',
                        help='Parcellation name (default: atlas-4S156Parcels)')
    parser.add_argument('--fo_threshold', type=float, default=0.01,
                        help='FO threshold for "active" state (default: 0.01)')
    parser.add_argument('--vt', type=str, default=None,
                        help="Variance threshold subdirectory (e.g., 0.95). "
                             "Reads from vt{VT}/ subdirs. If omitted, reads flat (legacy).")
    parser.add_argument('--exclude_sub_hrf', action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Exclude sub-HRF states from cross-stimulus analyses "
                             "(default: False - include all states, since FO-based "
                             "validation does not require per-block BOLD evidence). "
                             "Use --exclude_sub_hrf for sensitivity analysis.")
    return parser.parse_args()


def main():
    args = parse_args()
    sub_id = args.sub_id
    parc = normalize_parcellation_name(args.parcellation)
    fo_threshold = args.fo_threshold

    # =========================================================================
    # Input paths
    # =========================================================================

    # Friends state categories (vt-aware)
    recurrence_dir = os.path.join(SCRATCH_DIR, 'output', '05a_recurrence_analysis', parc, sub_id)
    if args.vt is not None:
        recurrence_dir = os.path.join(recurrence_dir, f'vt{args.vt}')
    recurrence_summary_path = os.path.join(recurrence_dir, 'recurrence_summary.json')
    recurrence_scores_path = os.path.join(recurrence_dir, 'recurrence_scores.npy')

    # Movie decoded states + FO (vt-aware)
    movie_dir = os.path.join(SCRATCH_DIR, 'output', 'm10_04_decoded', parc, sub_id)
    if args.vt is not None:
        movie_dir = os.path.join(movie_dir, f'vt{args.vt}')
    movie_fo_path = os.path.join(movie_dir, 'fractional_occupancy.pkl')
    movie_ll_path = os.path.join(movie_dir, 'movie_ll_summary.json')

    # Movie run IDs (vt-aware)
    proj_dir = os.path.join(SCRATCH_DIR, 'output', 'm10_03_projected', parc, sub_id)
    if args.vt is not None:
        proj_dir = os.path.join(proj_dir, f'vt{args.vt}')
    movie_run_ids_path = os.path.join(proj_dir, 'movie_run_ids.json')

    # run_id_map.json from m10_04 - maps long BIDS keys to short keys
    run_id_map_path = os.path.join(movie_dir, 'run_id_map.json')

    # PCA diagnostic
    pca_diag_path = os.path.join(proj_dir, 'pca_transfer_diagnostic.json')

    # Validate inputs
    required_files = {
        recurrence_summary_path: 'recurrence_summary.json (run 05a first)',
        recurrence_scores_path: 'recurrence_scores.npy (run 05a first)',
        movie_fo_path: 'movie FO (run m10_04 first)',
        movie_ll_path: 'movie LL summary (run m10_04 first)',
        movie_run_ids_path: 'movie_run_ids.json (run m10_03 first)',
        run_id_map_path: 'run_id_map.json (run m10_04 first)',
        pca_diag_path: 'pca_transfer_diagnostic.json (run m10_03 first)',
    }
    for path, label in required_files.items():
        if not os.path.exists(path):
            logger.error(f"Missing: {path} - {label}")
            sys.exit(1)

    # =========================================================================
    # Load data
    # =========================================================================

    with open(recurrence_summary_path, 'r') as f:
        recurrence_summary = json.load(f)

    recurrence_scores = np.load(recurrence_scores_path)
    n_states = len(recurrence_scores)

    with open(movie_fo_path, 'rb') as f:
        movie_fo = pickle.load(f)

    with open(movie_ll_path, 'r') as f:
        ll_summary = json.load(f)

    with open(movie_run_ids_path, 'r') as f:
        movie_run_ids_raw = json.load(f)

    with open(run_id_map_path, 'r') as f:
        run_id_map = json.load(f)

    # movie_run_ids.json uses long BIDS keys (from m10_03) but movie_fo.pkl
    # uses short keys (from m10_04's normalize_cross_stim_run_id). Remap so
    # the type->run_id lists match the FO dict keys.
    long_to_short = run_id_map['long_to_short']
    movie_run_ids = {}
    for mtype, long_ids in movie_run_ids_raw.items():
        short_ids = []
        for lid in long_ids:
            short = long_to_short.get(lid)
            if short is None:
                logger.warning("run_id_map has no short key for %s - skipping", lid)
                continue
            short_ids.append(short)
        movie_run_ids[mtype] = short_ids

    with open(pca_diag_path, 'r') as f:
        pca_diagnostic = json.load(f)

    # Sub-HRF filtering (default OFF for this script - FO-based validation
    # does not require per-block BOLD evidence)
    excluded_sub_hrf = []
    if args.exclude_sub_hrf:
        try:
            _, excluded_ids, _ = load_eligible_states(recurrence_dir)
            excluded_set = set(int(s) for s in excluded_ids)
            # Zero out recurrence for excluded states so they are treated as inactive
            for sid in excluded_set:
                recurrence_scores[sid] = 0.0
            excluded_sub_hrf = sorted(excluded_set)
            logger.info(
                "Sub-HRF exclusion ON: zeroed recurrence for %d states",
                len(excluded_set),
            )
        except FileNotFoundError:
            logger.warning(
                "eligible_states.json not found in %s; sub-HRF filtering skipped. "
                "Re-run 05a to generate it.",
                recurrence_dir,
            )

    n_active = int(np.sum(recurrence_scores > 0))
    logger.info(f"Loaded data for {sub_id}: {n_states} states, {len(movie_fo)} movie runs")
    logger.info(f"Active states (recurrence > 0): {n_active}")

    # =========================================================================
    # Output directory
    # =========================================================================

    out_dir = os.path.join(SCRATCH_DIR, 'output', 'm10_05_cross_validation', parc, sub_id)
    if args.vt is not None:
        out_dir = os.path.join(out_dir, f'vt{args.vt}')
    if args.exclude_sub_hrf:
        out_dir = os.path.join(out_dir, 'exclude_sub_hrf')
    os.makedirs(out_dir, exist_ok=True)

    # =========================================================================
    # Run analyses
    # =========================================================================

    summary = {
        'subject': sub_id,
        'parcellation': parc,
        'n_states': n_states,
        'n_movie_runs': len(movie_fo),
        'fo_threshold': fo_threshold,
        'exclude_sub_hrf': args.exclude_sub_hrf,
        'n_excluded_sub_hrf': len(excluded_sub_hrf),
        'excluded_sub_hrf_states': excluded_sub_hrf,
        'n_active_states': n_active,
    }

    summary['A1_recurrence_correlation'] = analysis_a1_recurrence_fo_correlation(
        movie_fo, recurrence_scores, n_states, out_dir)
    summary['A2_per_type'] = analysis_a2_per_type(
        movie_fo, movie_run_ids, recurrence_scores, n_states, out_dir)

    # Content-eligible-subset variants (project-wide 05e_a4 convention).
    # These mirror A1/A2 but restrict both vectors to the subset of states
    # flagged as ``eligible_for_content_analysis`` in 05e_a4's state_flags.csv
    # (intersected with active states). The full-repertoire A1/A2 above are
    # preserved unchanged so downstream figures can show both panels.
    eligibility = load_content_eligibility(
        sub_id=sub_id, parcellation=parc, scratch_dir=SCRATCH_DIR, vt=args.vt,
    )
    eligible_variants = compute_eligible_recurrence_correlations(
        movie_fo, movie_run_ids, recurrence_scores, n_states, eligibility,
    )
    summary['A1_recurrence_correlation_eligible'] = eligible_variants['A1_overall']
    summary['A1_recurrence_correlation_eligible'].update({
        'eligibility_source': eligible_variants['eligibility_source'],
        'n_content_eligible': eligible_variants['n_content_eligible'],
        'n_eligible_active': eligible_variants['n_eligible_active'],
        'eligible_state_ids': eligible_variants['eligible_state_ids'],
    })
    summary['A2_per_type_eligible'] = eligible_variants['A2_per_type']
    summary['eligibility_source'] = eligible_variants['eligibility_source']
    summary['n_content_eligible'] = eligible_variants['n_content_eligible']
    summary['n_eligible_active'] = eligible_variants['n_eligible_active']
    summary['eligible_state_ids'] = eligible_variants['eligible_state_ids']
    logger.info(
        "Eligible-subset A1: n_eligible_active=%d, rho=%s, p=%s (source=%s)",
        eligible_variants['n_eligible_active'],
        eligible_variants['A1_overall'].get('spearman_rho'),
        eligible_variants['A1_overall'].get('spearman_p'),
        eligible_variants['eligibility_source'],
    )

    summary['A3_ll_comparison'] = analysis_a3_ll_comparison(ll_summary, out_dir)
    summary['A4_state_coverage'] = analysis_a4_state_coverage(
        movie_fo, movie_run_ids, recurrence_scores, n_states, fo_threshold, out_dir)
    summary['A5_pca_diagnostic'] = analysis_a5_pca_diagnostic(pca_diagnostic, out_dir)

    # =========================================================================
    # Save summary
    # =========================================================================

    summary_path = os.path.join(out_dir, 'cross_stimulus_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # =========================================================================
    # Report
    # =========================================================================

    a1 = summary['A1_recurrence_correlation']
    a3 = summary['A3_ll_comparison']
    a4 = summary['A4_state_coverage']
    a5 = summary['A5_pca_diagnostic']

    print(f"\n{'='*60}")
    print(f"CROSS-STIMULUS VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Subject: {sub_id} | Parcellation: {parc}")
    print(f"Movie runs: {len(movie_fo)} | States: {n_states}")
    print(f"")
    print(f"A1 - Recurrence-FO Correlation:")
    if a1.get('spearman_rho') is not None:
        print(f"  Full repertoire:   ρ={a1['spearman_rho']:.3f}, p={a1['spearman_p']:.4f}")
    a1e = summary.get('A1_recurrence_correlation_eligible', {})
    if a1e.get('spearman_rho') is not None:
        print(f"  Content-eligible:  ρ={a1e['spearman_rho']:.3f}, p={a1e['spearman_p']:.4f} "
              f"(n={a1e.get('n_eligible_active', 0)}, source={a1e.get('eligibility_source')})")
    elif a1e:
        print(f"  Content-eligible:  n_eligible_active={a1e.get('n_eligible_active', 0)} "
              f"(too few for Spearman; source={a1e.get('eligibility_source')})")
    print(f"")
    print(f"A3 - LL Comparison:")
    print(f"  Friends test:  {a3['friends_test_ll']:.4f}")
    print(f"  Movie overall: {a3['movie_overall_ll']:.4f}")
    print(f"  Baseline:      {a3['baseline_ll']:.4f}")
    print(f"  Movie > base:  {a3['movie_above_baseline']}")
    print(f"")
    print(f"A4 - State Coverage:")
    cov_corr = a4.get('recurrence_vs_coverage_spearman', {})
    if cov_corr:
        print(f"  Recurrence-coverage Spearman rho: {cov_corr['rho']:.3f}, p={cov_corr['p']:.4f}")
    if a4.get('inactive_states_activated_in_movies'):
        print(f"  *** {len(a4['inactive_states_activated_in_movies'])} Friends-inactive states activated in movies ***")
    print(f"")
    print(f"A5 - PCA Diagnostic:")
    print(f"  Friends R²: {a5['friends_r2']:.4f}, Movie R²: {a5['movie_r2']:.4f}")
    if a5['flag_low_variance']:
        print(f"  *** WARNING: Movie R² < 0.70 ***")
    print(f"{'='*60}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
