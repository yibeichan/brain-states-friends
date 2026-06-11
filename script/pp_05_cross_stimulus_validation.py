#!/usr/bin/env python3
"""
pp_05_cross_stimulus_validation.py - Cross-stimulus validation: Petit Prince audio narration.

Tests whether Friends-recurring brain states generalize to unimodal audio narration
(Petit Prince, presented as continuous audio in French and English) by comparing PP
fractional occupancy against Friends recurrence scores.

Petit Prince uses audio-only narration (no visual input),
unlike Friends and Movie10 which are audiovisual. This removes all visual input,
providing only auditory linguistic input. This tests whether recurring
states require rich audiovisual input or reflect supramodal narrative/social processing.

Analyses:
    A1. Recurrence-FO Correlation (Spearman: Friends recurrence vs mean PP FO)
    A2. Per-Type Breakdown (2 types: lppFR, lppEN; FDR across 2 tests)
    A3. Log-Likelihood Comparison (Friends test vs PP vs baseline)
    A4. State Coverage (heatmap: state x PP run FO)
    A5. PCA Transfer Diagnostic (Friends vs PP R², with network-stratified breakdown)

Additional PP-specific analyses:
    B1. 5-Subject Movie10 Baseline (recompute Movie10 A1 for direct comparison)
    B2. Bootstrap Matched-Sample-Size Reference (18-run Movie10 subsamples)
    Language Comparison: Cosine similarity between French and English mean FO vectors

Prerequisites:
    - pp_04_score_and_decode.py completed (decoded states, FO, LL summary)
    - 05a_recurrence_analysis.py completed (state categories)
    - pp_03_project_pp_pca.py completed (PCA diagnostic)
    - m10_04_score_and_decode.py completed (for B1/B2 Movie10 comparison)

Outputs:
    {SCRATCH_DIR}/output/pp_05_cross_validation/{parcellation}/{sub_id}/
        cross_stimulus_summary.json   - All test results, effect sizes, p-values
        A1_recurrence_fo_scatter.png  - Scatter: recurrence score vs PP FO
        A2_per_type_scatter.png       - 1x2 scatter (lppFR, lppEN)
        A3_ll_comparison.png          - Bar chart: LL comparison
        A4_state_coverage_heatmap.png - Heatmap: state x PP run FO
        A5_pca_diagnostic.png         - Bar chart: Friends vs PP R²
        B2_bootstrap_reference.png    - PP effect sizes vs Movie10 bootstrap
        language_comparison.png       - French vs English FO comparison
        serial_dependence.png         - Per-language serial dependence

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
from scipy.spatial.distance import cosine as cosine_distance

# Add script directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils.stats import benjamini_hochberg
from utils.plot_style import (recurrence_color, make_recurrence_colorbar,
                               apply_publication_style,
                               NETWORK_ORDER, NETWORK_COLORS)
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

PP_TYPE_LABELS = {
    'lppFR': 'French (Audio)',
    'lppEN': 'English (Audio)',
}


def load_recurrence_scores_from_summary(recurrence_summary):
    """Extract recurrence_scores array from recurrence summary."""
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
    result = {'n': len(ids), 'state_ids': ids}
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
    """Eligible-subset Spearman ρ for overall and per-type FO.

    Mirrors analyses A1/A2 but restricts both vectors to content-eligible
    states (05e_a4 state_flags.csv) intersected with active states.
    Returns a dict with keys ``A1_overall`` and ``A2_per_type`` plus
    metadata (``eligibility_source``, ``n_content_eligible``, …).
    """
    eligible_ids = set(int(s) for s in eligibility.get('content_eligible', []))
    active_mask = recurrence_scores > 0
    active_ids = set(int(i) for i in np.where(active_mask)[0])
    eligible_active = sorted(eligible_ids & active_ids)

    if stim_fo:
        overall_mean_fo = np.mean(np.array(list(stim_fo.values())), axis=0)
    else:
        overall_mean_fo = np.zeros(n_states)
    a1_overall = _spearman_recurrence_vs_fo(
        recurrence_scores, overall_mean_fo, eligible_active,
    )

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
# Analysis functions (A1-A5: same structure as m10_05, adapted for PP)
# =========================================================================

def analysis_a1_recurrence_fo_correlation(pp_fo, recurrence_scores, n_states, out_dir):
    """A1: Recurrence-FO Correlation."""
    logger.info("A1: Recurrence-FO Correlation")

    fo_matrix = np.array(list(pp_fo.values()))
    mean_fo = np.mean(fo_matrix, axis=0)

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
            'Approximate: PP FO is compositional (sums to 1 per run), '
            'inducing weak negative dependence among state-level values. '
            'Effect is attenuated by cross-dataset design (recurrence from '
            'Friends, FO from PP). Additionally, PP chapters are consecutive '
            'sections of a continuous narrative (unlike Friends independent '
            'episodes), so serial dependence may inflate rho — interpret as '
            'an upper bound. See serial_dependence diagnostic for n_effective.')
        result['positive_correlation'] = bool(rho > 0 and p_val < 0.05)
    else:
        result['spearman_rho'] = None
        result['spearman_p'] = None
        result['positive_correlation'] = None

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [recurrence_color(recurrence_scores[i]) for i in active_indices]
    ax.scatter(rec_active, fo_active,
               c=colors, alpha=0.7, s=30, edgecolors='white', linewidths=0.5)
    ax.set_xlabel('Friends Recurrence Score')
    ax.set_ylabel('Mean PP FO')
    ax.set_title('A1: Friends Recurrence vs PP FO')
    make_recurrence_colorbar(ax)
    if result['spearman_rho'] is not None:
        rho_str = f"\u03c1 = {result['spearman_rho']:.3f}"
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


def analysis_a2_per_type(pp_fo, pp_run_ids, recurrence_scores, n_states, out_dir):
    """A2: Per-Type Breakdown (PP has 2 types: lppFR, lppEN; FDR across 2 tests)."""
    logger.info("A2: Per-Type Breakdown")

    active_mask = np.array([recurrence_scores[i] > 0 for i in range(n_states)])
    active_indices = np.where(active_mask)[0]
    rec_active = recurrence_scores[active_mask]

    type_results = {}

    for stype, run_ids in pp_run_ids.items():
        type_fo_arrays = [pp_fo[rid] for rid in run_ids if rid in pp_fo]
        if not type_fo_arrays:
            type_results[stype] = {'spearman_rho': None, 'spearman_p': None, 'n_runs': 0}
            continue

        type_mean_fo = np.mean(np.array(type_fo_arrays), axis=0)
        fo_active = type_mean_fo[active_mask]

        if len(rec_active) >= 3:
            rho, p_val = stats.spearmanr(rec_active, fo_active)
            type_results[stype] = {
                'spearman_rho': float(rho),
                'spearman_p': float(p_val),
                'spearman_p_note': 'Approximate due to FO compositionality; see A1 note.',
                'n_runs': len(type_fo_arrays),
            }
        else:
            type_results[stype] = {'spearman_rho': None, 'spearman_p': None, 'n_runs': len(type_fo_arrays)}

    # FDR correction across 2 types
    # Note: FR and EN are the same narrative in different languages, so their
    # FO-recurrence correlations are non-independent. The combined A1 result
    # is the primary inference; A2 per-language breakdown is descriptive.
    raw_pvals = []
    pval_keys = []
    for stype in pp_run_ids.keys():
        tr = type_results.get(stype, {})
        if tr.get('spearman_p') is not None:
            raw_pvals.append(tr['spearman_p'])
            pval_keys.append(stype)

    if len(raw_pvals) >= 2:
        corrected = benjamini_hochberg(raw_pvals)
        for stype, p_corr in zip(pval_keys, corrected):
            type_results[stype]['spearman_p_fdr'] = float(p_corr)

    # --- Plot: 1x2 subplots (one per language) ---
    stypes = list(pp_run_ids.keys())
    n_types = len(stypes)
    fig, axes = plt.subplots(1, n_types, figsize=(6 * n_types, 5))
    if n_types == 1:
        axes = [axes]

    for ax, stype in zip(axes, stypes):
        run_ids = pp_run_ids[stype]
        type_fo_arrays = [pp_fo[rid] for rid in run_ids if rid in pp_fo]
        if type_fo_arrays:
            type_mean_fo = np.mean(np.array(type_fo_arrays), axis=0)
        else:
            type_mean_fo = np.zeros(n_states)

        colors = [recurrence_color(recurrence_scores[idx]) for idx in active_indices]
        ax.scatter(rec_active, type_mean_fo[active_mask],
                   c=colors, alpha=0.7, s=20, edgecolors='white', linewidths=0.3)

        label = PP_TYPE_LABELS.get(stype, stype)
        ax.set_title(label)
        tr = type_results[stype]
        if tr['spearman_rho'] is not None:
            p_fdr_str = ''
            if 'spearman_p_fdr' in tr:
                p_fdr_str = f", p_fdr={tr['spearman_p_fdr']:.3f}"
            ax.annotate(f"\u03c1={tr['spearman_rho']:.3f}, p={tr['spearman_p']:.3f}{p_fdr_str}",
                        xy=(0.02, 0.98), xycoords='axes fraction',
                        ha='left', va='top', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        ax.set_xlabel('Friends Recurrence Score')
        ax.set_ylabel('Mean PP FO')
        make_recurrence_colorbar(ax)

    fig.suptitle('A2: Per-Type Recurrence-FO Correlation', fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'A2_per_type_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    return type_results


def analysis_a3_ll_comparison(ll_summary, out_dir):
    """A3: Log-Likelihood Comparison."""
    logger.info("A3: Log-Likelihood Comparison")

    friends_ll = ll_summary['friends_test_ll_per_sample']
    pp_ll = ll_summary['pp_overall_ll_per_sample']
    baseline_ll = ll_summary['baseline_ll_per_sample']

    result = {
        'friends_test_ll': friends_ll,
        'pp_overall_ll': pp_ll,
        'baseline_ll': baseline_ll,
        'baseline_note': (
            'Heuristic reference point only: log(1/n_active_states) is a '
            'uniform state-assignment baseline, not on the same scale as '
            'Gaussian-emission HMM log-likelihood.'),
        'll_gap': ll_summary['ll_gap_friends_minus_pp'],
        'pp_above_baseline': ll_summary['pp_above_baseline'],
        'll_note': (
            'PP LL may be lower than Movie10 LL partly because visual cortex PCs '
            'carry no stimulus-driven variance for audio-only data. '
            'See A5 network-stratified diagnostic.'),
    }

    per_type = {}
    for stype, info in ll_summary.get('per_type', {}).items():
        per_type[stype] = info.get('ll_per_sample')
    result['per_type_ll'] = per_type

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 5))

    labels = ['Friends\nTest', 'PP\nAudio', 'Baseline\n(uniform)']
    values = [friends_ll, pp_ll, baseline_ll]
    colors = ['#4477AA', '#AA3377', '#999999']

    bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.8, edgecolor='white')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('LL / sample')
    ax.set_title('A3: Log-Likelihood Comparison (PP Audio)')
    ax.axhline(y=baseline_ll, color='#999999', linestyle='--', alpha=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'A3_ll_comparison.png'), dpi=150)
    plt.close(fig)

    return result


def analysis_a4_state_coverage(pp_fo, pp_run_ids, recurrence_scores, n_states,
                                fo_threshold, out_dir):
    """A4: State Coverage Analysis.

    Spearman: Friends recurrence vs PP coverage (continuous metric).
    """
    logger.info("A4: State Coverage Analysis")

    active_states = [i for i in range(n_states) if recurrence_scores[i] > 0]
    active_states_sorted = sorted(active_states, key=lambda s: recurrence_scores[s], reverse=True)

    all_run_ids = []
    for stype in pp_run_ids:
        for rid in pp_run_ids.get(stype, []):
            if rid in pp_fo:
                all_run_ids.append(rid)

    fo_matrix = np.zeros((len(active_states_sorted), len(all_run_ids)))
    for j, rid in enumerate(all_run_ids):
        for i, state_id in enumerate(active_states_sorted):
            fo_matrix[i, j] = pp_fo[rid][state_id]

    coverage = {}
    for state_id in active_states_sorted:
        n_active_runs = sum(1 for rid in all_run_ids if pp_fo[rid][state_id] > fo_threshold)
        frac = n_active_runs / len(all_run_ids) if all_run_ids else 0
        coverage[int(state_id)] = {
            'pp_coverage': float(frac),
            'friends_recurrence': float(recurrence_scores[state_id]),
        }

    # Spearman: Friends recurrence vs PP coverage
    rec_vals = np.array([recurrence_scores[s] for s in active_states_sorted])
    cov_vals = np.array([coverage[s]['pp_coverage'] for s in active_states_sorted])
    coverage_corr = {}
    if len(rec_vals) >= 5:
        rho, p = stats.spearmanr(rec_vals, cov_vals)
        coverage_corr = {'rho': float(rho), 'p': float(p), 'n': len(rec_vals)}

    inactive_states = [i for i in range(n_states) if recurrence_scores[i] == 0]
    inactive_activated = {}
    for state_id in inactive_states:
        pp_mean_fo = float(np.mean([pp_fo[rid][state_id] for rid in all_run_ids]))
        if pp_mean_fo > fo_threshold:
            inactive_activated[int(state_id)] = float(pp_mean_fo)

    result = {
        'state_coverage': coverage,
        'recurrence_vs_coverage_spearman': coverage_corr,
        'n_active_states_plotted': len(active_states_sorted),
        'inactive_states_activated_in_pp': inactive_activated,
    }

    # --- Plot ---
    if len(active_states_sorted) > 0 and len(all_run_ids) > 0:
        fig_height = max(4, len(active_states_sorted) * 0.15 + 2)
        fig_width = max(6, len(all_run_ids) * 0.4 + 3)
        fig, ax = plt.subplots(figsize=(min(fig_width, 12), min(fig_height, 30)))

        im = ax.imshow(fo_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        plt.colorbar(im, ax=ax, label='Fractional Occupancy', shrink=0.8)

        y_labels = []
        y_colors = []
        for state_id in active_states_sorted:
            y_labels.append(f"S{state_id}")
            y_colors.append(recurrence_color(recurrence_scores[state_id]))

        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=5)
        for tick, color in zip(ax.get_yticklabels(), y_colors):
            tick.set_color(color)

        # X-axis labels: short form "FR-1", "FR-2", ..., "EN-1", "EN-2", etc.
        x_labels = []
        type_counters = {}
        for rid in all_run_ids:
            # Determine which type this run belongs to
            run_type = None
            for stype, rids in pp_run_ids.items():
                if rid in rids:
                    run_type = stype
                    break
            if run_type is not None:
                prefix = 'FR' if run_type == 'lppFR' else 'EN'
                type_counters[run_type] = type_counters.get(run_type, 0) + 1
                x_labels.append(f"{prefix}-{type_counters[run_type]}")
            else:
                x_labels.append(rid)

        ax.set_xticks(range(len(all_run_ids)))
        ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha='right')
        ax.set_xlabel('PP Runs (chapters)')
        ax.set_ylabel('States (sorted by Friends recurrence, descending)')
        ax.set_title('A4: State Coverage Across PP Runs')

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'A4_state_coverage_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    return result


def analysis_a5_pca_diagnostic(pca_diagnostic, out_dir):
    """A5: PCA Transfer Diagnostic (overall + network-stratified R²)."""
    logger.info("A5: PCA Transfer Diagnostic")

    result = {
        'friends_r2': pca_diagnostic['friends_r2_n_pcs'],
        'pp_r2': pca_diagnostic['pp_r2_n_pcs'],
        'transfer_gap': pca_diagnostic['transfer_gap'],
        'flag_low_variance': pca_diagnostic['flag_low_variance'],
        'n_pcs': pca_diagnostic['n_pcs'],
        'note': ('PP uses audio-only narration, so visual cortex PCs carry '
                 'no stimulus-driven variance for audio-only data. Lower PP R² '
                 'than Movie10 is expected and does not invalidate FO-based analyses (A1).'),
    }

    r2_by_network = pca_diagnostic.get('r2_by_network', {})
    if r2_by_network:
        result['r2_by_network'] = r2_by_network

    # --- Plot: overall + per-network ---
    has_network = bool(r2_by_network)
    if has_network:
        fig, (ax_overall, ax_net) = plt.subplots(1, 2, figsize=(12, 5),
                                                  gridspec_kw={'width_ratios': [1, 2.5]})
    else:
        fig, ax_overall = plt.subplots(figsize=(5, 4))

    # Left panel: overall R²
    labels = ['Friends\n(training)', 'PP\n(audio)']
    values = [pca_diagnostic['friends_r2_n_pcs'], pca_diagnostic['pp_r2_n_pcs']]
    colors = ['#4477AA', '#AA3377']

    bars = ax_overall.bar(labels, values, color=colors, alpha=0.8, width=0.5)
    ax_overall.set_ylabel(f'Variance Explained (R\u00b2, {pca_diagnostic["n_pcs"]} PCs)')
    ax_overall.set_title('Overall R\u00b2')
    ax_overall.set_ylim(0, 1.0)
    ax_overall.axhline(y=0.70, color='red', linestyle='--', alpha=0.5, label='70% threshold')
    ax_overall.legend(fontsize=8)

    for bar, val in zip(bars, values):
        ax_overall.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # Right panel: per-network R²
    if has_network:
        net_names = [n for n in NETWORK_ORDER if n in r2_by_network]
        net_r2 = [r2_by_network[n]['r2_n_pcs'] for n in net_names]
        net_colors = [NETWORK_COLORS.get(n, '#888888') for n in net_names]

        x = np.arange(len(net_names))
        bars_net = ax_net.bar(x, net_r2, color=net_colors, alpha=0.8, width=0.7)
        ax_net.set_xticks(x)
        ax_net.set_xticklabels(net_names, rotation=45, ha='right', fontsize=8)
        ax_net.set_ylabel(f'PP R\u00b2 ({pca_diagnostic["n_pcs"]} PCs)')
        ax_net.set_title('Per-Network R\u00b2 (PP Audio)')
        ax_net.set_ylim(0, 1.0)
        ax_net.axhline(y=0.70, color='red', linestyle='--', alpha=0.5)

        for bar, val in zip(bars_net, net_r2):
            ax_net.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle('A5: PCA Transfer Diagnostic (PP Audio)', fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'A5_pca_diagnostic.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    return result


# =========================================================================
# PP-specific analyses (B1, B2)
# =========================================================================

def analysis_b1_movie10_baseline(sub_id, parc, vt, recurrence_scores, n_states):
    """B1: 5-Subject Movie10 Baseline.

    Recompute Movie10 A1 (recurrence-FO correlation) for the same subject
    (to provide direct comparison with PP results).
    """
    logger.info("B1: Movie10 Baseline Comparison")

    # Load Movie10 FO if available
    movie_dir = os.path.join(SCRATCH_DIR, 'output', 'm10_04_decoded', parc, sub_id)
    if vt is not None:
        movie_dir = os.path.join(movie_dir, f'vt{vt}')
    movie_fo_path = os.path.join(movie_dir, 'fractional_occupancy.pkl')

    if not os.path.exists(movie_fo_path):
        logger.warning(f"Movie10 FO not found at {movie_fo_path} -- skipping B1")
        return {'available': False, 'note': 'Movie10 data not available for this subject'}

    with open(movie_fo_path, 'rb') as f:
        movie_fo = pickle.load(f)

    fo_matrix = np.array(list(movie_fo.values()))
    mean_fo = np.mean(fo_matrix, axis=0)

    result = {
        'available': True,
        'n_movie_runs': len(movie_fo),
    }

    # A1-equivalent
    active_mask = np.array([recurrence_scores[i] > 0 for i in range(n_states)])
    rec_active = recurrence_scores[active_mask]
    fo_active = mean_fo[active_mask]

    if len(rec_active) >= 3:
        rho, p_val = stats.spearmanr(rec_active, fo_active)
        result['movie10_spearman_rho'] = float(rho)
        result['movie10_spearman_p'] = float(p_val)

    return result


def analysis_b2_bootstrap_reference(sub_id, parc, vt, pp_fo,
                                     recurrence_scores, n_states,
                                     n_bootstrap=1000, out_dir=None):
    """B2: Bootstrap Matched-Sample-Size Reference.

    Draw 18 random Movie10 runs (1000x), compute A1 statistics, build
    reference distribution. Compare PP effect sizes to this distribution.
    """
    logger.info("B2: Bootstrap Matched-Sample-Size Reference")

    # Load Movie10 FO
    movie_dir = os.path.join(SCRATCH_DIR, 'output', 'm10_04_decoded', parc, sub_id)
    if vt is not None:
        movie_dir = os.path.join(movie_dir, f'vt{vt}')
    movie_fo_path = os.path.join(movie_dir, 'fractional_occupancy.pkl')

    if not os.path.exists(movie_fo_path):
        logger.warning(f"Movie10 FO not found -- skipping B2")
        return {'available': False}

    with open(movie_fo_path, 'rb') as f:
        movie_fo = pickle.load(f)

    movie_run_ids = list(movie_fo.keys())
    n_pp_runs = len(pp_fo)

    if len(movie_run_ids) < n_pp_runs:
        logger.warning(f"Movie10 has fewer runs ({len(movie_run_ids)}) than PP ({n_pp_runs}) -- skipping B2")
        return {'available': False}

    active_mask = np.array([recurrence_scores[i] > 0 for i in range(n_states)])
    rec_active = recurrence_scores[active_mask]

    rng = np.random.default_rng(42)
    bootstrap_rho = []

    for _ in range(n_bootstrap):
        # Sample n_pp_runs Movie10 runs without replacement
        sample_ids = rng.choice(movie_run_ids, size=n_pp_runs, replace=False)
        sample_fo = {rid: movie_fo[rid] for rid in sample_ids}

        fo_matrix = np.array(list(sample_fo.values()))
        mean_fo = np.mean(fo_matrix, axis=0)

        # A1: Spearman rho
        fo_active = mean_fo[active_mask]
        if len(rec_active) >= 3:
            rho, _ = stats.spearmanr(rec_active, fo_active)
            bootstrap_rho.append(rho)

    # Compute PP effect sizes for comparison
    pp_fo_matrix = np.array(list(pp_fo.values()))
    pp_mean_fo = np.mean(pp_fo_matrix, axis=0)

    pp_fo_active = pp_mean_fo[active_mask]
    pp_rho = None
    if len(rec_active) >= 3:
        pp_rho, _ = stats.spearmanr(rec_active, pp_fo_active)

    result = {
        'available': True,
        'n_bootstrap': n_bootstrap,
        'n_subsample_runs': n_pp_runs,
        'n_movie_runs_total': len(movie_run_ids),
    }

    if bootstrap_rho:
        bootstrap_rho = np.array(bootstrap_rho)
        result['movie10_bootstrap_rho_mean'] = float(np.mean(bootstrap_rho))
        result['movie10_bootstrap_rho_ci95'] = [float(np.percentile(bootstrap_rho, 2.5)),
                                                  float(np.percentile(bootstrap_rho, 97.5))]
        if pp_rho is not None:
            result['pp_rho'] = float(pp_rho)
            result['pp_rho_percentile'] = float(np.mean(bootstrap_rho <= pp_rho) * 100)

    # --- Plot ---
    if out_dir and len(bootstrap_rho) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        ax.hist(bootstrap_rho, bins=40, alpha=0.6, color='#4477AA', label=f'Movie10 ({n_pp_runs}-run samples)')
        if pp_rho is not None:
            ax.axvline(pp_rho, color='#AA3377', linewidth=2, label=f'PP (\u03c1={pp_rho:.3f})')
        ax.set_xlabel('Spearman \u03c1')
        ax.set_ylabel('Count')
        ax.set_title('A1: Recurrence-FO Correlation')
        ax.legend(fontsize=8)

        fig.suptitle(f'B2: PP vs Movie10 Bootstrap Reference (matched n={n_pp_runs} runs)', fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'B2_bootstrap_reference.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    return result


def analysis_serial_dependence(pp_fo, pp_run_ids, n_states, out_dir, type_labels=None):
    """Compute between-run FO autocorrelation for serial dependence diagnostic.

    PP runs are consecutive chapters within each language, not independent episodes.
    This reports how correlated FO vectors are between consecutive runs,
    separately for each language (lppFR and lppEN).

    Parameters
    ----------
    pp_fo : dict
        Run ID -> FO vector mapping.
    pp_run_ids : dict
        Type -> list of run IDs mapping.
    n_states : int
        Number of states.
    out_dir : str
        Output directory.
    type_labels : dict, optional
        Type -> display label mapping. Defaults to PP_TYPE_LABELS.
    """
    logger.info("Serial Dependence Diagnostic")

    if type_labels is None:
        type_labels = PP_TYPE_LABELS

    lang_results = {}
    n_langs = len(pp_run_ids)

    fig, axes = plt.subplots(n_langs, 2, figsize=(10, 4 * n_langs))
    if n_langs == 1:
        axes = axes[np.newaxis, :]

    for row_idx, stype in enumerate(sorted(pp_run_ids.keys())):
        run_ids = pp_run_ids[stype]
        ordered_runs = [rid for rid in run_ids if rid in pp_fo]
        lang_label = type_labels.get(stype, stype)

        if len(ordered_runs) < 3:
            lang_results[stype] = {
                'n_runs': len(ordered_runs),
                'note': 'Too few runs for autocorrelation',
            }
            continue

        fo_matrix = np.array([pp_fo[rid] for rid in ordered_runs])  # (n_runs, n_states)

        # Lag-1 autocorrelation: corr(FO_i, FO_{i+1}) for consecutive runs
        lag1_corrs = []
        for i in range(len(ordered_runs) - 1):
            r, _ = stats.pearsonr(fo_matrix[i], fo_matrix[i + 1])
            lag1_corrs.append(float(r))

        mean_lag1 = float(np.mean(lag1_corrs))

        # Full between-run correlation matrix
        corr_matrix = np.corrcoef(fo_matrix)  # (n_runs, n_runs)

        # Effective number of independent samples (Bartlett's formula approximation)
        n_runs = len(ordered_runs)
        if abs(mean_lag1) < 1.0:
            n_effective = n_runs * (1 - mean_lag1) / (1 + mean_lag1)
            n_effective = max(2.0, min(float(n_runs), n_effective))
        else:
            n_effective = 2.0

        lang_results[stype] = {
            'n_runs': n_runs,
            'lag1_correlations': lag1_corrs,
            'mean_lag1_autocorrelation': mean_lag1,
            'n_effective_independent': float(round(n_effective, 1)),
            'note': (f'PP {lang_label} runs are consecutive chapters with serial dependence. '
                     'n_effective estimates independent observations for p-value interpretation.'),
        }

        # --- Plot row for this language ---
        ax1, ax2 = axes[row_idx, 0], axes[row_idx, 1]

        # Left: lag-1 autocorrelations
        ax1.bar(range(len(lag1_corrs)), lag1_corrs, color='#4477AA', alpha=0.8)
        ax1.set_xlabel('Run pair (i, i+1)')
        ax1.set_ylabel('Pearson r (FO vectors)')
        ax1.set_title(f'{lang_label}: Lag-1 (mean = {mean_lag1:.3f})')
        ax1.set_xticks(range(len(lag1_corrs)))
        ax1.set_xticklabels([f'{i+1}-{i+2}' for i in range(len(lag1_corrs))], fontsize=7)
        ax1.axhline(0, color='grey', linewidth=0.5)

        # Right: full correlation matrix
        im = ax2.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        plt.colorbar(im, ax=ax2, shrink=0.8, label='Pearson r')
        ax2.set_xticks(range(n_runs))
        ax2.set_yticks(range(n_runs))
        ax2.set_xticklabels([f'R{i+1}' for i in range(n_runs)], fontsize=7)
        ax2.set_yticklabels([f'R{i+1}' for i in range(n_runs)], fontsize=7)
        ax2.set_title(f'{lang_label}: Between-Run FO Corr (n_eff = {n_effective:.1f})')

    fig.suptitle('Serial Dependence: PP Chapters (per language)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'serial_dependence.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    return lang_results


def analysis_language_comparison(pp_fo, pp_run_ids, n_states, out_dir):
    """Compare FO profiles between French and English.

    Computes cosine similarity between French and English mean FO vectors,
    and produces a bar chart comparing per-state FO for the top 10 most
    occupied states.
    """
    logger.info("Language Comparison: French vs English")

    # Compute mean FO per language
    lang_mean_fo = {}
    for stype, run_ids in pp_run_ids.items():
        type_fo_arrays = [pp_fo[rid] for rid in run_ids if rid in pp_fo]
        if type_fo_arrays:
            lang_mean_fo[stype] = np.mean(np.array(type_fo_arrays), axis=0)
        else:
            lang_mean_fo[stype] = np.zeros(n_states)

    result = {}

    stypes = list(pp_run_ids.keys())
    if len(stypes) >= 2 and stypes[0] in lang_mean_fo and stypes[1] in lang_mean_fo:
        fo_fr = lang_mean_fo.get('lppFR', lang_mean_fo[stypes[0]])
        fo_en = lang_mean_fo.get('lppEN', lang_mean_fo[stypes[1]])

        # Cosine similarity
        cos_sim = 1.0 - cosine_distance(fo_fr, fo_en)
        result['cosine_similarity'] = float(cos_sim)

        # Pearson correlation
        r, p = stats.pearsonr(fo_fr, fo_en)
        result['pearson_r'] = float(r)
        result['pearson_p'] = float(p)

        # Top 10 most occupied states (by combined mean FO)
        combined_fo = (fo_fr + fo_en) / 2.0
        top_states = np.argsort(combined_fo)[::-1][:10]
        result['top_10_states'] = [int(s) for s in top_states]
        result['top_10_fo_fr'] = [float(fo_fr[s]) for s in top_states]
        result['top_10_fo_en'] = [float(fo_en[s]) for s in top_states]

        # --- Plot: bar chart comparing per-state FO for top 10 ---
        fig, ax = plt.subplots(figsize=(10, 5))

        x = np.arange(len(top_states))
        width = 0.35

        bars_fr = ax.bar(x - width / 2, [fo_fr[s] for s in top_states],
                         width, label='French (lppFR)', color='#4477AA', alpha=0.8)
        bars_en = ax.bar(x + width / 2, [fo_en[s] for s in top_states],
                         width, label='English (lppEN)', color='#AA3377', alpha=0.8)

        ax.set_xlabel('State')
        ax.set_ylabel('Mean Fractional Occupancy')
        ax.set_title(f'Language Comparison: Top 10 States (cosine sim = {cos_sim:.3f})')
        ax.set_xticks(x)
        ax.set_xticklabels([f'S{s}' for s in top_states], fontsize=8)
        ax.legend(fontsize=9)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'language_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        result['note'] = 'Fewer than 2 language types available; comparison skipped.'

    return result


# =========================================================================
# Main
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Cross-stimulus validation: test Friends brain state generalization to PP audio narration.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script/pp_05_cross_stimulus_validation.py --sub_id sub-01
  python script/pp_05_cross_stimulus_validation.py --sub_id sub-01 --parcellation atlas-4S456Parcels
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
                             "(default: False).")
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                        help='Number of bootstrap samples for B2 (default: 1000)')
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

    # PP decoded states + FO (vt-aware)
    pp_dir = os.path.join(SCRATCH_DIR, 'output', 'pp_04_decoded', parc, sub_id)
    if args.vt is not None:
        pp_dir = os.path.join(pp_dir, f'vt{args.vt}')
    pp_fo_path = os.path.join(pp_dir, 'fractional_occupancy.pkl')
    pp_ll_path = os.path.join(pp_dir, 'pp_ll_summary.json')

    # PP run IDs (vt-aware)
    proj_dir = os.path.join(SCRATCH_DIR, 'output', 'pp_03_projected', parc, sub_id)
    if args.vt is not None:
        proj_dir = os.path.join(proj_dir, f'vt{args.vt}')
    pp_run_ids_path = os.path.join(proj_dir, 'pp_run_ids.json')

    # run_id_map.json from pp_04 — maps long BIDS keys to short keys
    run_id_map_path = os.path.join(pp_dir, 'run_id_map.json')

    # PCA diagnostic
    pca_diag_path = os.path.join(proj_dir, 'pca_transfer_diagnostic.json')

    # Validate inputs
    required_files = {
        recurrence_summary_path: 'recurrence_summary.json (run 05a first)',
        recurrence_scores_path: 'recurrence_scores.npy (run 05a first)',
        pp_fo_path: 'PP FO (run pp_04 first)',
        pp_ll_path: 'PP LL summary (run pp_04 first)',
        pp_run_ids_path: 'pp_run_ids.json (run pp_03 first)',
        run_id_map_path: 'run_id_map.json (run pp_04 first)',
        pca_diag_path: 'pca_transfer_diagnostic.json (run pp_03 first)',
    }
    for path, label in required_files.items():
        if not os.path.exists(path):
            logger.error(f"Missing: {path} -- {label}")
            sys.exit(1)

    # =========================================================================
    # Load data
    # =========================================================================

    with open(recurrence_summary_path, 'r') as f:
        recurrence_summary = json.load(f)

    recurrence_scores = np.load(recurrence_scores_path)
    n_states = len(recurrence_scores)

    with open(pp_fo_path, 'rb') as f:
        pp_fo = pickle.load(f)

    with open(pp_ll_path, 'r') as f:
        ll_summary = json.load(f)

    with open(pp_run_ids_path, 'r') as f:
        pp_run_ids_raw = json.load(f)

    with open(run_id_map_path, 'r') as f:
        run_id_map = json.load(f)

    # pp_run_ids.json uses long BIDS keys (from pp_03) but pp_fo.pkl uses
    # short keys (from pp_04's normalize_cross_stim_run_id). Remap.
    long_to_short = run_id_map['long_to_short']
    pp_run_ids = {}
    for stype, long_ids in pp_run_ids_raw.items():
        short_ids = []
        for lid in long_ids:
            short = long_to_short.get(lid)
            if short is None:
                logger.warning("run_id_map has no short key for %s — skipping", lid)
                continue
            short_ids.append(short)
        pp_run_ids[stype] = short_ids

    with open(pca_diag_path, 'r') as f:
        pca_diagnostic = json.load(f)

    # Sub-HRF filtering
    excluded_sub_hrf = []
    if args.exclude_sub_hrf:
        try:
            _, excluded_ids, _ = load_eligible_states(recurrence_dir)
            excluded_set = set(int(s) for s in excluded_ids)
            for sid in excluded_set:
                recurrence_scores[sid] = 0.0
            excluded_sub_hrf = sorted(excluded_set)
            logger.info("Sub-HRF exclusion ON: zeroed recurrence for %d states", len(excluded_set))
        except FileNotFoundError:
            logger.warning("eligible_states.json not found; sub-HRF filtering skipped.")

    if not pp_fo:
        logger.error("No PP runs in fractional_occupancy.pkl -- check pp_04 outputs")
        sys.exit(1)

    n_active = int(np.sum(recurrence_scores > 0))
    logger.info(f"Loaded data for {sub_id}: {n_states} states, {len(pp_fo)} PP runs")
    logger.info(f"Active states (recurrence > 0): {n_active}")

    # =========================================================================
    # Output directory
    # =========================================================================

    out_dir = os.path.join(SCRATCH_DIR, 'output', 'pp_05_cross_validation', parc, sub_id)
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
        'stimulus': 'petit-prince',
        'stimulus_modality': 'audio_only',
        'n_states': n_states,
        'n_pp_runs': len(pp_fo),
        'fo_threshold': fo_threshold,
        'exclude_sub_hrf': args.exclude_sub_hrf,
        'n_excluded_sub_hrf': len(excluded_sub_hrf),
        'excluded_sub_hrf_states': excluded_sub_hrf,
        'n_active_states': n_active,
    }

    summary['A1_recurrence_correlation'] = analysis_a1_recurrence_fo_correlation(
        pp_fo, recurrence_scores, n_states, out_dir)
    summary['A2_per_type'] = analysis_a2_per_type(
        pp_fo, pp_run_ids, recurrence_scores, n_states, out_dir)

    # Content-eligible-subset variants (project-wide 05e_a4 convention).
    # Mirrors A1/A2 but restricts both vectors to the subset of states flagged
    # as ``eligible_for_content_analysis`` in 05e_a4's state_flags.csv
    # (intersected with active states). The full-repertoire A1/A2 above are
    # preserved unchanged so downstream figures can show both panels.
    eligibility = load_content_eligibility(
        sub_id=sub_id, parcellation=parc, scratch_dir=SCRATCH_DIR, vt=args.vt,
    )
    eligible_variants = compute_eligible_recurrence_correlations(
        pp_fo, pp_run_ids, recurrence_scores, n_states, eligibility,
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
        pp_fo, pp_run_ids, recurrence_scores, n_states, fo_threshold, out_dir)
    summary['A5_pca_diagnostic'] = analysis_a5_pca_diagnostic(pca_diagnostic, out_dir)

    # PP-specific analyses
    summary['B1_movie10_baseline'] = analysis_b1_movie10_baseline(
        sub_id, parc, args.vt, recurrence_scores, n_states)
    summary['B2_bootstrap_reference'] = analysis_b2_bootstrap_reference(
        sub_id, parc, args.vt, pp_fo, recurrence_scores, n_states,
        n_bootstrap=args.n_bootstrap, out_dir=out_dir)

    # Serial dependence diagnostic (PP chapters are not independent)
    summary['serial_dependence'] = analysis_serial_dependence(
        pp_fo, pp_run_ids, n_states, out_dir, type_labels=PP_TYPE_LABELS)

    # Language comparison (French vs English FO profiles)
    summary['language_comparison'] = analysis_language_comparison(
        pp_fo, pp_run_ids, n_states, out_dir)

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
    b1 = summary['B1_movie10_baseline']
    b2 = summary['B2_bootstrap_reference']

    print(f"\n{'='*60}")
    print(f"CROSS-STIMULUS VALIDATION SUMMARY (PETIT PRINCE AUDIO)")
    print(f"{'='*60}")
    print(f"Subject: {sub_id} | Parcellation: {parc}")
    print(f"PP runs: {len(pp_fo)} | States: {n_states}")
    print(f"")
    print(f"A1 - Recurrence-FO Correlation:")
    if a1.get('spearman_rho') is not None:
        print(f"  Full repertoire:   rho={a1['spearman_rho']:.3f}, p={a1['spearman_p']:.4f}")
    a1e = summary.get('A1_recurrence_correlation_eligible', {})
    if a1e.get('spearman_rho') is not None:
        print(f"  Content-eligible:  rho={a1e['spearman_rho']:.3f}, p={a1e['spearman_p']:.4f} "
              f"(n={a1e.get('n_eligible_active', 0)}, source={a1e.get('eligibility_source')})")
    elif a1e:
        print(f"  Content-eligible:  n_eligible_active={a1e.get('n_eligible_active', 0)} "
              f"(too few for Spearman; source={a1e.get('eligibility_source')})")
    print(f"")
    print(f"A2 - Per-Type Breakdown:")
    for stype, tr in summary['A2_per_type'].items():
        label = PP_TYPE_LABELS.get(stype, stype)
        if tr.get('spearman_rho') is not None:
            p_fdr_str = f", p_fdr={tr['spearman_p_fdr']:.4f}" if 'spearman_p_fdr' in tr else ''
            print(f"  {label}: rho={tr['spearman_rho']:.3f}, p={tr['spearman_p']:.4f}{p_fdr_str}")
    print(f"")
    print(f"A3 - LL Comparison:")
    print(f"  Friends test:  {a3['friends_test_ll']:.4f}")
    print(f"  PP overall:    {a3['pp_overall_ll']:.4f}")
    print(f"  Baseline:      {a3['baseline_ll']:.4f}")
    print(f"  PP > base:     {a3['pp_above_baseline']}")
    print(f"")
    print(f"A4 - State Coverage:")
    cov_corr = a4.get('recurrence_vs_coverage_spearman', {})
    if cov_corr:
        print(f"  Recurrence-coverage Spearman rho: {cov_corr['rho']:.3f}, p={cov_corr['p']:.4f}")
    print(f"")
    print(f"A5 - PCA Diagnostic:")
    print(f"  Friends R2: {a5['friends_r2']:.4f}, PP R2: {a5['pp_r2']:.4f}")
    if a5['flag_low_variance']:
        print(f"  *** WARNING: PP R2 < 0.70 ***")
    if a5.get('r2_by_network'):
        print(f"  Per-network PP R2:")
        for net in NETWORK_ORDER:
            if net in a5['r2_by_network']:
                print(f"    {net:15s}: {a5['r2_by_network'][net]['r2_n_pcs']:.4f}")
    print(f"")
    print(f"B1 - Movie10 Baseline:")
    if b1.get('available'):
        if b1.get('movie10_spearman_rho') is not None:
            print(f"  Movie10 Spearman rho: {b1['movie10_spearman_rho']:.3f}")
    else:
        print(f"  Movie10 data not available")
    print(f"")
    print(f"B2 - Bootstrap Reference:")
    if b2.get('available'):
        if b2.get('pp_rho') is not None:
            print(f"  PP rho: {b2['pp_rho']:.3f} (percentile: {b2.get('pp_rho_percentile', '?'):.1f}%)")
        if b2.get('movie10_bootstrap_rho_ci95'):
            ci = b2['movie10_bootstrap_rho_ci95']
            print(f"  Movie10 {summary['n_pp_runs']}-run rho 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    else:
        print(f"  Movie10 data not available")
    print(f"")
    sd = summary['serial_dependence']
    print(f"Serial Dependence (per language):")
    for stype in sorted(sd.keys()):
        label = PP_TYPE_LABELS.get(stype, stype)
        lang_sd = sd[stype]
        print(f"  {label}:")
        print(f"    Mean lag-1 autocorr: {lang_sd.get('mean_lag1_autocorrelation', 'N/A')}")
        print(f"    Effective n:         {lang_sd.get('n_effective_independent', 'N/A')}")
    print(f"")
    lc = summary.get('language_comparison', {})
    if lc.get('cosine_similarity') is not None:
        print(f"Language Comparison:")
        print(f"  Cosine similarity (FR vs EN): {lc['cosine_similarity']:.4f}")
        print(f"  Pearson r: {lc['pearson_r']:.4f}, p={lc['pearson_p']:.4f}")
    print(f"{'='*60}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
