#!/usr/bin/env python3
"""
hp_05_cross_stimulus_validation.py - Cross-stimulus validation: Harry Potter word-by-word reading.

Tests whether Friends-recurring brain states generalize to unimodal reading
(Harry Potter word-by-word RSVP at 2 Hz) by comparing HP fractional occupancy against
Friends recurrence scores.

Harry Potter uses word-by-word visual reading (RSVP, word presentation on screen),
unlike Friends and Movie10 which are audiovisual. This removes scene-based visual and
auditory input, providing only orthographic visual input. This tests whether recurring
states require rich audiovisual input or reflect supramodal narrative/social processing.

Analyses:
    A1. Recurrence-FO Correlation (Spearman: Friends recurrence vs mean HP FO)
    A2. Per-Type Breakdown (single type for HP; included for structural consistency)
    A3. Log-Likelihood Comparison (Friends test vs HP vs baseline)
    A4. State Coverage (heatmap: state × HP run FO)
    A5. PCA Transfer Diagnostic (Friends vs HP R², with network-stratified breakdown)

Additional HP-specific analyses:
    B1. 5-Subject Movie10 Baseline (recompute Movie10 A1 for direct comparison)
    B2. Bootstrap Matched-Sample-Size Reference (7-run Movie10 subsamples)

Prerequisites:
    - hp_04_score_and_decode.py completed (decoded states, FO, LL summary)
    - 05a_recurrence_analysis.py completed (state categories)
    - hp_03_project_hp_pca.py completed (PCA diagnostic)
    - m10_04_score_and_decode.py completed (for B1/B2 Movie10 comparison)

Outputs:
    {SCRATCH_DIR}/output/hp_05_cross_validation/{parcellation}/{sub_id}/
        cross_stimulus_summary.json   - All test results, effect sizes, p-values
        A1_recurrence_fo_scatter.png  - Scatter: recurrence score vs HP FO
        A2_per_type_scatter.png       - Single scatter (HP has 1 type)
        A3_ll_comparison.png          - Bar chart: LL comparison
        A4_state_coverage_heatmap.png - Heatmap: state × HP run FO
        A5_pca_diagnostic.png         - Bar chart: Friends vs HP R²
        B2_bootstrap_reference.png    - HP effect sizes vs Movie10 bootstrap

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

HP_TYPE_LABELS = {
    'harrypotter': 'Harry Potter (Reading)',
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
# Analysis functions (A1-A5: same structure as m10_05, adapted for HP)
# =========================================================================

def analysis_a1_recurrence_fo_correlation(hp_fo, recurrence_scores, n_states, out_dir):
    """A1: Recurrence-FO Correlation."""
    logger.info("A1: Recurrence-FO Correlation")

    fo_matrix = np.array(list(hp_fo.values()))
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
            'Approximate: HP FO is compositional (sums to 1 per run), '
            'inducing weak negative dependence among state-level values. '
            'Effect is attenuated by cross-dataset design (recurrence from '
            'Friends, FO from HP).')
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
    ax.set_ylabel('Mean HP FO')
    ax.set_title('A1: Friends Recurrence vs HP FO')
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


def analysis_a2_per_type(hp_fo, hp_run_ids, recurrence_scores, n_states, out_dir):
    """A2: Per-Type Breakdown (HP has 1 type -- report single Spearman, no FDR)."""
    logger.info("A2: Per-Type Breakdown")

    active_mask = np.array([recurrence_scores[i] > 0 for i in range(n_states)])
    active_indices = np.where(active_mask)[0]
    rec_active = recurrence_scores[active_mask]

    type_results = {}

    for stype, run_ids in hp_run_ids.items():
        type_fo_arrays = [hp_fo[rid] for rid in run_ids if rid in hp_fo]
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

    # No FDR correction needed (single type)

    # --- Plot ---
    stypes = list(hp_run_ids.keys())
    fig, ax = plt.subplots(figsize=(6, 5))

    for stype in stypes:
        run_ids = hp_run_ids[stype]
        type_fo_arrays = [hp_fo[rid] for rid in run_ids if rid in hp_fo]
        if type_fo_arrays:
            type_mean_fo = np.mean(np.array(type_fo_arrays), axis=0)
        else:
            type_mean_fo = np.zeros(n_states)

        colors = [recurrence_color(recurrence_scores[idx]) for idx in active_indices]
        ax.scatter(rec_active, type_mean_fo[active_mask],
                   c=colors, alpha=0.7, s=20, edgecolors='white', linewidths=0.3)

        label = HP_TYPE_LABELS.get(stype, stype)
        ax.set_title(label)
        tr = type_results[stype]
        if tr['spearman_rho'] is not None:
            ax.annotate(f"ρ={tr['spearman_rho']:.3f}, p={tr['spearman_p']:.3f}",
                        xy=(0.02, 0.98), xycoords='axes fraction',
                        ha='left', va='top', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    ax.set_xlabel('Friends Recurrence Score')
    ax.set_ylabel('Mean HP FO')
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
    hp_ll = ll_summary['hp_overall_ll_per_sample']
    baseline_ll = ll_summary['baseline_ll_per_sample']

    result = {
        'friends_test_ll': friends_ll,
        'hp_overall_ll': hp_ll,
        'baseline_ll': baseline_ll,
        'baseline_note': (
            'Heuristic reference point only: log(1/n_active_states) is a '
            'uniform state-assignment baseline, not on the same scale as '
            'Gaussian-emission HMM log-likelihood.'),
        'll_gap': ll_summary['ll_gap_friends_minus_hp'],
        'hp_above_baseline': ll_summary['hp_above_baseline'],
        'll_note': (
            'HP LL may be lower than Movie10 LL partly because visual cortex PCs '
            'carry word-form but no scene-based variance for reading-only data. '
            'See A5 network-stratified diagnostic.'),
    }

    per_type = {}
    for stype, info in ll_summary.get('per_type', {}).items():
        per_type[stype] = info.get('ll_per_sample')
    result['per_type_ll'] = per_type

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 5))

    labels = ['Friends\nTest', 'HP\nOverall', 'Baseline\n(uniform)']
    values = [friends_ll, hp_ll, baseline_ll]
    colors = ['#4477AA', '#AA3377', '#999999']

    bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.8, edgecolor='white')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('LL / sample')
    ax.set_title('A3: Log-Likelihood Comparison (HP Reading)')
    ax.axhline(y=baseline_ll, color='#999999', linestyle='--', alpha=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'A3_ll_comparison.png'), dpi=150)
    plt.close(fig)

    return result


def analysis_a4_state_coverage(hp_fo, hp_run_ids, recurrence_scores, n_states,
                                fo_threshold, out_dir):
    """A4: State Coverage Analysis.

    Spearman: Friends recurrence vs HP coverage (continuous metric).
    """
    logger.info("A4: State Coverage Analysis")

    active_states = [i for i in range(n_states) if recurrence_scores[i] > 0]
    active_states_sorted = sorted(active_states, key=lambda s: recurrence_scores[s], reverse=True)

    all_run_ids = []
    for stype in hp_run_ids:
        for rid in hp_run_ids.get(stype, []):
            if rid in hp_fo:
                all_run_ids.append(rid)

    fo_matrix = np.zeros((len(active_states_sorted), len(all_run_ids)))
    for j, rid in enumerate(all_run_ids):
        for i, state_id in enumerate(active_states_sorted):
            fo_matrix[i, j] = hp_fo[rid][state_id]

    coverage = {}
    for state_id in active_states_sorted:
        n_active_runs = sum(1 for rid in all_run_ids if hp_fo[rid][state_id] > fo_threshold)
        frac = n_active_runs / len(all_run_ids) if all_run_ids else 0
        coverage[int(state_id)] = {
            'hp_coverage': float(frac),
            'friends_recurrence': float(recurrence_scores[state_id]),
        }

    # Spearman: Friends recurrence vs HP coverage
    rec_vals = np.array([recurrence_scores[s] for s in active_states_sorted])
    cov_vals = np.array([coverage[s]['hp_coverage'] for s in active_states_sorted])
    coverage_corr = {}
    if len(rec_vals) >= 5:
        rho, p = stats.spearmanr(rec_vals, cov_vals)
        coverage_corr = {'rho': float(rho), 'p': float(p), 'n': len(rec_vals)}

    inactive_states = [i for i in range(n_states) if recurrence_scores[i] == 0]
    inactive_activated = {}
    for state_id in inactive_states:
        hp_mean_fo = float(np.mean([hp_fo[rid][state_id] for rid in all_run_ids]))
        if hp_mean_fo > fo_threshold:
            inactive_activated[int(state_id)] = float(hp_mean_fo)

    result = {
        'state_coverage': coverage,
        'recurrence_vs_coverage_spearman': coverage_corr,
        'n_active_states_plotted': len(active_states_sorted),
        'inactive_states_activated_in_hp': inactive_activated,
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

        ax.set_xticks(range(len(all_run_ids)))
        ax.set_xticklabels([f"R{i+1}" for i in range(len(all_run_ids))], fontsize=7)
        ax.set_xlabel('HP Runs (chapters)')
        ax.set_ylabel('States (sorted by Friends recurrence, descending)')
        ax.set_title('A4: State Coverage Across HP Runs')

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'A4_state_coverage_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    return result


def analysis_a5_pca_diagnostic(pca_diagnostic, out_dir):
    """A5: PCA Transfer Diagnostic (overall + network-stratified R²)."""
    logger.info("A5: PCA Transfer Diagnostic")

    result = {
        'friends_r2': pca_diagnostic['friends_r2_n_pcs'],
        'hp_r2': pca_diagnostic['hp_r2_n_pcs'],
        'transfer_gap': pca_diagnostic['transfer_gap'],
        'flag_low_variance': pca_diagnostic['flag_low_variance'],
        'n_pcs': pca_diagnostic['n_pcs'],
        'note': ('HP uses word-by-word reading (RSVP), so visual cortex PCs carry '
                 'word-form but no scene-based stimulus-driven variance. Lower HP R² '
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
    labels = ['Friends\n(training)', 'HP\n(reading)']
    values = [pca_diagnostic['friends_r2_n_pcs'], pca_diagnostic['hp_r2_n_pcs']]
    colors = ['#4477AA', '#AA3377']

    bars = ax_overall.bar(labels, values, color=colors, alpha=0.8, width=0.5)
    ax_overall.set_ylabel(f'Variance Explained (R², {pca_diagnostic["n_pcs"]} PCs)')
    ax_overall.set_title('Overall R²')
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
        ax_net.set_ylabel(f'HP R² ({pca_diagnostic["n_pcs"]} PCs)')
        ax_net.set_title('Per-Network R² (HP Reading)')
        ax_net.set_ylim(0, 1.0)
        ax_net.axhline(y=0.70, color='red', linestyle='--', alpha=0.5)

        for bar, val in zip(bars_net, net_r2):
            ax_net.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle('A5: PCA Transfer Diagnostic (HP Reading)', fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'A5_pca_diagnostic.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    return result


# =========================================================================
# HP-specific analyses (B1, B2)
# =========================================================================

def analysis_b1_movie10_baseline(sub_id, parc, vt, recurrence_scores, n_states):
    """B1: 5-Subject Movie10 Baseline.

    Recompute Movie10 A1 (recurrence-FO correlation) for the same subject
    (to provide direct comparison with HP results). Also notes that sub-04
    is absent from HP.
    """
    logger.info("B1: Movie10 Baseline Comparison")

    # Load Movie10 FO if available
    movie_dir = os.path.join(SCRATCH_DIR, 'output', 'm10_04_decoded', parc, sub_id)
    if vt is not None:
        movie_dir = os.path.join(movie_dir, f'vt{vt}')
    movie_fo_path = os.path.join(movie_dir, 'fractional_occupancy.pkl')

    if not os.path.exists(movie_fo_path):
        logger.warning(f"Movie10 FO not found at {movie_fo_path} — skipping B1")
        return {'available': False, 'note': 'Movie10 data not available for this subject'}

    with open(movie_fo_path, 'rb') as f:
        movie_fo = pickle.load(f)

    fo_matrix = np.array(list(movie_fo.values()))
    mean_fo = np.mean(fo_matrix, axis=0)

    result = {
        'available': True,
        'n_movie_runs': len(movie_fo),
        'sub_04_note': 'sub-04 is absent from HP data. This baseline uses the same subject for direct comparison.',
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


def analysis_b2_bootstrap_reference(sub_id, parc, vt, hp_fo,
                                     recurrence_scores, n_states,
                                     n_bootstrap=1000, out_dir=None):
    """B2: Bootstrap Matched-Sample-Size Reference.

    Draw 7 random Movie10 runs (1000x), compute A1 statistics, build
    reference distribution. Compare HP effect sizes to this distribution.
    """
    logger.info("B2: Bootstrap Matched-Sample-Size Reference")

    # Load Movie10 FO
    movie_dir = os.path.join(SCRATCH_DIR, 'output', 'm10_04_decoded', parc, sub_id)
    if vt is not None:
        movie_dir = os.path.join(movie_dir, f'vt{vt}')
    movie_fo_path = os.path.join(movie_dir, 'fractional_occupancy.pkl')

    if not os.path.exists(movie_fo_path):
        logger.warning(f"Movie10 FO not found — skipping B2")
        return {'available': False}

    with open(movie_fo_path, 'rb') as f:
        movie_fo = pickle.load(f)

    movie_run_ids = list(movie_fo.keys())
    n_hp_runs = len(hp_fo)

    if len(movie_run_ids) < n_hp_runs:
        logger.warning(f"Movie10 has fewer runs ({len(movie_run_ids)}) than HP ({n_hp_runs}) — skipping B2")
        return {'available': False}

    active_mask = np.array([recurrence_scores[i] > 0 for i in range(n_states)])
    rec_active = recurrence_scores[active_mask]

    rng = np.random.default_rng(42)
    bootstrap_rho = []

    for _ in range(n_bootstrap):
        # Sample n_hp_runs Movie10 runs without replacement
        sample_ids = rng.choice(movie_run_ids, size=n_hp_runs, replace=False)
        sample_fo = {rid: movie_fo[rid] for rid in sample_ids}

        fo_matrix = np.array(list(sample_fo.values()))
        mean_fo = np.mean(fo_matrix, axis=0)

        # A1: Spearman rho
        fo_active = mean_fo[active_mask]
        if len(rec_active) >= 3:
            rho, _ = stats.spearmanr(rec_active, fo_active)
            bootstrap_rho.append(rho)

    # Compute HP effect sizes for comparison
    hp_fo_matrix = np.array(list(hp_fo.values()))
    hp_mean_fo = np.mean(hp_fo_matrix, axis=0)

    hp_fo_active = hp_mean_fo[active_mask]
    hp_rho = None
    if len(rec_active) >= 3:
        hp_rho, _ = stats.spearmanr(rec_active, hp_fo_active)

    result = {
        'available': True,
        'n_bootstrap': n_bootstrap,
        'n_subsample_runs': n_hp_runs,
        'n_movie_runs_total': len(movie_run_ids),
    }

    if bootstrap_rho:
        bootstrap_rho = np.array(bootstrap_rho)
        result['movie10_bootstrap_rho_mean'] = float(np.mean(bootstrap_rho))
        result['movie10_bootstrap_rho_ci95'] = [float(np.percentile(bootstrap_rho, 2.5)),
                                                  float(np.percentile(bootstrap_rho, 97.5))]
        if hp_rho is not None:
            result['hp_rho'] = float(hp_rho)
            result['hp_rho_percentile'] = float(np.mean(bootstrap_rho <= hp_rho) * 100)

    # --- Plot ---
    if out_dir and len(bootstrap_rho) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        ax.hist(bootstrap_rho, bins=40, alpha=0.6, color='#4477AA', label='Movie10 (7-run samples)')
        if hp_rho is not None:
            ax.axvline(hp_rho, color='#AA3377', linewidth=2, label=f'HP (ρ={hp_rho:.3f})')
        ax.set_xlabel('Spearman ρ')
        ax.set_ylabel('Count')
        ax.set_title('A1: Recurrence-FO Correlation')
        ax.legend(fontsize=8)

        fig.suptitle('B2: HP vs Movie10 Bootstrap Reference (matched n=7 runs)', fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'B2_bootstrap_reference.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    return result


def analysis_serial_dependence(hp_fo, hp_run_ids, n_states, out_dir):
    """Compute between-run FO autocorrelation for serial dependence diagnostic.

    HP runs are consecutive book chapters, not independent episodes.
    This reports how correlated FO vectors are between consecutive runs.
    """
    logger.info("Serial Dependence Diagnostic")

    # Flatten run IDs in order
    all_run_ids = []
    for stype in sorted(hp_run_ids.keys()):
        all_run_ids.extend(hp_run_ids[stype])
    ordered_runs = [rid for rid in all_run_ids if rid in hp_fo]

    if len(ordered_runs) < 3:
        return {'n_runs': len(ordered_runs), 'note': 'Too few runs for autocorrelation'}

    fo_matrix = np.array([hp_fo[rid] for rid in ordered_runs])  # (n_runs, n_states)

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

    result = {
        'n_runs': n_runs,
        'lag1_correlations': lag1_corrs,
        'mean_lag1_autocorrelation': mean_lag1,
        'n_effective_independent': float(round(n_effective, 1)),
        'note': ('HP runs are consecutive chapters with serial dependence. '
                 'n_effective estimates independent observations for p-value interpretation.'),
    }

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: lag-1 autocorrelations
    ax1.bar(range(len(lag1_corrs)), lag1_corrs, color='#4477AA', alpha=0.8)
    ax1.set_xlabel('Run pair (i, i+1)')
    ax1.set_ylabel('Pearson r (FO vectors)')
    ax1.set_title(f'Lag-1 FO Autocorrelation (mean = {mean_lag1:.3f})')
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
    ax2.set_title(f'Between-Run FO Correlation (n_eff = {n_effective:.1f})')

    fig.suptitle('Serial Dependence: HP Chapters', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'serial_dependence.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    return result


# =========================================================================
# Main
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Cross-stimulus validation: test Friends brain state generalization to HP word-by-word reading.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script/hp_05_cross_stimulus_validation.py --sub_id sub-01
  python script/hp_05_cross_stimulus_validation.py --sub_id sub-01 --parcellation atlas-4S456Parcels
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

    # HP decoded states + FO (vt-aware)
    hp_dir = os.path.join(SCRATCH_DIR, 'output', 'hp_04_decoded', parc, sub_id)
    if args.vt is not None:
        hp_dir = os.path.join(hp_dir, f'vt{args.vt}')
    hp_fo_path = os.path.join(hp_dir, 'fractional_occupancy.pkl')
    hp_ll_path = os.path.join(hp_dir, 'hp_ll_summary.json')

    # HP run IDs (vt-aware)
    proj_dir = os.path.join(SCRATCH_DIR, 'output', 'hp_03_projected', parc, sub_id)
    if args.vt is not None:
        proj_dir = os.path.join(proj_dir, f'vt{args.vt}')
    hp_run_ids_path = os.path.join(proj_dir, 'hp_run_ids.json')

    # run_id_map.json from hp_04 — maps long BIDS keys to short keys
    run_id_map_path = os.path.join(hp_dir, 'run_id_map.json')

    # PCA diagnostic
    pca_diag_path = os.path.join(proj_dir, 'pca_transfer_diagnostic.json')

    # Validate inputs
    required_files = {
        recurrence_summary_path: 'recurrence_summary.json (run 05a first)',
        recurrence_scores_path: 'recurrence_scores.npy (run 05a first)',
        hp_fo_path: 'HP FO (run hp_04 first)',
        hp_ll_path: 'HP LL summary (run hp_04 first)',
        hp_run_ids_path: 'hp_run_ids.json (run hp_03 first)',
        run_id_map_path: 'run_id_map.json (run hp_04 first)',
        pca_diag_path: 'pca_transfer_diagnostic.json (run hp_03 first)',
    }
    for path, label in required_files.items():
        if not os.path.exists(path):
            logger.error(f"Missing: {path} — {label}")
            sys.exit(1)

    # =========================================================================
    # Load data
    # =========================================================================

    with open(recurrence_summary_path, 'r') as f:
        recurrence_summary = json.load(f)

    recurrence_scores = np.load(recurrence_scores_path)
    n_states = len(recurrence_scores)

    with open(hp_fo_path, 'rb') as f:
        hp_fo = pickle.load(f)

    with open(hp_ll_path, 'r') as f:
        ll_summary = json.load(f)

    with open(hp_run_ids_path, 'r') as f:
        hp_run_ids_raw = json.load(f)

    with open(run_id_map_path, 'r') as f:
        run_id_map = json.load(f)

    # hp_run_ids.json uses long BIDS keys (from hp_03) but hp_fo.pkl uses
    # short keys (from hp_04's normalize_cross_stim_run_id). Remap.
    long_to_short = run_id_map['long_to_short']
    hp_run_ids = {}
    for stype, long_ids in hp_run_ids_raw.items():
        short_ids = []
        for lid in long_ids:
            short = long_to_short.get(lid)
            if short is None:
                logger.warning("run_id_map has no short key for %s — skipping", lid)
                continue
            short_ids.append(short)
        hp_run_ids[stype] = short_ids

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

    if not hp_fo:
        logger.error("No HP runs in fractional_occupancy.pkl — check hp_04 outputs")
        sys.exit(1)

    n_active = int(np.sum(recurrence_scores > 0))
    logger.info(f"Loaded data for {sub_id}: {n_states} states, {len(hp_fo)} HP runs")
    logger.info(f"Active states (recurrence > 0): {n_active}")

    # =========================================================================
    # Output directory
    # =========================================================================

    out_dir = os.path.join(SCRATCH_DIR, 'output', 'hp_05_cross_validation', parc, sub_id)
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
        'stimulus': 'harrypotter',
        'stimulus_modality': 'reading_only',
        'n_states': n_states,
        'n_hp_runs': len(hp_fo),
        'fo_threshold': fo_threshold,
        'exclude_sub_hrf': args.exclude_sub_hrf,
        'n_excluded_sub_hrf': len(excluded_sub_hrf),
        'excluded_sub_hrf_states': excluded_sub_hrf,
        'n_active_states': n_active,
    }

    summary['A1_recurrence_correlation'] = analysis_a1_recurrence_fo_correlation(
        hp_fo, recurrence_scores, n_states, out_dir)
    summary['A2_per_type'] = analysis_a2_per_type(
        hp_fo, hp_run_ids, recurrence_scores, n_states, out_dir)

    # Content-eligible-subset variants (project-wide 05e_a4 convention).
    # Mirrors A1/A2 but restricts both vectors to the subset of states flagged
    # as ``eligible_for_content_analysis`` in 05e_a4's state_flags.csv
    # (intersected with active states). The full-repertoire A1/A2 above are
    # preserved unchanged so downstream figures can show both panels.
    eligibility = load_content_eligibility(
        sub_id=sub_id, parcellation=parc, scratch_dir=SCRATCH_DIR, vt=args.vt,
    )
    eligible_variants = compute_eligible_recurrence_correlations(
        hp_fo, hp_run_ids, recurrence_scores, n_states, eligibility,
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
        hp_fo, hp_run_ids, recurrence_scores, n_states, fo_threshold, out_dir)
    summary['A5_pca_diagnostic'] = analysis_a5_pca_diagnostic(pca_diagnostic, out_dir)

    # HP-specific analyses
    summary['B1_movie10_baseline'] = analysis_b1_movie10_baseline(
        sub_id, parc, args.vt, recurrence_scores, n_states)
    summary['B2_bootstrap_reference'] = analysis_b2_bootstrap_reference(
        sub_id, parc, args.vt, hp_fo, recurrence_scores, n_states,
        n_bootstrap=args.n_bootstrap, out_dir=out_dir)

    # Serial dependence diagnostic (HP chapters are not independent)
    summary['serial_dependence'] = analysis_serial_dependence(
        hp_fo, hp_run_ids, n_states, out_dir)

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
    print(f"CROSS-STIMULUS VALIDATION SUMMARY (HP READING)")
    print(f"{'='*60}")
    print(f"Subject: {sub_id} | Parcellation: {parc}")
    print(f"HP runs: {len(hp_fo)} | States: {n_states}")
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
    print(f"  HP overall:    {a3['hp_overall_ll']:.4f}")
    print(f"  Baseline:      {a3['baseline_ll']:.4f}")
    print(f"  HP > base:     {a3['hp_above_baseline']}")
    print(f"")
    print(f"A4 - State Coverage:")
    cov_corr = a4.get('recurrence_vs_coverage_spearman', {})
    if cov_corr:
        print(f"  Recurrence-coverage Spearman rho: {cov_corr['rho']:.3f}, p={cov_corr['p']:.4f}")
    print(f"")
    print(f"A5 - PCA Diagnostic:")
    print(f"  Friends R²: {a5['friends_r2']:.4f}, HP R²: {a5['hp_r2']:.4f}")
    if a5['flag_low_variance']:
        print(f"  *** WARNING: HP R² < 0.70 ***")
    if a5.get('r2_by_network'):
        print(f"  Per-network HP R²:")
        for net in NETWORK_ORDER:
            if net in a5['r2_by_network']:
                print(f"    {net:15s}: {a5['r2_by_network'][net]['r2_n_pcs']:.4f}")
    print(f"")
    print(f"B1 - Movie10 Baseline:")
    if b1.get('available'):
        if b1.get('movie10_spearman_rho') is not None:
            print(f"  Movie10 Spearman ρ: {b1['movie10_spearman_rho']:.3f}")
    else:
        print(f"  Movie10 data not available")
    print(f"")
    print(f"B2 - Bootstrap Reference:")
    if b2.get('available'):
        if b2.get('hp_rho') is not None:
            print(f"  HP ρ: {b2['hp_rho']:.3f} (percentile: {b2.get('hp_rho_percentile', '?'):.1f}%)")
        if b2.get('movie10_bootstrap_rho_ci95'):
            ci = b2['movie10_bootstrap_rho_ci95']
            print(f"  Movie10 7-run ρ 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    else:
        print(f"  Movie10 data not available")
    print(f"")
    sd = summary['serial_dependence']
    print(f"Serial Dependence:")
    print(f"  Mean lag-1 autocorr: {sd.get('mean_lag1_autocorrelation', 'N/A')}")
    print(f"  Effective n:         {sd.get('n_effective_independent', 'N/A')}")
    print(f"{'='*60}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
