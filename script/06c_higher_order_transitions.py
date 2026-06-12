#!/usr/bin/env python3
"""
06c_higher_order_transitions.py - Model adequacy diagnostic for transition structure.

Tests how much sequential structure in the HMM's decoded state sequences exceeds
what the model's 1st-order transition matrix predicts.  This is a diagnostic of
model adequacy, not a competing model: we do not refit the HMM with higher-order
transitions.

Framing
-------
The HDP-HMM assumes P(z_t | z_{<t}) = P(z_t | z_{t-1}).  Script 06b showed that
multi-step chains occur far more often than this 1st-order assumption predicts
(Chain Excess Index up to 681x).  However, high CEI alone does not tell us *why*.

Two interpretations are equally consistent with high CEI:
  1. Genuine higher-order dynamics (the brain's latent process is not 1st-order;
     the HMM captures spatial states well but undermodels transitions).
  2. Insufficient state granularity (the HMM needs more sub-states to absorb
     context-dependent exit routes; with enough states the process would be
     1st-order again).

This script quantifies the magnitude and character of the non-Markov structure
without adjudicating between interpretations 1 and 2.

Analyses
--------
  1. Conditional entropy reduction:  ΔH = H(S_t|S_{t-1}) - H(S_t|S_{t-1},S_{t-2})
  2. Context-dependence test per trigram:  P(C|A,B) vs P(C|B, not from A)
  3. BIC Markov order comparison (order 1 vs restricted order 2)
  4. Hierarchical null for 4-grams (2nd-order Markov baseline)
  5. Hub/role classification + state-type cross-reference

Statistical design notes
------------------------
  - Analysis 1: Conditional entropy uses plugin estimator with Miller-Madow bias
    correction.  Singleton contexts (count < 2) are excluded from both the entropy
    sum and its normalizing denominator, so they do not bias H downward.  Bootstrap
    CIs are computed by resampling episodes; the CI is *not* a null-hypothesis test
    (there is no null distribution centered at zero), so we report only whether the
    CI excludes zero.

  - Analysis 2: Tests *all* well-observed (A,B) contexts - not just trigrams
    pre-selected as significant by 06b - to avoid selective-inference bias.  Uses a
    leave-one-out baseline: P(C|B, not from A) = (N(B->C) - N(A->B->C)) /
    (N(B->*) - N(A->B->*)), so the tested events do not contaminate the null.

  - Analysis 3: Both order-1 and restricted order-2 models are evaluated on the
    *same* observation set (trigrams) so that log-likelihoods and BIC are directly
    comparable.  The restricted order-2 parameter count includes the 1st-order
    fallback parameters used by sparse contexts.

  - Analysis 4: Tests *all* 4-grams with sufficient counts - not pre-selected by
    06b - with leave-one-out P(D|B,C) baseline, same design as Analysis 2.

Prerequisites
-------------
  - 04_combined_hdphmm.py (mode: select) completed.
  - 05a_recurrence_analysis.py completed.

Outputs
-------
  Saves to {SCRATCH_DIR}/output/06c_higher_order_transitions/{parcellation}/{sub_id}/
"""

import os
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import matplotlib.pyplot as plt

from dotenv import load_dotenv

# Setup paths and logger
sys.path.insert(0, str(Path(__file__).parent))
from utils.plot_style import (
    recurrence_color, make_recurrence_colorbar, apply_publication_style,
)
from utils.common import normalize_parcellation_name
from utils.stats import benjamini_hochberg

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


# ---------------------------------------------------------------------------
# Transition utilities (shared with 06b)
# ---------------------------------------------------------------------------
from utils.transition_utils import (
    collapse_to_state_changes,
    count_ngrams,
)


# ===================================================================
# Analysis 1: Conditional Entropy Reduction
# ===================================================================

def _plugin_conditional_entropy(ngram_counts, context_order):
    """Compute plugin estimate of H(S_t | context) with Miller-Madow correction.

    Parameters
    ----------
    ngram_counts : Counter
        Counts of (context_order+1)-grams.  The last element is the predicted
        state; the first *context_order* elements form the context.
    context_order : int
        Number of conditioning states (1 for H1, 2 for H2).

    Returns
    -------
    H : float
        Conditional entropy in bits.
    n_contexts : int
        Number of distinct contexts with data.
    n_total : int
        Total transitions used.
    """
    # Group by context
    context_counts = {}  # context_tuple -> Counter of successors
    for ngram, count in ngram_counts.items():
        ctx = ngram[:context_order]
        succ = ngram[context_order]
        if ctx not in context_counts:
            context_counts[ctx] = Counter()
        context_counts[ctx][succ] += count

    H = 0.0
    n_total = sum(ngram_counts.values())
    n_contexts_used = 0
    # Track the denominator for weighting: only count transitions from
    # non-singleton contexts, so dropped contexts don't bias H downward.
    n_used = 0

    for ctx, succ_counts in context_counts.items():
        n_ctx = sum(succ_counts.values())
        if n_ctx < 2:
            # Need at least 2 observations for meaningful entropy
            continue
        n_contexts_used += 1
        n_used += n_ctx
        # Plugin entropy for this context
        h_ctx = 0.0
        m_nonzero = 0  # number of nonzero bins (for Miller-Madow)
        for count in succ_counts.values():
            if count > 0:
                p = count / n_ctx
                h_ctx -= p * np.log2(p)
                m_nonzero += 1
        # Miller-Madow bias correction: add (m-1) / (2*N*ln2)
        if n_ctx > 1:
            h_ctx += (m_nonzero - 1) / (2 * n_ctx * np.log(2))
        # Weight by context frequency (among used contexts only)
        H += n_ctx * h_ctx  # accumulate weighted; divide by n_used below

    if n_used > 0:
        H /= n_used

    return H, n_contexts_used, n_used


def compute_conditional_entropy(change_sequences, n_bootstrap=1000, rng=None):
    """Analysis 1: Conditional entropy reduction with bootstrap CI.

    Returns
    -------
    results : dict
        h1, h2, delta_h, bootstrap_ci_95, n_contexts_order1, n_contexts_order2,
        n_total, delta_h_pct.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Count bigrams and trigrams from full data
    bigram_counts, _ = count_ngrams(change_sequences, 2)
    trigram_counts, _ = count_ngrams(change_sequences, 3)

    h1, n_ctx1, n_total = _plugin_conditional_entropy(bigram_counts, context_order=1)
    h2, n_ctx2, _ = _plugin_conditional_entropy(trigram_counts, context_order=2)
    delta_h = h1 - h2

    logger.info(f"  H(S_t|S_{{t-1}}) = {h1:.4f} bits  ({n_ctx1} contexts)")
    logger.info(f"  H(S_t|S_{{t-1}},S_{{t-2}}) = {h2:.4f} bits  ({n_ctx2} contexts)")
    logger.info(f"  ΔH = {delta_h:.4f} bits  ({100*delta_h/h1:.1f}% reduction)")

    # Bootstrap CI by resampling episodes
    run_ids = list(change_sequences.keys())
    n_runs = len(run_ids)
    boot_deltas = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        sample_ids = rng.choice(run_ids, size=n_runs, replace=True)
        boot_seqs = {f"boot_{i}": change_sequences[rid]
                     for i, rid in enumerate(sample_ids)}
        bi_counts, _ = count_ngrams(boot_seqs, 2)
        tri_counts, _ = count_ngrams(boot_seqs, 3)
        bh1, _, _ = _plugin_conditional_entropy(bi_counts, context_order=1)
        bh2, _, _ = _plugin_conditional_entropy(tri_counts, context_order=2)
        boot_deltas[b] = bh1 - bh2

    ci_lo, ci_hi = np.percentile(boot_deltas, [2.5, 97.5])
    boot_median = float(np.median(boot_deltas))
    ci_excludes_zero = bool(ci_lo > 0)

    logger.info(f"  Bootstrap median ΔH: {boot_median:.4f}")
    logger.info(f"  Bootstrap 95% CI for ΔH: [{ci_lo:.4f}, {ci_hi:.4f}]")
    logger.info(f"  CI excludes zero: {ci_excludes_zero}")

    return {
        'h1_bits': float(h1),
        'h2_bits': float(h2),
        'delta_h_bits': float(delta_h),
        'delta_h_pct': float(100 * delta_h / h1) if h1 > 0 else 0.0,
        'bootstrap_median_delta_h': boot_median,
        'bootstrap_ci_95': [float(ci_lo), float(ci_hi)],
        'ci_excludes_zero': ci_excludes_zero,
        'n_bootstrap': n_bootstrap,
        'n_contexts_order1': n_ctx1,
        'n_contexts_order2': n_ctx2,
        'n_total_transitions': n_total,
    }


def plot_conditional_entropy(results, out_dir):
    """Two-panel figure: H1 vs H2 bars + bootstrap ΔH distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: bar chart
    ax = axes[0]
    bars = ax.bar(['H(S_t | S_{t-1})', 'H(S_t | S_{t-1}, S_{t-2})'],
                  [results['h1_bits'], results['h2_bits']],
                  color=['#56B4E9', '#E69F00'], edgecolor='k', linewidth=0.5)
    ax.set_ylabel('Conditional entropy (bits)')
    ax.set_title('Transition uncertainty')
    # Annotate delta
    ax.annotate(f"ΔH = {results['delta_h_bits']:.3f} bits\n"
                f"({results['delta_h_pct']:.1f}% reduction)",
                xy=(0.5, 0.9), xycoords='axes fraction', ha='center',
                fontsize=9, bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    # Panel B: bootstrap CI
    ax = axes[1]
    ci = results['bootstrap_ci_95']
    # Show bootstrap median and CI as point + error bar
    boot_median = (ci[0] + ci[1]) / 2
    ax.barh(0, results['delta_h_bits'], color='#E69F00', edgecolor='k',
            linewidth=0.5, height=0.4, label='Full-data estimate')
    # CI as a horizontal line
    ax.plot(ci, [0, 0], 'k-', linewidth=2)
    ax.plot(ci, [0, 0], 'k|', markersize=10, linewidth=2)
    ax.axvline(0, color='grey', ls='--', lw=0.8)
    ax.set_xlabel('ΔH (bits)')
    ax.set_title('Information gain from 2nd-order context')
    ax.set_yticks([])
    boot_med = results.get('bootstrap_median_delta_h', (ci[0]+ci[1])/2)
    ci_note = ("CI excludes zero" if results.get('ci_excludes_zero', ci[0] > 0)
               else "CI includes zero")
    ax.annotate(f"Point estimate: {results['delta_h_bits']:.3f}\n"
                f"Bootstrap median: {boot_med:.3f}\n"
                f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]\n{ci_note}",
                xy=(0.95, 0.9), xycoords='axes fraction', ha='right',
                fontsize=8, bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'conditional_entropy.{ext}'),
                    dpi=300 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)


# ===================================================================
# Analysis 2: Context-Dependence Test per Trigram
# ===================================================================

def test_context_dependence(change_sequences, min_context_count=10):
    """Test whether the predecessor state modulates B's exit distribution.

    For each well-observed (A,B) context, test whether P(C|A,B) differs
    from P(C|B, not from A) - the leave-one-out baseline that excludes
    the tested context to avoid circularity.

    Tests *all* (A,B) contexts with >= min_context_count observations,
    not just those pre-selected by 06b, to avoid selective-inference bias.

    Parameters
    ----------
    change_sequences : dict
        run_id -> state-change sequence.
    min_context_count : int
        Minimum N(A->B->*) to test a context.

    Returns
    -------
    records : list[dict]
    """
    # Count all bigrams and trigrams
    bigram_counts, _ = count_ngrams(change_sequences, 2)
    trigram_counts, _ = count_ngrams(change_sequences, 3)

    # Build full successor distributions per (A,B) context
    ab_succ = {}  # (A, B) -> Counter of successors
    for (s1, s2, s3), count in trigram_counts.items():
        key = (s1, s2)
        if key not in ab_succ:
            ab_succ[key] = Counter()
        ab_succ[key][s3] += count

    # Build full successor distributions per B (for baseline)
    b_succ = {}  # B -> Counter of successors
    for (s1, s2), count in bigram_counts.items():
        if s1 not in b_succ:
            b_succ[s1] = Counter()
        b_succ[s1][s2] += count

    records = []
    for (A, B), succ_counts in ab_succ.items():
        n_ab_total = sum(succ_counts.values())
        if n_ab_total < min_context_count:
            continue

        # Leave-one-out baseline: P(C|B, not from A)
        # = (N(B->C) - N(A->B->C)) / (N(B->*) - N(A->B->*))
        if B not in b_succ:
            continue
        n_b_total = sum(b_succ[B].values())
        n_b_excl_total = n_b_total - n_ab_total
        if n_b_excl_total < 1:
            # Context (A,B) accounts for all exits from B; no baseline
            continue

        # Test each successor C that appears in this context
        for C, n_ab_c in succ_counts.items():
            if n_ab_c == 0:
                continue

            n_b_c = b_succ[B].get(C, 0)
            n_b_c_excl = n_b_c - n_ab_c
            p_c_given_b_excl = max(n_b_c_excl, 0) / n_b_excl_total

            if p_c_given_b_excl < 1e-15:
                continue

            p_c_given_ab = n_ab_c / n_ab_total

            # Binomial test: is P(C|A,B) > P(C|B, not from A)?
            result = sp_stats.binomtest(n_ab_c, n_ab_total, p_c_given_b_excl,
                                        alternative='greater')

            records.append({
                'chain': f'{A}->{B}->{C}',
                'A': A, 'B': B, 'C': C,
                'n_ab_total': n_ab_total,
                'n_ab_c': n_ab_c,
                'p_c_given_ab': float(p_c_given_ab),
                'p_c_given_b_excl': float(p_c_given_b_excl),
                'enrichment_ratio': float(p_c_given_ab / p_c_given_b_excl),
                'binomial_p': float(result.pvalue),
            })

    # FDR correction
    if records:
        p_vals = np.array([r['binomial_p'] for r in records])
        q_vals = benjamini_hochberg(p_vals)
        for r, q in zip(records, q_vals):
            r['fdr_q'] = float(q)

    n_sig = sum(1 for r in records if r.get('fdr_q', 1.0) < 0.05)
    logger.info(f"  Context-dependence: {len(records)} testable (A,B,C) triples, "
                f"{n_sig} significant (FDR<0.05)")

    return records


def plot_context_dependence(records, out_dir):
    """Scatter: P(C|A,B) vs P(C|B), colored by significance."""
    if not records:
        return

    df = pd.DataFrame(records)
    sig = df['fdr_q'] < 0.05

    fig, ax = plt.subplots(figsize=(7, 6))

    # Non-significant
    if (~sig).any():
        ax.scatter(df.loc[~sig, 'p_c_given_b_excl'],
                   df.loc[~sig, 'p_c_given_ab'],
                   alpha=0.4, color='grey', edgecolors='k', linewidths=0.3,
                   s=40, label=f'Not significant (n={int((~sig).sum())})', zorder=2)
    # Significant
    if sig.any():
        ax.scatter(df.loc[sig, 'p_c_given_b_excl'],
                   df.loc[sig, 'p_c_given_ab'],
                   alpha=0.8, color='#E69F00', edgecolors='k', linewidths=0.5,
                   s=60, label=f'Context-dependent (n={int(sig.sum())})', zorder=3)
        # Label top enrichment
        top = df.loc[sig].nlargest(5, 'enrichment_ratio')
        for _, r in top.iterrows():
            ax.annotate(r['chain'],
                        (r['p_c_given_b_excl'], r['p_c_given_ab']),
                        fontsize=7, alpha=0.9, xytext=(4, 4),
                        textcoords='offset points')

    # Diagonal
    lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5, zorder=1)

    ax.set_xlabel('P(C | B, not from A)  [leave-one-out baseline]')
    ax.set_ylabel('P(C | A, B)  [context-dependent probability]')
    ax.set_title('Context-dependence of trigram transitions')
    ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'context_dependence_scatter.{ext}'),
                    dpi=300 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)


# ===================================================================
# Analysis 3: BIC Markov Order Comparison
# ===================================================================

def compute_markov_order_bic(change_sequences, max_order=2,
                             min_context_count=10):
    """Fit order-1 and restricted order-2 Markov chains, compare via BIC.

    Both models are evaluated on the **same observation set** (trigrams) so
    that log-likelihoods and BIC values are directly comparable.

    - Order-1: scores each trigram (A,B,C) using P(C|B) only.
    - Order-2 (restricted VLMC): for (A,B) contexts with >= min_context_count
      observations, uses P(C|A,B); others fall back to P(C|B).

    Returns
    -------
    results : dict
        Per-order: {log_likelihood, n_params, n_obs, bic, aic}.
    """
    # Pool all change sequences (respecting episode boundaries)
    all_bigrams = Counter()
    all_trigrams = Counter()
    all_unigrams = Counter()

    for seq in change_sequences.values():
        for i in range(len(seq)):
            all_unigrams[seq[i]] += 1
        for i in range(len(seq) - 1):
            all_bigrams[(seq[i], seq[i + 1])] += 1
        for i in range(len(seq) - 2):
            all_trigrams[(seq[i], seq[i + 1], seq[i + 2])] += 1

    active_states = sorted(all_unigrams.keys())
    K = len(active_states)
    N_trigrams = sum(all_trigrams.values())

    results = {}

    # Pre-compute 1st-order row totals for P(C|B) = N(B,C) / N(B,*)
    row_totals = Counter()
    for (i, j), c in all_bigrams.items():
        row_totals[i] += c

    # --- Order 1 evaluated on trigrams ---
    # For each trigram (A,B,C), score as log P(C|B)
    ll_1 = 0.0
    for (a, b, c), count in all_trigrams.items():
        rt = row_totals.get(b, 0)
        n_bc = all_bigrams.get((b, c), 0)
        if rt > 0 and n_bc > 0:
            ll_1 += count * np.log(n_bc / rt)

    # Count free params: for each state B with outgoing transitions,
    # (n_distinct_successors - 1)
    n_params_1 = 0
    for i in active_states:
        n_succ = sum(1 for (s1, s2) in all_bigrams
                     if s1 == i and all_bigrams[(s1, s2)] > 0)
        if n_succ > 0:
            n_params_1 += n_succ - 1

    bic_1 = -2 * ll_1 + n_params_1 * np.log(N_trigrams)
    aic_1 = -2 * ll_1 + 2 * n_params_1

    results['order_1'] = {
        'log_likelihood': float(ll_1),
        'n_params': int(n_params_1),
        'n_obs': int(N_trigrams),
        'bic': float(bic_1),
        'aic': float(aic_1),
    }
    logger.info(f"  Order 1 (on trigrams): LL={ll_1:.1f}, k={n_params_1}, "
                f"BIC={bic_1:.1f}, AIC={aic_1:.1f}")

    # --- Restricted Order 2 (VLMC) evaluated on same trigrams ---
    context_ab_totals = Counter()
    for (a, b, c), count in all_trigrams.items():
        context_ab_totals[(a, b)] += count

    # Identify well-observed 2nd-order contexts
    rich_contexts = {ctx for ctx, n in context_ab_totals.items()
                     if n >= min_context_count}

    ll_2 = 0.0
    n_params_2_extra = 0  # additional params beyond order-1

    # Build successor distributions for rich contexts
    rich_succ = {}
    for (a, b, c), count in all_trigrams.items():
        if (a, b) in rich_contexts:
            if (a, b) not in rich_succ:
                rich_succ[(a, b)] = Counter()
            rich_succ[(a, b)][c] += count

    for ctx, succ_counts in rich_succ.items():
        ctx_total = sum(succ_counts.values())
        for s3, count in succ_counts.items():
            if count > 0:
                ll_2 += count * np.log(count / ctx_total)
        n_succ = sum(1 for c in succ_counts.values() if c > 0)
        if n_succ > 0:
            n_params_2_extra += n_succ - 1

    # For sparse contexts: fall back to 1st-order P(C|B)
    for (a, b, c), count in all_trigrams.items():
        if (a, b) in rich_contexts:
            continue
        rt = row_totals.get(b, 0)
        n_bc = all_bigrams.get((b, c), 0)
        if rt > 0 and n_bc > 0:
            ll_2 += count * np.log(n_bc / rt)

    # Total params = order-1 params (used for fallback) + extra 2nd-order params
    n_params_2 = n_params_1 + n_params_2_extra

    bic_2 = -2 * ll_2 + n_params_2 * np.log(N_trigrams)
    aic_2 = -2 * ll_2 + 2 * n_params_2

    results['order_2_restricted'] = {
        'log_likelihood': float(ll_2),
        'n_params': int(n_params_2),
        'n_params_extra': int(n_params_2_extra),
        'n_params_fallback': int(n_params_1),
        'n_obs': int(N_trigrams),
        'n_rich_contexts': len(rich_contexts),
        'n_total_contexts': len(context_ab_totals),
        'bic': float(bic_2),
        'aic': float(aic_2),
        'note': (f'Restricted VLMC: {len(rich_contexts)}/{len(context_ab_totals)} '
                 f'contexts use 2nd-order (>= {min_context_count} obs); '
                 f'remainder use 1st-order fallback. '
                 f'Params = {n_params_1} (fallback) + {n_params_2_extra} (2nd-order).'),
    }
    logger.info(f"  Order 2 (restricted): LL={ll_2:.1f}, k={n_params_2}, "
                f"BIC={bic_2:.1f}, AIC={aic_2:.1f}")
    logger.info(f"    Rich contexts: {len(rich_contexts)}/{len(context_ab_totals)}, "
                f"extra params: {n_params_2_extra}")

    # --- Summary ---
    preferred = 'order_1' if bic_1 < bic_2 else 'order_2_restricted'
    results['preferred_by_bic'] = preferred
    delta_bic = bic_1 - bic_2
    results['delta_bic_1_minus_2'] = float(delta_bic)
    logger.info(f"  ΔBIC (order1 - order2): {delta_bic:.1f}  "
                f"→ prefer {preferred}")

    return results


def plot_markov_order(results, out_dir):
    """Bar chart of BIC by order."""
    labels = ['Order 1', 'Order 2\n(restricted)']
    bics = [results['order_1']['bic'], results['order_2_restricted']['bic']]

    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ['#56B4E9', '#E69F00']
    bars = ax.bar(labels, bics, color=colors, edgecolor='k', linewidth=0.5)

    # Highlight preferred
    pref_idx = 0 if results['preferred_by_bic'] == 'order_1' else 1
    bars[pref_idx].set_edgecolor('red')
    bars[pref_idx].set_linewidth(2)

    ax.set_ylabel('BIC (lower is better)')
    ax.set_title('Markov order comparison')
    ax.annotate(f"ΔBIC = {results['delta_bic_1_minus_2']:.0f}",
                xy=(0.5, 0.95), xycoords='axes fraction', ha='center',
                fontsize=10, bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'markov_order_comparison.{ext}'),
                    dpi=300 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)


# ===================================================================
# Analysis 4: Hierarchical Null for 4-grams
# ===================================================================

def compute_hierarchical_null_4grams(change_sequences, min_count=3,
                                     min_context_count=5):
    """Test 4-grams against a 2nd-order Markov null.

    Tests *all* 4-grams with >= min_count observations (not pre-selected
    from 06b) to avoid selective-inference bias.  Uses leave-one-out
    baseline: P(D|B,C) is estimated excluding the A->B->C->D events
    being tested.

    For each 4-gram A->B->C->D:
      E_2 = N(A->B->C) * P_loo(D|B,C)
      where P_loo(D|B,C) = (N(B->C->D) - N(A->B->C->D)) / (N(B->C->*) - N(A->B->C->*))
      CEI_2 = (O - E_2) / E_2

    Parameters
    ----------
    change_sequences : dict
    min_count : int
        Minimum observed count for a 4-gram to be tested.
    min_context_count : int
        Minimum N(B->C->*) (after exclusion) to estimate P_loo(D|B,C).

    Returns
    -------
    records : list[dict]
    """
    # Count trigrams and 4-grams
    trigram_counts, _ = count_ngrams(change_sequences, 3)
    fourgram_counts, _ = count_ngrams(change_sequences, 4)

    # Build successor distributions: (B,C) -> Counter of D
    bc_successor_counts = {}
    for (s1, s2, s3), count in trigram_counts.items():
        key = (s1, s2)
        if key not in bc_successor_counts:
            bc_successor_counts[key] = Counter()
        bc_successor_counts[key][s3] += count

    # Build (A,B,C) -> Counter of D from 4-grams
    abc_successor_counts = {}
    for (s1, s2, s3, s4), count in fourgram_counts.items():
        key = (s1, s2, s3)
        if key not in abc_successor_counts:
            abc_successor_counts[key] = Counter()
        abc_successor_counts[key][s4] += count

    records = []
    for (A, B, C, D), obs in fourgram_counts.items():
        if obs < min_count:
            continue

        # N(A->B->C->*) = total successors of this trigram context
        abc_key = (A, B, C)
        n_abc_total = sum(abc_successor_counts.get(abc_key, {}).values())
        if n_abc_total == 0:
            continue

        # Leave-one-out baseline: P(D|B,C) excluding A->B->C->* events
        bc_key = (B, C)
        if bc_key not in bc_successor_counts:
            continue
        n_bc_total = sum(bc_successor_counts[bc_key].values())
        n_bc_excl_total = n_bc_total - n_abc_total
        if n_bc_excl_total < min_context_count:
            continue

        n_bcd = bc_successor_counts[bc_key].get(D, 0)
        n_bcd_excl = n_bcd - obs
        if n_bcd_excl < 0:
            n_bcd_excl = 0
        p_d_given_bc_loo = n_bcd_excl / n_bc_excl_total

        if p_d_given_bc_loo < 1e-15:
            continue

        # 2nd-order expected
        expected_2 = n_abc_total * p_d_given_bc_loo
        if expected_2 < 1e-10:
            continue

        cei_2 = (obs - expected_2) / expected_2

        # Normal approximation requires expected >= 5 for reliability
        if expected_2 < 5:
            z_2 = np.nan
            p_val_2 = np.nan
        else:
            z_2 = (obs - expected_2) / np.sqrt(expected_2)
            p_val_2 = 2.0 * sp_stats.norm.sf(abs(z_2))

        records.append({
            'chain': f'{A}->{B}->{C}->{D}',
            'observed': int(obs),
            'expected_order2': float(expected_2),
            'cei_order2': float(cei_2),
            'z_order2': float(z_2),
            'p_value_order2': float(p_val_2),
            'n_abc_total': int(n_abc_total),
            'p_d_given_bc_loo': float(p_d_given_bc_loo),
            'n_bc_excl_total': int(n_bc_excl_total),
        })

    # FDR correction on 2nd-order p-values
    if records:
        p_vals = np.array([r['p_value_order2'] for r in records])
        q_vals = benjamini_hochberg(p_vals)
        for r, q in zip(records, q_vals):
            r['fdr_q_order2'] = float(q)

    n_tested = len(records)
    n_survive = sum(1 for r in records
                    if r.get('fdr_q_order2', 1.0) < 0.05 and r['cei_order2'] > 0)
    n_absorbed = n_tested - n_survive
    logger.info(f"  Hierarchical null: {n_tested} 4-grams testable, "
                f"{n_survive} survive 2nd-order null, "
                f"{n_absorbed} absorbed by 2nd-order model")

    return records


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Model adequacy diagnostic for transition structure.")
    parser.add_argument('--sub_id', type=str, required=True)
    parser.add_argument('--parcellation', type=str, default='atlas-4S156Parcels')
    parser.add_argument('--min_dwell_tr', type=int, default=2)
    parser.add_argument('--n_bootstrap', type=int, default=1000)
    parser.add_argument('--min_context_count', type=int, default=10)
    parser.add_argument('--vt', type=str, default=None,
                        help="Variance threshold subdirectory (e.g., 0.95)")
    args = parser.parse_args()

    parc = normalize_parcellation_name(args.parcellation)
    sub_id = args.sub_id

    logger.info("==============================================")
    logger.info("06c - Higher-Order Transition Structure Diagnostic")
    logger.info("==============================================")
    logger.info(f"Subject: {sub_id}, Parcellation: {parc}")

    # --- Paths ---
    vt_suffix = f'vt{args.vt}' if args.vt else None

    hmm_base = os.path.join(SCRATCH_DIR, 'output', '04_combined_hdphmm',
                            parc, sub_id, 'final')
    if vt_suffix:
        hmm_base = os.path.join(hmm_base, vt_suffix)
    decoded_path = os.path.join(hmm_base, 'decoded_states.pkl')

    recur_base = os.path.join(SCRATCH_DIR, 'output', '05a_recurrence_analysis',
                              parc, sub_id)
    if vt_suffix:
        recur_base = os.path.join(recur_base, vt_suffix)
    summary_path = os.path.join(recur_base, 'recurrence_summary.json')

    out_dir = os.path.join(SCRATCH_DIR, 'output', '06c_higher_order_transitions',
                           parc, sub_id)
    if vt_suffix:
        out_dir = os.path.join(out_dir, vt_suffix)
    os.makedirs(out_dir, exist_ok=True)

    # --- Validate inputs ---
    for fpath, label in [(decoded_path, 'decoded states'),
                         (summary_path, 'recurrence summary')]:
        if not os.path.exists(fpath):
            logger.error(f"Missing {label}: {fpath}")
            sys.exit(1)

    # --- Load inputs ---
    logger.info("Loading inputs...")
    with open(decoded_path, 'rb') as f:
        decoded_states = pickle.load(f)

    with open(summary_path, 'r') as f:
        recurrence_summary = json.load(f)

    n_episodes = len(decoded_states)
    logger.info(f"Loaded {n_episodes} episodes.")

    # --- Preprocessing ---
    logger.info(f"Collapsing to state-change sequences "
                f"(min_dwell_tr={args.min_dwell_tr})...")
    change_sequences = {}
    for run_id, seq in decoded_states.items():
        change_sequences[run_id] = collapse_to_state_changes(
            seq, min_dwell_tr=args.min_dwell_tr)

    total_changes = sum(len(s) for s in change_sequences.values())
    logger.info(f"Total state tokens in change sequences: {total_changes}")

    summary = {}

    # =================================================================
    # Analysis 1: Conditional Entropy Reduction
    # =================================================================
    logger.info("--- Analysis 1: Conditional Entropy Reduction ---")
    entropy_results = compute_conditional_entropy(
        change_sequences, n_bootstrap=args.n_bootstrap)
    summary['conditional_entropy'] = entropy_results

    with open(os.path.join(out_dir, 'conditional_entropy.json'), 'w') as f:
        json.dump(entropy_results, f, indent=2)
    plot_conditional_entropy(entropy_results, out_dir)

    # =================================================================
    # Analysis 2: Context-Dependence Test per Trigram
    # =================================================================
    logger.info("--- Analysis 2: Context-Dependence per Trigram ---")
    ctx_records = test_context_dependence(
        change_sequences, min_context_count=args.min_context_count)
    df_ctx = pd.DataFrame(ctx_records)
    if not df_ctx.empty:
        df_ctx = df_ctx.sort_values('enrichment_ratio', ascending=False)
    df_ctx.to_csv(os.path.join(out_dir, 'context_dependence.csv'), index=False)
    plot_context_dependence(ctx_records, out_dir)
    n_sig = sum(1 for r in ctx_records if r.get('fdr_q', 1.0) < 0.05)
    summary['context_dependence'] = {
        'n_testable': len(ctx_records),
        'n_significant': n_sig,
        'fraction_significant': n_sig / len(ctx_records) if ctx_records else 0,
    }

    # =================================================================
    # Analysis 3: BIC Markov Order Comparison
    # =================================================================
    logger.info("--- Analysis 3: BIC Markov Order Comparison ---")
    bic_results = compute_markov_order_bic(
        change_sequences, min_context_count=args.min_context_count)
    summary['markov_order'] = bic_results

    with open(os.path.join(out_dir, 'markov_order_comparison.json'), 'w') as f:
        json.dump(bic_results, f, indent=2)
    plot_markov_order(bic_results, out_dir)

    # =================================================================
    # Analysis 4: Hierarchical Null for 4-grams
    # =================================================================
    logger.info("--- Analysis 4: Hierarchical Null for 4-grams ---")
    hier_records = compute_hierarchical_null_4grams(
        change_sequences, min_count=3,
        min_context_count=args.min_context_count)
    df_hier = pd.DataFrame(hier_records)
    if not df_hier.empty:
        df_hier = df_hier.sort_values('cei_order2', ascending=False)
    df_hier.to_csv(os.path.join(out_dir, 'hierarchical_null_4grams.csv'),
                   index=False)
    n_survive = sum(1 for r in hier_records
                    if r.get('fdr_q_order2', 1.0) < 0.05 and r['cei_order2'] > 0)
    summary['hierarchical_null_4grams'] = {
        'n_testable': len(hier_records),
        'n_survive_order2_null': n_survive,
        'n_absorbed_by_order2': len(hier_records) - n_survive,
    }

    # =================================================================
    # Save summary
    # =================================================================
    with open(os.path.join(out_dir, 'higher_order_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"All results saved to {out_dir}")
    logger.info("==============================================")
    logger.info("06c - Complete")
    logger.info("==============================================")


if __name__ == '__main__':
    main()
