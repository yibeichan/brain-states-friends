#!/usr/bin/env python3
"""
06d_preserved_chains.py - Identify state transition chains preserved across episodes.

For each n-gram order (bigrams, trigrams), computes a Chain Preservation Score
(CPS) - the fraction of episodes containing the chain - and tests whether CPS
exceeds what a first-order Markov process would produce via simulation.

Statistical framework
---------------------
  - Null model: Per-episode Markov-1 surrogate sequences. For each episode,
    simulate a state-change sequence of the same length from P_change, the
    state-change transition matrix derived from model.transmat_.  This jointly
    tests enrichment (chain over-representation given transition probabilities)
    and preservation (cross-episode consistency).

  - Test statistic: CPS(g) = (# episodes where g appears >= 1) / total_episodes.
    Binary presence eliminates within-episode dependence from overlapping n-grams.

  - P-value (one-tailed, simulation-based):
        p(g) = (#{sims where CPS_null >= CPS_obs} + 1) / (N_sim + 1)
    Uses Phipson & Smyth (2010) finite-sampling correction to avoid zero p-values.

  - FDR correction: Benjamini-Hochberg across all n-grams passing the
    min_episodes pre-filter.

  - Season-specificity test (for significant chains): Permutes season labels,
    computes season-specificity index (max - min per-season CPS).  Parallels
    05a's season-specificity analysis for individual states.

Prerequisites
-------------
  - 04_combined_hdphmm.py (mode: select) completed.
  - 05a_recurrence_analysis.py completed.

Outputs
-------
  Saves to {SCRATCH_DIR}/output/06d_preserved_chains/{parcellation}/{sub_id}/
"""

import os
import re
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from dotenv import load_dotenv

# Setup paths and logger
sys.path.insert(0, str(Path(__file__).parent))
from utils.plot_style import (
    recurrence_color, make_recurrence_colorbar, apply_publication_style,
    NETWORK_COLORS, NETWORK_ORDER,
    load_parcel_networks, compute_dominant_networks,
)
from utils.common import normalize_parcellation_name
from utils.stats import benjamini_hochberg
from utils.transition_utils import (
    collapse_to_state_changes,
    compute_state_change_transmat,
    count_ngrams,
    simulate_markov_change_seq,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_season(run_id):
    """Extract integer season number from run_id like 'task-s01e03a'."""
    m = re.search(r's(\d+)', run_id)
    return int(m.group(1)) if m else 0


def _chain_label(ngram):
    """Format n-gram tuple as arrow-delimited string."""
    return '→'.join(str(s) for s in ngram)


# ---------------------------------------------------------------------------
# Core analysis: Chain Preservation Score with Markov-1 surrogates
# ---------------------------------------------------------------------------

def compute_chain_preservation(change_sequences, P_change, order,
                               min_episodes=5, n_simulations=1000,
                               seed=42):
    """Identify n-grams preserved across episodes beyond Markov-1 expectation.

    Parameters
    ----------
    change_sequences : dict[str, list[int]]
        run_id → state-change sequence.
    P_change : np.ndarray, shape (K, K)
        State-change transition matrix (diagonal = 0).
    order : int
        N-gram order (2 = bigrams, 3 = trigrams).
    min_episodes : int
        Pre-filter: only test n-grams present in >= this many episodes.
    n_simulations : int
        Number of Markov-1 surrogate simulation rounds.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    results_df : pd.DataFrame
        One row per tested n-gram with columns: chain, order, cps_obs,
        total_count, n_episodes_present, cps_null_mean, cps_null_std,
        p_value, fdr_q.
    """
    rng = np.random.default_rng(seed)
    n_episodes = len(change_sequences)
    run_ids = list(change_sequences.keys())

    # --- Observed n-grams ---
    total_counts, per_episode_counts = count_ngrams(change_sequences, order)

    # Per-n-gram: set of episodes where it appears
    ngram_episodes = {}
    for run_id, ep_counter in per_episode_counts.items():
        for ng in ep_counter:
            ngram_episodes.setdefault(ng, set()).add(run_id)

    # Pre-filter by minimum episode presence
    candidates = {ng for ng, eps in ngram_episodes.items()
                  if len(eps) >= min_episodes}

    if not candidates:
        logger.warning(f"Order {order}: no n-grams present in >= {min_episodes} episodes")
        return pd.DataFrame()

    logger.info(f"Order {order}: {len(candidates)} n-grams pass min_episodes={min_episodes} "
                f"filter (out of {len(total_counts)} total)")

    # Observed CPS
    cps_obs = {ng: len(ngram_episodes[ng]) / n_episodes for ng in candidates}

    # --- Null distribution via Markov-1 surrogates ---
    # Pre-compute episode metadata (lengths and starting states)
    ep_meta = []
    for run_id in run_ids:
        seq = change_sequences[run_id]
        ep_meta.append((run_id, len(seq), seq[0] if seq else 0))

    null_cps_counts = {ng: 0 for ng in candidates}
    null_cps_sum = {ng: 0.0 for ng in candidates}
    null_cps_sumsq = {ng: 0.0 for ng in candidates}

    for sim_idx in range(n_simulations):
        if sim_idx > 0 and sim_idx % 200 == 0:
            logger.info(f"  Order {order}: simulation {sim_idx}/{n_simulations}")

        # Generate one surrogate per episode
        sim_sequences = {}
        for run_id, seq_len, start_state in ep_meta:
            if seq_len < order:
                sim_sequences[run_id] = []
                continue
            sim_sequences[run_id] = simulate_markov_change_seq(
                seq_len, P_change, start_state, rng)

        # Count n-grams in surrogates
        _, sim_per_ep = count_ngrams(sim_sequences, order)

        # Compute CPS for this simulation - count how many episodes contain each
        sim_ep_counts = {ng: 0 for ng in candidates}
        for ep_counter in sim_per_ep.values():
            for ng in candidates:
                if ep_counter.get(ng, 0) > 0:
                    sim_ep_counts[ng] += 1

        for ng in candidates:
            sim_cps = sim_ep_counts[ng] / n_episodes
            null_cps_sum[ng] += sim_cps
            null_cps_sumsq[ng] += sim_cps * sim_cps
            if sim_cps >= cps_obs[ng]:
                null_cps_counts[ng] += 1

    # --- P-values with finite-sampling correction ---
    p_values = {ng: (null_cps_counts[ng] + 1) / (n_simulations + 1)
                for ng in candidates}

    # --- FDR correction ---
    ng_list = sorted(candidates)
    raw_p = np.array([p_values[ng] for ng in ng_list])
    fdr_q = benjamini_hochberg(raw_p)

    # --- Assemble DataFrame ---
    rows = []
    for i, ng in enumerate(ng_list):
        null_mean = null_cps_sum[ng] / n_simulations
        null_var = null_cps_sumsq[ng] / n_simulations - null_mean ** 2
        null_std = np.sqrt(max(null_var, 0.0))
        rows.append({
            'chain': _chain_label(ng),
            'chain_tuple': ng,
            'order': order,
            'cps_obs': cps_obs[ng],
            'total_count': total_counts[ng],
            'n_episodes_present': len(ngram_episodes[ng]),
            'cps_null_mean': null_mean,
            'cps_null_std': null_std,
            'p_value': p_values[ng],
            'fdr_q': fdr_q[i],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('p_value').reset_index(drop=True)

    n_sig = int((df['fdr_q'] < 0.05).sum())
    logger.info(f"Order {order}: {n_sig}/{len(df)} n-grams significant (FDR < 0.05)")

    return df


# ---------------------------------------------------------------------------
# Season-specificity test for preserved chains
# ---------------------------------------------------------------------------

def test_season_specificity(change_sequences, significant_chains, order,
                            n_permutations=5000, seed=42):
    """Test whether preserved chains are season-specific or season-invariant.

    For each significant chain, computes a Season Specificity Index (SSI) =
    max(per-season CPS) - min(per-season CPS).  Significance is assessed by
    permuting season labels across episodes.

    Parameters
    ----------
    change_sequences : dict[str, list[int]]
        run_id → state-change sequence.
    significant_chains : list[tuple]
        N-gram tuples that passed FDR < 0.05 in preservation test.
    order : int
        N-gram order.
    n_permutations : int
        Number of season-label permutations.
    seed : int
        Random seed.

    Returns
    -------
    results : list[dict]
        Per-chain: chain, ssi_obs, perm_p, per_season_cps.
    """
    if not significant_chains:
        return []

    rng = np.random.default_rng(seed)
    run_ids = list(change_sequences.keys())
    season_labels = np.array([_get_season(r) for r in run_ids])
    seasons = sorted(set(season_labels))
    n_episodes = len(run_ids)

    # Count observed per-episode n-grams
    _, per_episode_counts = count_ngrams(change_sequences, order)

    def _per_season_cps(chains, labels):
        """Compute per-season CPS for a set of chains given season labels."""
        season_cps = {ng: {} for ng in chains}
        for s in seasons:
            ep_mask = [run_ids[i] for i in range(n_episodes) if labels[i] == s]
            n_season = len(ep_mask)
            if n_season == 0:
                for ng in chains:
                    season_cps[ng][s] = np.nan
                continue
            for ng in chains:
                present = sum(1 for rid in ep_mask
                              if per_episode_counts[rid].get(ng, 0) > 0)
                season_cps[ng][s] = present / n_season
        return season_cps

    def _ssi(per_season):
        """Season Specificity Index: max - min (ignoring NaN)."""
        vals = [v for v in per_season.values() if not np.isnan(v)]
        return max(vals) - min(vals) if len(vals) >= 2 else 0.0

    # Observed
    obs_season_cps = _per_season_cps(significant_chains, season_labels)
    obs_ssi = {ng: _ssi(obs_season_cps[ng]) for ng in significant_chains}

    # Permutation null
    perm_counts = {ng: 0 for ng in significant_chains}
    for p in range(n_permutations):
        if p > 0 and p % 1000 == 0:
            logger.info(f"  Season-specificity permutation {p}/{n_permutations}")

        shuffled = rng.permutation(season_labels)
        perm_season_cps = _per_season_cps(significant_chains, shuffled)
        for ng in significant_chains:
            if _ssi(perm_season_cps[ng]) >= obs_ssi[ng]:
                perm_counts[ng] += 1

    results = []
    for ng in significant_chains:
        perm_p = (perm_counts[ng] + 1) / (n_permutations + 1)
        results.append({
            'chain': _chain_label(ng),
            'chain_tuple': ng,
            'ssi_obs': obs_ssi[ng],
            'perm_p': perm_p,
            'per_season_cps': {int(s): float(v) for s, v in obs_season_cps[ng].items()},
        })

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_preservation_scatter(df, order, out_dir):
    """Scatter: CPS_obs vs p-value, colored by significance."""
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    sig = df['fdr_q'] < 0.05
    ax.scatter(df.loc[~sig, 'cps_obs'], -np.log10(df.loc[~sig, 'p_value']),
               c='gray', alpha=0.4, s=25, label='Not significant')
    ax.scatter(df.loc[sig, 'cps_obs'], -np.log10(df.loc[sig, 'p_value']),
               c='#E74C3C', alpha=0.8, s=40, edgecolors='black', linewidth=0.5,
               label=f'FDR < 0.05 (n={sig.sum()})')

    # Label top hits
    top = df.loc[sig].nsmallest(min(8, sig.sum()), 'p_value')
    for _, row in top.iterrows():
        ax.annotate(row['chain'], (row['cps_obs'], -np.log10(row['p_value'])),
                    fontsize=6, alpha=0.7, ha='left',
                    xytext=(4, 2), textcoords='offset points')

    ax.axhline(-np.log10(0.05), color='gray', linestyle='--', alpha=0.5,
               label='p = 0.05')
    ax.set_xlabel('Chain Preservation Score (CPS)')
    ax.set_ylabel('−log₁₀(p)')
    ax.set_title(f'Preserved {order}-grams (Markov-1 surrogate null)')
    ax.legend(fontsize=8)

    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'fig_preservation_{order}grams.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_top_preserved(df, order, out_dir, top_n=20):
    """Horizontal bar chart of top preserved chains by CPS."""
    sig_df = df[df['fdr_q'] < 0.05].nlargest(top_n, 'cps_obs')
    if sig_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(sig_df))))
    y_pos = np.arange(len(sig_df))

    ax.barh(y_pos, sig_df['cps_obs'].values, color='#3498DB', alpha=0.8,
            edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sig_df['chain'].values, fontsize=7)
    ax.set_xlabel('Chain Preservation Score')
    ax.set_title(f'Top preserved {order}-grams (FDR < 0.05)')
    ax.invert_yaxis()

    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'fig_top_preserved_{order}grams.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_preservation_vs_recurrence(df, recurrence_scores, order, out_dir):
    """Scatter: CPS of chain vs mean recurrence of constituent states."""
    if df.empty:
        return

    mean_rec = []
    for ng in df['chain_tuple']:
        mean_rec.append(np.mean([recurrence_scores[s] for s in ng
                                 if s < len(recurrence_scores)]))
    df = df.copy()
    df['mean_constituent_recurrence'] = mean_rec

    fig, ax = plt.subplots(figsize=(6, 5))
    sig = df['fdr_q'] < 0.05
    ax.scatter(df.loc[~sig, 'mean_constituent_recurrence'],
               df.loc[~sig, 'cps_obs'],
               c='gray', alpha=0.3, s=20, label='Not significant')
    ax.scatter(df.loc[sig, 'mean_constituent_recurrence'],
               df.loc[sig, 'cps_obs'],
               c='#E74C3C', alpha=0.7, s=35, edgecolors='black', linewidth=0.5,
               label=f'FDR < 0.05 (n={sig.sum()})')

    ax.set_xlabel('Mean constituent state recurrence')
    ax.set_ylabel('Chain Preservation Score')
    ax.set_title(f'{order}-gram preservation vs state recurrence')
    ax.legend(fontsize=8)

    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'fig_preservation_vs_recurrence_{order}grams.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Identify state transition chains preserved across episodes.")
    parser.add_argument('--sub_id', type=str, required=True)
    parser.add_argument('--parcellation', type=str, default='atlas-4S156Parcels')
    parser.add_argument('--min_dwell_tr', type=int, default=2)
    parser.add_argument('--min_episodes', type=int, default=5,
                        help="Pre-filter: only test n-grams in >= this many episodes")
    parser.add_argument('--n_simulations', type=int, default=1000,
                        help="Number of Markov-1 surrogate rounds")
    parser.add_argument('--n_season_perms', type=int, default=5000,
                        help="Number of season-label permutations for specificity test")
    parser.add_argument('--max_order', type=int, default=3,
                        help="Maximum n-gram order to test (2=bigrams, 3=trigrams)")
    parser.add_argument('--vt', type=str, default=None,
                        help="Variance threshold subdirectory (e.g., 0.95)")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    parc = normalize_parcellation_name(args.parcellation)
    sub_id = args.sub_id

    logger.info("==============================================")
    logger.info("06d - Preserved Transition Chains")
    logger.info("==============================================")
    logger.info(f"Subject: {sub_id}, Parcellation: {parc}")
    logger.info(f"N-gram orders: 2–{args.max_order}, "
                f"N_sim={args.n_simulations}, min_episodes={args.min_episodes}")

    # --- Paths ---
    vt_suffix = f'vt{args.vt}' if args.vt else None

    hmm_base = os.path.join(SCRATCH_DIR, 'output', '04_combined_hdphmm',
                            parc, sub_id, 'final')
    if vt_suffix:
        hmm_base = os.path.join(hmm_base, vt_suffix)

    decoded_path = os.path.join(hmm_base, 'decoded_states.pkl')
    model_path = os.path.join(hmm_base, 'best_model.pkl')

    recur_base = os.path.join(SCRATCH_DIR, 'output', '05a_recurrence_analysis',
                              parc, sub_id)
    if vt_suffix:
        recur_base = os.path.join(recur_base, vt_suffix)
    summary_path = os.path.join(recur_base, 'recurrence_summary.json')

    out_dir = os.path.join(SCRATCH_DIR, 'output', '06d_preserved_chains',
                           parc, sub_id)
    if vt_suffix:
        out_dir = os.path.join(out_dir, vt_suffix)
    os.makedirs(out_dir, exist_ok=True)

    # --- Validate inputs ---
    for fpath, label in [(decoded_path, 'decoded states'),
                         (model_path, 'best model'),
                         (summary_path, 'recurrence summary')]:
        if not os.path.exists(fpath):
            logger.error(f"Missing {label}: {fpath}")
            sys.exit(1)

    # --- Load inputs ---
    logger.info("Loading inputs...")
    with open(decoded_path, 'rb') as f:
        decoded_states = pickle.load(f)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    model_transmat = model.transmat_.copy()
    del model

    with open(summary_path, 'r') as f:
        recurrence_summary = json.load(f)
    recurrence_scores = np.array(recurrence_summary['recurrence_scores'])

    n_episodes = len(decoded_states)
    logger.info(f"Loaded {n_episodes} episodes, "
                f"{model_transmat.shape[0]} states in model")

    # --- Preprocessing ---
    logger.info(f"Collapsing to state-change sequences "
                f"(min_dwell_tr={args.min_dwell_tr})...")
    change_sequences = {}
    for run_id, seq in decoded_states.items():
        change_sequences[run_id] = collapse_to_state_changes(
            seq, min_dwell_tr=args.min_dwell_tr)

    total_changes = sum(len(s) for s in change_sequences.values())
    logger.info(f"Total state tokens in change sequences: {total_changes}")

    # State-change transition matrix from model
    P_change = compute_state_change_transmat(model_transmat)

    # --- Run preservation analysis per order ---
    summary = {
        'sub_id': sub_id,
        'parcellation': parc,
        'n_episodes': n_episodes,
        'n_simulations': args.n_simulations,
        'min_episodes': args.min_episodes,
        'min_dwell_tr': args.min_dwell_tr,
        'seed': args.seed,
        'orders': {},
    }

    all_sig_chains = {}  # order → list of significant chain tuples

    for order in range(2, args.max_order + 1):
        logger.info(f"--- Order {order} ---")

        df = compute_chain_preservation(
            change_sequences, P_change, order,
            min_episodes=args.min_episodes,
            n_simulations=args.n_simulations,
            seed=args.seed,
        )

        if df.empty:
            summary['orders'][order] = {'n_tested': 0, 'n_significant': 0}
            continue

        # Save CSV (drop chain_tuple column for clean output)
        csv_path = os.path.join(out_dir, f'preserved_chains_{order}grams.csv')
        df.drop(columns=['chain_tuple']).to_csv(csv_path, index=False)
        logger.info(f"Saved {csv_path}")

        # Figures
        plot_preservation_scatter(df, order, out_dir)
        plot_top_preserved(df, order, out_dir)
        plot_preservation_vs_recurrence(df, recurrence_scores, order, out_dir)

        # Track significant chains for season test
        sig_chains = list(df.loc[df['fdr_q'] < 0.05, 'chain_tuple'])
        all_sig_chains[order] = sig_chains

        summary['orders'][order] = {
            'n_tested': len(df),
            'n_significant': len(sig_chains),
            'top_5': df.head(5)[['chain', 'cps_obs', 'p_value', 'fdr_q']].to_dict('records'),
        }

    # --- Season-specificity test on significant chains ---
    for order, sig_chains in all_sig_chains.items():
        if not sig_chains:
            continue

        logger.info(f"--- Season-specificity test for {len(sig_chains)} "
                     f"significant {order}-grams ---")

        season_results = test_season_specificity(
            change_sequences, sig_chains, order,
            n_permutations=args.n_season_perms,
            seed=args.seed + order,  # different seed per order
        )

        if season_results:
            df_season = pd.DataFrame(season_results)
            # FDR correction across chains
            df_season['perm_fdr_q'] = benjamini_hochberg(
                df_season['perm_p'].values)
            # Flatten per_season_cps to separate columns
            season_cols = pd.json_normalize(df_season['per_season_cps'])
            season_cols.columns = [f'season_{c}_cps' for c in season_cols.columns]
            df_season = pd.concat([df_season.drop(columns=['per_season_cps', 'chain_tuple']),
                                   season_cols], axis=1)

            csv_path = os.path.join(out_dir, f'season_specificity_{order}grams.csv')
            df_season.to_csv(csv_path, index=False)
            logger.info(f"Saved {csv_path}")

            n_season_sig = int((df_season['perm_fdr_q'] < 0.05).sum())
            summary['orders'][order]['n_season_specific'] = n_season_sig
            logger.info(f"Order {order}: {n_season_sig}/{len(df_season)} "
                         f"chains are season-specific (FDR < 0.05)")

    # --- Save summary ---
    summary_path_out = os.path.join(out_dir, 'preservation_summary.json')
    with open(summary_path_out, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary: {summary_path_out}")

    logger.info("==============================================")
    logger.info("06d - Preserved chains analysis complete!")
    logger.info("==============================================")


if __name__ == '__main__':
    main()
