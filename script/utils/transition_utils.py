#!/usr/bin/env python3
"""
transition_utils.py - Utilities for state-change sequences and n-gram statistics.

Core functions for collapsing self-transitions, computing state-change transition
matrices, counting n-grams, and simulating Markov sequences. Used by 06c and 06d.
"""

from collections import Counter
from itertools import groupby

import numpy as np


def collapse_to_state_changes(state_seq, min_dwell_tr=2):
    """Collapse a decoded state sequence to state-change sequence.

    1. Extract contiguous blocks with durations.
    2. Remove blocks shorter than *min_dwell_tr* (hemodynamic blur artifacts).
    3. Return the sequence of state identities (self-transitions removed).

    Parameters
    ----------
    state_seq : np.ndarray
        Raw decoded state sequence (1-D, one entry per TR).
    min_dwell_tr : int
        Minimum dwell duration in TRs to keep a block (default 2).

    Returns
    -------
    change_seq : list[int]
        State identities after collapsing and filtering.
    """
    if len(state_seq) == 0:
        return []

    blocks = []
    for state_val, group in groupby(state_seq):
        duration = sum(1 for _ in group)
        blocks.append((int(state_val), duration))

    # Filter out short-dwell blocks
    filtered = [s for s, d in blocks if d >= min_dwell_tr]

    # Re-collapse adjacent duplicates that may arise from removing short blocks
    # (e.g., A(3)->B(1)->A(4) with min_dwell=2 removes B, leaving [A, A])
    if not filtered:
        return []
    collapsed = [filtered[0]]
    for s in filtered[1:]:
        if s != collapsed[-1]:
            collapsed.append(s)
    return collapsed


def compute_state_change_transmat(transmat):
    """Derive state-change transition matrix from the full transition matrix.

    P_change(j|i) = P(j|i) / (1 - P(i|i))   for j != i

    Parameters
    ----------
    transmat : np.ndarray, shape (K, K)
        Full transition probability matrix (rows sum to 1).

    Returns
    -------
    P_change : np.ndarray, shape (K, K)
        State-change transition matrix (diagonal is 0, rows sum to 1).
    """
    P_change = transmat.copy()
    np.fill_diagonal(P_change, 0.0)
    denom = 1.0 - np.diag(transmat)
    safe = denom > 1e-12
    P_change[safe] /= denom[safe, np.newaxis]
    P_change[~safe] = 0.0
    return P_change


def count_ngrams(change_sequences, n):
    """Count n-gram occurrences across all episodes.

    Parameters
    ----------
    change_sequences : dict[str, list[int]]
        run_id -> state-change sequence.
    n : int
        N-gram order (2 = bigrams, 3 = trigrams, etc.).

    Returns
    -------
    total_counts : Counter
        Mapping from n-gram tuple to total count across episodes.
    per_episode_counts : dict[str, Counter]
        Per-episode n-gram counts.
    """
    total_counts = Counter()
    per_episode_counts = {}

    for run_id, seq in change_sequences.items():
        ep_counter = Counter()
        for i in range(len(seq) - n + 1):
            ngram = tuple(seq[i:i + n])
            ep_counter[ngram] += 1
        per_episode_counts[run_id] = ep_counter
        total_counts.update(ep_counter)

    return total_counts, per_episode_counts


def simulate_markov_change_seq(length, P_change, start_state, rng):
    """Simulate a 1st-order Markov state-change sequence.

    Parameters
    ----------
    length : int
        Desired sequence length (number of state changes).
    P_change : np.ndarray, shape (K, K)
        State-change transition matrix (diagonal = 0, rows sum to 1).
    start_state : int
        Starting state index.
    rng : np.random.Generator
        Seeded random number generator.

    Returns
    -------
    seq : list[int]
        Simulated state-change sequence of the given length.
    """
    K = P_change.shape[0]
    seq = [start_state]
    s = start_state
    states = np.arange(K)

    for _ in range(length - 1):
        row = P_change[s]
        row_sum = row.sum()
        if row_sum < 1e-12:
            # Dead-end state: pick uniformly from non-self states
            candidates = states[states != s]
            s = rng.choice(candidates)
        else:
            s = rng.choice(K, p=row / row_sum)
        seq.append(s)

    return seq


def compute_recurrence_assortativity(G, decoded_states, active_states,
                                     recurrence_scores, edge_thresh=0.005,
                                     n_perm=5000, n_bootstrap=1000, seed=42,
                                     logger=None):
    """Recurrence assortativity with permutation test + bootstrap CI.

    Parameters
    ----------
    G : nx.DiGraph
        Pre-built transition graph with 'recurrence_score' node attribute.
    decoded_states : dict[str, np.ndarray]
        run_id → raw decoded state sequence.
    active_states : list[int]
        State indices present in the graph.
    recurrence_scores : np.ndarray
        Per-state recurrence scores (length K).
    edge_thresh : float
        Minimum transition probability for bootstrap graph edges.
    n_perm : int
        Number of node-label permutations.
    n_bootstrap : int
        Number of episode-bootstrap resamples for CI.
    seed : int
        Random seed.
    logger : logging.Logger, optional
        Logger instance.

    Returns
    -------
    dict with point_estimate, perm_p_value, bootstrap_ci, n_perm,
    n_bootstrap, n_edges.
    """
    import networkx as nx
    import logging as _logging
    if logger is None:
        logger = _logging.getLogger(__name__)

    rng = np.random.default_rng(seed)

    if G.number_of_edges() < 5:
        logger.warning("Too few edges for assortativity")
        return {'point_estimate': np.nan, 'perm_p_value': np.nan,
                'bootstrap_ci': [np.nan, np.nan]}

    # Point estimate
    try:
        r_obs = nx.numeric_assortativity_coefficient(G, 'recurrence_score')
    except Exception:
        logger.warning("Assortativity computation failed")
        return {'point_estimate': np.nan, 'perm_p_value': np.nan,
                'bootstrap_ci': [np.nan, np.nan]}

    # Node-permutation test
    rec_values = {n: G.nodes[n]['recurrence_score'] for n in G.nodes()}
    node_list = list(G.nodes())
    r_null = np.zeros(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(list(rec_values.values()))
        for n, val in zip(node_list, shuffled):
            G.nodes[n]['recurrence_score'] = val
        try:
            r_null[i] = nx.numeric_assortativity_coefficient(
                G, 'recurrence_score')
        except Exception:
            r_null[i] = 0.0
    # Restore original values
    for n, val in rec_values.items():
        G.nodes[n]['recurrence_score'] = val

    from utils.stats import permutation_pvalue
    perm_p = permutation_pvalue(r_obs, r_null, alternative='two-sided')

    # Episode-bootstrap CI
    run_ids = list(decoded_states.keys())
    K = recurrence_scores.shape[0]
    r_boot = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        boot_runs = rng.choice(run_ids, size=len(run_ids), replace=True)
        counts = np.zeros((K, K))
        for rid in boot_runs:
            seq = decoded_states[rid]
            if len(seq) < 2:
                continue
            for t in range(len(seq) - 1):
                counts[seq[t], seq[t + 1]] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            P_boot = np.nan_to_num(counts / row_sums, nan=0.0)

        G_boot = nx.DiGraph()
        for s in active_states:
            G_boot.add_node(s, recurrence_score=float(recurrence_scores[s]))
        for i in active_states:
            for j in active_states:
                if i != j and P_boot[i, j] > edge_thresh:
                    G_boot.add_edge(i, j, weight=float(P_boot[i, j]))
        try:
            r_boot[b] = nx.numeric_assortativity_coefficient(
                G_boot, 'recurrence_score')
        except Exception:
            r_boot[b] = np.nan

    r_boot = r_boot[~np.isnan(r_boot)]
    ci = [float(np.percentile(r_boot, 2.5)),
          float(np.percentile(r_boot, 97.5))] if len(r_boot) > 10 else [np.nan, np.nan]

    logger.info(f"Recurrence assortativity: r={r_obs:.3f}, perm p={perm_p:.4f}, "
                f"CI=[{ci[0]:.3f}, {ci[1]:.3f}]")

    return {
        'point_estimate': float(r_obs),
        'perm_p_value': float(perm_p),
        'bootstrap_ci': ci,
        'n_perm': n_perm,
        'n_bootstrap': n_bootstrap,
        'n_edges': G.number_of_edges(),
    }


def count_state_starts(change_sequences, n_gram_order):
    """Count how many times each state appears at a valid n-gram start position.

    Only positions with at least (n_gram_order - 1) successors are counted,
    avoiding upward bias in expected n-gram counts for higher-order n-grams.
    """
    counter = Counter()
    for seq in change_sequences.values():
        max_start = len(seq) - n_gram_order + 1
        for i in range(max(0, max_start)):
            counter[seq[i]] += 1
    return counter
