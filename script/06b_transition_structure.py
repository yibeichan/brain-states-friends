#!/usr/bin/env python3
"""
06b_transition_structure.py - Transition structure analysis of brain states.

Analyzes the directed transition graph between brain states discovered by the
combined HDP-HMM. Four analysis areas:

  A1. Transition Graph Topology — community detection, centrality, visualization
  A2. Transition Selectivity & Asymmetry — directional flow, concentration
  A3. Transition ↔ State Properties — recurrence assortativity, FC-transition
      correlation, network homophily
  A4. Transition Distance & Landscape — mean first passage time, MDS embedding

Design principles:
  - model.transmat_ for graph topology (A1) and MFPT (A4)
  - Empirical P (from 06a) for asymmetry (A2) and assortativity (A3)
  - networkx for directed weighted graph analysis

Prerequisites:
    - 04_combined_hdphmm.py (mode: select) completed
    - 05a_recurrence_analysis.py completed
    - 06a_state_temp_dynamics.py completed (transition_probabilities.npy)
    - 05f_state_fc.py completed (fc_similarity_corr_rv.npy; optional for A3)

Outputs:
    {SCRATCH_DIR}/output/06b_transition_structure/{parcellation}/{sub_id}/[vt{VT}/]
"""

import os
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import seaborn as sns

import networkx as nx

from dotenv import load_dotenv

# Setup paths and logger
sys.path.insert(0, str(Path(__file__).parent))
from utils.plot_style import (
    RECURRENCE_CMAP, recurrence_color, make_recurrence_colorbar,
    apply_publication_style,
    NETWORK_ORDER, NETWORK_COLORS,
    load_parcel_networks, compute_dominant_networks,
)
from utils.common import normalize_parcellation_name
from utils.stats import permutation_pvalue
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

# Edge-weight thresholds for graph construction.
# Topology analysis uses a sparser graph; assortativity uses a denser one.
_EDGE_THRESH_TOPOLOGY = 0.01
_EDGE_THRESH_ASSORTATIVITY = 0.005


# ===========================================================================
# A1: Transition Graph Topology
# ===========================================================================

def build_transition_graph(transmat, active_states, recurrence_scores,
                           dominant_networks, edge_threshold=_EDGE_THRESH_TOPOLOGY):
    """Build directed weighted graph from model transition matrix.

    Parameters
    ----------
    transmat : (K, K) array
        Full model transition matrix.
    active_states : list[int]
        Indices of active states (recurrence > 0).
    recurrence_scores : (K,) array
        Per-state recurrence scores.
    dominant_networks : dict
        state_id -> network name string.
    edge_threshold : float
        Minimum transition probability to include an edge.

    Returns
    -------
    G : nx.DiGraph
        Directed weighted graph with node/edge attributes.
    """
    G = nx.DiGraph()

    for s in active_states:
        G.add_node(s,
                   recurrence_score=float(recurrence_scores[s]),
                   dominant_network=dominant_networks.get(int(s), "Unknown"))

    for i in active_states:
        for j in active_states:
            if i == j:
                continue  # skip self-loops
            p = transmat[i, j]
            if p >= edge_threshold:
                G.add_edge(i, j, weight=float(p))

    logger.info(f"A1: Graph built — {G.number_of_nodes()} nodes, "
                f"{G.number_of_edges()} edges (threshold={edge_threshold})")
    return G


def compute_graph_metrics(G):
    """Compute per-node graph metrics.

    Returns DataFrame with columns: state_id, in_degree, out_degree,
    in_strength, out_strength, betweenness.
    """
    # Betweenness: invert weights so higher P = shorter path
    G_inv = G.copy()
    for u, v, d in G_inv.edges(data=True):
        d['distance'] = 1.0 / d['weight'] if d['weight'] > 0 else 1e10

    bc = nx.betweenness_centrality(G_inv, weight='distance')

    rows = []
    for n in G.nodes():
        in_str = sum(d['weight'] for _, _, d in G.in_edges(n, data=True))
        out_str = sum(d['weight'] for _, _, d in G.out_edges(n, data=True))
        rows.append({
            'state_id': n,
            'recurrence_score': G.nodes[n]['recurrence_score'],
            'dominant_network': G.nodes[n]['dominant_network'],
            'in_degree': G.in_degree(n),
            'out_degree': G.out_degree(n),
            'in_strength': float(in_str),
            'out_strength': float(out_str),
            'betweenness': float(bc[n]),
        })
    return pd.DataFrame(rows)


def detect_communities_with_stability(G, decoded_states, transmat,
                                      active_states, n_bootstrap=200,
                                      seed=42):
    """Community detection with bootstrap stability analysis.

    Returns community assignments and a consensus matrix showing how often
    each node pair co-clusters across bootstrap resamples.
    """
    rng = np.random.default_rng(seed)
    nodes = sorted(G.nodes())
    n_nodes = len(nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}

    # Primary community detection
    communities = nx.community.louvain_communities(G, weight='weight', seed=seed)
    primary_assignment = {}
    for cid, members in enumerate(communities):
        for m in members:
            primary_assignment[m] = cid

    # Bootstrap consensus matrix
    consensus = np.zeros((n_nodes, n_nodes))
    run_ids = list(decoded_states.keys())

    for b in range(n_bootstrap):
        # Resample episodes
        boot_runs = rng.choice(run_ids, size=len(run_ids), replace=True)

        # Rebuild empirical transition counts from resampled episodes
        K = transmat.shape[0]
        counts = np.zeros((K, K))
        for rid in boot_runs:
            seq = decoded_states[rid]
            if len(seq) < 2:
                continue
            for t in range(len(seq) - 1):
                counts[seq[t], seq[t + 1]] += 1

        # Normalize to probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            P_boot = np.nan_to_num(counts / row_sums, nan=0.0)

        # Build graph from bootstrap transition matrix
        G_boot = nx.DiGraph()
        for s in active_states:
            G_boot.add_node(s)
        for i in active_states:
            for j in active_states:
                if i != j and P_boot[i, j] > _EDGE_THRESH_TOPOLOGY:
                    G_boot.add_edge(i, j, weight=float(P_boot[i, j]))

        if G_boot.number_of_edges() == 0:
            continue

        # Detect communities
        try:
            boot_comms = nx.community.louvain_communities(
                G_boot, weight='weight', seed=seed + b)
        except Exception:
            continue

        # Update consensus
        for members in boot_comms:
            members_list = [node_idx[m] for m in members if m in node_idx]
            for i_idx in members_list:
                for j_idx in members_list:
                    consensus[i_idx, j_idx] += 1

    consensus /= max(n_bootstrap, 1)
    logger.info(f"A1: Community detection — {len(communities)} communities, "
                f"{n_bootstrap} bootstrap resamples")

    return primary_assignment, consensus, nodes


def plot_transition_graph(G, recurrence_scores, dominant_networks,
                          communities, out_dir):
    """Force-directed graph visualization."""
    fig, ax = plt.subplots(figsize=(12, 10))

    pos = nx.spring_layout(G, weight='weight', k=2.0, seed=42, iterations=100)

    # Node sizes proportional to recurrence
    node_sizes = [300 + 700 * G.nodes[n]['recurrence_score'] for n in G.nodes()]

    # Node colors from dominant network
    node_colors = []
    for n in G.nodes():
        net = G.nodes[n].get('dominant_network', 'Unknown')
        node_colors.append(NETWORK_COLORS.get(net, '#888888'))

    # Edge widths and alpha proportional to weight
    edges = G.edges(data=True)
    if edges:
        weights = [d['weight'] for _, _, d in edges]
        max_w = max(weights) if weights else 1
        edge_widths = [0.5 + 3.0 * (w / max_w) for w in weights]
        edge_alphas = [0.15 + 0.6 * (w / max_w) for w in weights]
    else:
        edge_widths = []
        edge_alphas = []

    # Draw edges
    for (u, v, d), ew, ea in zip(edges, edge_widths, edge_alphas):
        ax.annotate("", xy=pos[v], xytext=pos[u],
                     arrowprops=dict(arrowstyle='->', color='gray',
                                     lw=ew, alpha=ea,
                                     connectionstyle='arc3,rad=0.1'))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color=node_colors, edgecolors='black',
                           linewidths=0.5, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color='black')

    # Legend for network colors
    present_nets = set(G.nodes[n].get('dominant_network', 'Unknown')
                       for n in G.nodes())
    handles = [mpatches.Patch(color=NETWORK_COLORS.get(net, '#888888'), label=net)
               for net in NETWORK_ORDER if net in present_nets]
    ax.legend(handles=handles, loc='upper left', fontsize=7, framealpha=0.8)

    ax.set_title('Transition Graph (model transmat_, self-loops removed)')
    ax.axis('off')

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'fig_A1_transition_graph.{ext}'),
                    dpi=300 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)


def plot_degree_centrality(df_metrics, out_dir):
    """2-panel: degree distribution + centrality vs recurrence."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) In-degree vs out-degree
    ax = axes[0]
    colors = [recurrence_color(r) for r in df_metrics['recurrence_score']]
    ax.scatter(df_metrics['in_degree'], df_metrics['out_degree'],
               c=colors, s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
    for _, row in df_metrics.iterrows():
        ax.annotate(str(int(row['state_id'])),
                     (row['in_degree'], row['out_degree']),
                     fontsize=6, alpha=0.5, ha='center', va='bottom')
    ax.set_xlabel('In-Degree')
    ax.set_ylabel('Out-Degree')
    ax.set_title('(a) Degree Distribution')
    # Diagonal reference
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], '--', color='gray', alpha=0.4)

    # (b) Betweenness centrality vs recurrence
    ax = axes[1]
    ax.scatter(df_metrics['recurrence_score'], df_metrics['betweenness'],
               c=colors, s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
    for _, row in df_metrics.iterrows():
        ax.annotate(str(int(row['state_id'])),
                     (row['recurrence_score'], row['betweenness']),
                     fontsize=6, alpha=0.5, ha='center', va='bottom')
    ax.set_xlabel('Recurrence Score')
    ax.set_ylabel('Betweenness Centrality')
    ax.set_title('(b) Centrality vs Recurrence')

    if len(df_metrics) >= 5:
        rho, p = sp_stats.spearmanr(df_metrics['recurrence_score'],
                                     df_metrics['betweenness'])
        ax.text(0.02, 0.98, f'Spearman ρ={rho:.2f}, p={p:.2e}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                          alpha=0.8))

    make_recurrence_colorbar(axes[0])
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'fig_A1b_degree_centrality.{ext}'),
                    dpi=300 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# A2: Transition Selectivity & Asymmetry
# ===========================================================================

def compute_asymmetry_matrix(P_empirical, active_states):
    """Compute pairwise transition asymmetry.

    asym(i,j) = (P(j|i) - P(i|j)) / (P(j|i) + P(i|j))
    Only computed for pairs where both P > 1/(K-1).

    Returns (n_active, n_active) matrix, NaN for filtered pairs.
    """
    n = len(active_states)
    K_minus_1 = max(n - 1, 1)
    threshold = 1.0 / K_minus_1

    asym = np.full((n, n), np.nan)
    for ii, i in enumerate(active_states):
        for jj, j in enumerate(active_states):
            if ii == jj:
                asym[ii, jj] = 0.0
                continue
            p_ij = P_empirical[i, j]
            p_ji = P_empirical[j, i]
            if p_ij >= threshold or p_ji >= threshold:
                denom = p_ij + p_ji
                if denom > 1e-15:
                    asym[ii, jj] = (p_ij - p_ji) / denom
                else:
                    asym[ii, jj] = 0.0
    return asym


def compute_selectivity_metrics(P_empirical, active_states, recurrence_scores):
    """Per-state transition selectivity metrics.

    Returns DataFrame with: state_id, recurrence_score, top1_target, top1_prob,
    top2_target, top2_prob, concentration_ratio, bidirectionality.
    """
    rows = []
    for s in active_states:
        # Exit probabilities (excluding self-transition)
        exit_probs = P_empirical[s, :].copy()
        exit_probs[s] = 0.0
        total_exit = exit_probs.sum()

        if total_exit < 1e-15:
            rows.append({
                'state_id': s,
                'recurrence_score': float(recurrence_scores[s]),
                'top1_target': -1, 'top1_prob': 0.0,
                'top2_target': -1, 'top2_prob': 0.0,
                'concentration_ratio': 0.0,
            })
            continue

        sorted_idx = np.argsort(exit_probs)[::-1]
        top1 = sorted_idx[0]
        top2 = sorted_idx[1] if len(sorted_idx) > 1 else -1
        top2_prob = exit_probs[top2] if top2 >= 0 else 0.0
        conc = (exit_probs[top1] + top2_prob) / total_exit

        rows.append({
            'state_id': s,
            'recurrence_score': float(recurrence_scores[s]),
            'top1_target': int(top1),
            'top1_prob': float(exit_probs[top1]),
            'top2_target': int(top2),
            'top2_prob': float(top2_prob),
            'concentration_ratio': float(conc),
        })

    return pd.DataFrame(rows)


def compute_bidirectionality(P_empirical, active_states, threshold=1e-6):
    """Fraction of non-zero directed edges that are reciprocated."""
    n_edges = 0
    n_reciprocated = 0
    for i in active_states:
        for j in active_states:
            if i == j:
                continue
            if P_empirical[i, j] > threshold:
                n_edges += 1
                if P_empirical[j, i] > threshold:
                    n_reciprocated += 1
    bidir = n_reciprocated / n_edges if n_edges > 0 else 0.0
    return {'n_directed_edges': n_edges,
            'n_reciprocated': n_reciprocated,
            'bidirectionality_index': bidir}


def plot_asymmetry_heatmap(asym_matrix, active_states, recurrence_scores,
                           out_dir):
    """Asymmetry heatmap sorted by recurrence, with marginals."""
    n = len(active_states)
    # Sort by descending recurrence
    order = np.argsort([recurrence_scores[s] for s in active_states])[::-1]
    sorted_states = [active_states[i] for i in order]
    asym_sorted = asym_matrix[order, :][:, order]

    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[20, 1], height_ratios=[1, 20],
                          hspace=0.05, wspace=0.05)

    # Main heatmap
    ax_main = fig.add_subplot(gs[1, 0])
    mask = np.isnan(asym_sorted)
    sns.heatmap(asym_sorted, mask=mask, cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, xticklabels=False, yticklabels=False,
                ax=ax_main, cbar_ax=fig.add_subplot(gs[1, 1]))
    ax_main.set_xlabel('To State (sorted by recurrence)')
    ax_main.set_ylabel('From State (sorted by recurrence)')
    ax_main.set_title('Transition Asymmetry Matrix')

    # Top marginal: net inflow per state
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    net_inflow = np.nanmean(asym_sorted, axis=0)  # mean asymmetry toward this state
    colors = [recurrence_color(recurrence_scores[s]) for s in sorted_states]
    ax_top.bar(range(n), net_inflow, color=colors, edgecolor='none', width=1.0)
    ax_top.axhline(0, color='black', lw=0.5)
    ax_top.set_ylabel('Net inflow')
    ax_top.tick_params(labelbottom=False)
    ax_top.set_xlim(-0.5, n - 0.5)

    plt.suptitle('A2: Transition Asymmetry (positive = A→B > B→A)', y=1.01)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'fig_A2_asymmetry.{ext}'),
                    dpi=300 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# A3: Transition ↔ State Properties
# ===========================================================================

def compute_recurrence_assortativity(G, decoded_states, active_states,
                                     recurrence_scores,
                                     n_perm=5000, n_bootstrap=1000, seed=42):
    """Recurrence assortativity with permutation test + bootstrap CI.

    Delegates to shared implementation in transition_utils.
    Returns dict with point_estimate, perm_p_value, bootstrap_ci.
    """
    return _shared_assortativity(
        G, decoded_states, active_states, recurrence_scores,
        edge_thresh=_EDGE_THRESH_ASSORTATIVITY,
        n_perm=n_perm, n_bootstrap=n_bootstrap, seed=seed,
        logger=logger,
    )


def test_fc_transition_correlation(P_empirical, rv_matrix, active_states,
                                   n_perm=5000, seed=42):
    """Mantel test: FC similarity vs symmetrized transition probability.

    NaN handling: the rv_matrix from 05f embeds NaN for state pairs where
    one or both states had insufficient TRs (n_trs < min_trs=30) to estimate
    a reliable FC profile. The observed rho is computed only on non-NaN
    cells. The permutation null requires per-permutation NaN handling:
    after row/column permutation of rv_sub, NaN cells move to new positions,
    so the original position mask cannot be re-used (doing so yields NaN
    rho for nearly every permutation, collapsing the effective null sample
    to ~1 and producing spurious p ≈ 0.5; bug fixed 2026-05-22).
    """
    rng = np.random.default_rng(seed)
    n = len(active_states)
    idx = np.array(active_states)

    # Extract sub-matrices for active states
    P_sub = P_empirical[np.ix_(idx, idx)].copy()
    rv_sub = rv_matrix[np.ix_(idx, idx)].copy()

    # Symmetrize transition matrix
    T_sym = (P_sub + P_sub.T) / 2.0
    np.fill_diagonal(T_sym, 0)
    np.fill_diagonal(rv_sub, 0)

    # Extract upper triangle (full vectors retained for per-permutation masking)
    triu_idx = np.triu_indices(n, k=1)
    t_vec_full = T_sym[triu_idx]
    rv_vec_full = rv_sub[triu_idx]

    # Observed-side mask: keep cells where (i) at least one of t/rv is non-zero
    # (informative pair) and (ii) rv is non-NaN.
    mask = ((t_vec_full > 1e-10) | (rv_vec_full > 1e-10)) & ~np.isnan(rv_vec_full)
    t_vec = t_vec_full[mask]
    rv_vec = rv_vec_full[mask]

    if len(t_vec) < 10:
        return {'rho': np.nan, 'p_value': np.nan, 'n_pairs': 0}

    rho_obs, _ = sp_stats.spearmanr(t_vec, rv_vec)
    if np.isnan(rho_obs):
        return {'rho': np.nan, 'p_value': np.nan, 'n_pairs': int(len(t_vec))}

    # Permutation null: shuffle rv state labels, then recompute per-permutation
    # mask = original informativeness mask ∩ non-NaN positions in permuted matrix.
    rho_null = np.full(n_perm, np.nan)
    n_skipped = 0
    n_eff_min = None
    for i in range(n_perm):
        perm = rng.permutation(n)
        rv_perm_full = rv_sub[np.ix_(perm, perm)][triu_idx]
        mask_i = mask & ~np.isnan(rv_perm_full)
        n_eff = int(mask_i.sum())
        if n_eff_min is None or n_eff < n_eff_min:
            n_eff_min = n_eff
        if n_eff < 10:
            n_skipped += 1
            continue
        rho_null[i], _ = sp_stats.spearmanr(t_vec_full[mask_i], rv_perm_full[mask_i])

    if n_skipped == n_perm:
        logger.warning(
            "A3: ALL %d permutations skipped (n_eff < 10 in every permuted matrix). "
            "Permutation null has no support; p_value will be NaN.",
            n_perm,
        )
    p_value = permutation_pvalue(rho_obs, rho_null, alternative='two-sided')

    logger.info(
        f"A3: FC-transition correlation rho={rho_obs:.3f}, p={p_value:.4f} "
        f"(n_pairs_obs={int(mask.sum())}, n_perm_skipped={n_skipped}, n_eff_min={n_eff_min})"
    )
    return {
        'rho': float(rho_obs),
        'p_value': float(p_value),
        'n_pairs': int(mask.sum()),
        'n_perm': n_perm,
        'n_perm_skipped': n_skipped,
        'n_eff_min': int(n_eff_min) if n_eff_min is not None else None,
    }


def test_network_homophily(P_empirical, active_states, dominant_networks,
                           n_perm=5000, seed=42):
    """Test within-network vs between-network mean transition probability."""
    rng = np.random.default_rng(seed)

    within_probs = []
    between_probs = []
    for i in active_states:
        for j in active_states:
            if i == j:
                continue
            p = P_empirical[i, j]
            if p < 1e-15:
                continue
            net_i = dominant_networks.get(int(i), "Unknown")
            net_j = dominant_networks.get(int(j), "Unknown")
            if net_i == net_j and net_i != "Unknown":
                within_probs.append(p)
            else:
                between_probs.append(p)

    within_mean = float(np.mean(within_probs)) if within_probs else 0.0
    between_mean = float(np.mean(between_probs)) if between_probs else 0.0
    ratio_obs = within_mean / between_mean if between_mean > 0 else np.inf
    diff_obs = within_mean - between_mean

    # Permutation test: shuffle network labels (use difference statistic
    # to avoid undefined ratio when between_mean = 0)
    net_labels = [dominant_networks.get(int(s), "Unknown") for s in active_states]
    diff_null = np.zeros(n_perm)
    for p_idx in range(n_perm):
        shuffled_labels = rng.permutation(net_labels)
        label_map = {s: shuffled_labels[ii] for ii, s in enumerate(active_states)}
        w, b = [], []
        for i in active_states:
            for j in active_states:
                if i == j:
                    continue
                p = P_empirical[i, j]
                if p < 1e-15:
                    continue
                if label_map[i] == label_map[j] and label_map[i] != "Unknown":
                    w.append(p)
                else:
                    b.append(p)
        w_m = float(np.mean(w)) if w else 0.0
        b_m = float(np.mean(b)) if b else 0.0
        diff_null[p_idx] = w_m - b_m

    p_value = permutation_pvalue(diff_obs, diff_null, alternative='greater')

    logger.info(f"A3: Network homophily within={within_mean:.4f}, "
                f"between={between_mean:.4f}, ratio={ratio_obs:.2f}, "
                f"diff={diff_obs:.4f}, p={p_value:.4f}")
    return {
        'within_mean': within_mean,
        'between_mean': between_mean,
        'ratio': float(ratio_obs),
        'difference': float(diff_obs),
        'p_value': float(p_value),
        'n_within_edges': len(within_probs),
        'n_between_edges': len(between_probs),
        'n_perm': n_perm,
    }


def plot_assortativity_panel(assort_result, fc_result, homophily_result,
                             out_dir):
    """3-panel figure for A3 results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) Assortativity with CI
    ax = axes[0]
    r = assort_result['point_estimate']
    ci = assort_result['bootstrap_ci']
    ax.barh(0, r, color='steelblue', height=0.5, alpha=0.8)
    if not np.isnan(ci[0]):
        ax.errorbar(r, 0, xerr=[[r - ci[0]], [ci[1] - r]], fmt='none',
                    color='black', capsize=5, lw=2)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('Assortativity Coefficient')
    ax.set_yticks([])
    p_str = f"p={assort_result['perm_p_value']:.3f}" if not np.isnan(assort_result['perm_p_value']) else "p=N/A"
    ax.set_title(f'(a) Recurrence Assortativity\nr={r:.3f}, {p_str}')

    # (b) FC similarity vs transition
    ax = axes[1]
    rho = fc_result.get('rho', np.nan)
    p_val = fc_result.get('p_value', np.nan)
    ax.text(0.5, 0.5, f'ρ = {rho:.3f}\np = {p_val:.3f}\nn = {fc_result.get("n_pairs", 0)} pairs',
            transform=ax.transAxes, ha='center', va='center', fontsize=14)
    ax.set_title('(b) FC Similarity ↔ Transition')
    ax.set_xlabel('(Mantel test: Spearman correlation)')
    ax.set_yticks([])

    # (c) Network homophily
    ax = axes[2]
    bars = ax.bar(['Within\nNetwork', 'Between\nNetwork'],
                  [homophily_result['within_mean'], homophily_result['between_mean']],
                  color=['#4CAF50', '#FF9800'], alpha=0.8)
    ax.set_ylabel('Mean Transition Probability')
    ratio = homophily_result['ratio']
    p_hom = homophily_result['p_value']
    ax.set_title(f'(c) Network Homophily\nratio={ratio:.2f}, p={p_hom:.3f}')

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'fig_A3_assortativity_panel.{ext}'),
                    dpi=300 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# A4: Transition Distance & Landscape
# ===========================================================================

def compute_mfpt(transmat, active_states):
    """Compute Mean First Passage Time from model transition matrix.

    Restricts to the largest strongly connected component of active states.
    Uses Z = (I - P + W)^{-1}, MFPT_ij = (Z_jj - Z_ij) / pi_j.

    Returns
    -------
    mfpt : (n_scc, n_scc) array
        MFPT matrix for states in the largest SCC.
    stationary : (n_scc,) array
        Stationary distribution.
    scc_states : list[int]
        State indices in the largest SCC.
    """
    # Build directed graph to find SCC
    G_scc = nx.DiGraph()
    for s in active_states:
        G_scc.add_node(s)
    for i in active_states:
        for j in active_states:
            if i != j and transmat[i, j] > 1e-15:
                G_scc.add_edge(i, j)

    sccs = list(nx.strongly_connected_components(G_scc))
    if not sccs:
        logger.warning("A4: No strongly connected components found")
        return None, None, []

    largest_scc = max(sccs, key=len)
    scc_states = sorted(largest_scc)
    n = len(scc_states)
    logger.info(f"A4: Largest SCC has {n}/{len(active_states)} active states")

    if n < 3:
        logger.warning("A4: SCC too small for MFPT computation")
        return None, None, scc_states

    # Extract sub-matrix
    idx = np.array(scc_states)
    P = transmat[np.ix_(idx, idx)].copy()
    # Validate row sums — for a true SCC they should already be ~1.0
    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        logger.warning(
            f"A4: SCC rows don't sum to 1 "
            f"(range {row_sums.min():.6f}–{row_sums.max():.6f}); renormalizing"
        )
        P = P / row_sums[:, np.newaxis]

    # Stationary distribution: left eigenvector of P^T with eigenvalue 1
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    # Find eigenvalue closest to 1
    idx_1 = np.argmin(np.abs(eigenvalues - 1.0))
    imag_norm = np.linalg.norm(np.imag(eigenvectors[:, idx_1]))
    if imag_norm > 1e-8:
        logger.warning(
            f"A4: Stationary eigenvector has non-negligible imaginary part "
            f"(norm={imag_norm:.2e})"
        )
    pi = np.real(eigenvectors[:, idx_1])
    if pi.sum() < 0:
        pi = -pi
    pi = pi / pi.sum()  # normalize
    pi = np.maximum(pi, 1e-15)  # avoid division by zero

    # Fundamental matrix
    I = np.eye(n)
    W = np.ones((n, 1)) @ pi[np.newaxis, :]
    A = I - P + W

    # Check condition number
    cond = np.linalg.cond(A)
    if cond > 1e12:
        logger.warning(f"A4: Ill-conditioned fundamental matrix (cond={cond:.1e})")

    try:
        Z = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        logger.error("A4: Fundamental matrix inversion failed")
        return None, None, scc_states

    # MFPT_ij = (Z_jj - Z_ij) / pi_j
    mfpt = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                mfpt[i, j] = 0.0
            else:
                mfpt[i, j] = (Z[j, j] - Z[i, j]) / pi[j]

    # Sanity check: MFPT should be non-negative
    if np.any(mfpt < -0.01):
        logger.warning(f"A4: Negative MFPT values detected (min={mfpt.min():.3f})")
        mfpt = np.maximum(mfpt, 0.0)

    return mfpt, pi, scc_states


def plot_mfpt_matrix(mfpt, scc_states, recurrence_scores, out_dir):
    """MFPT heatmap sorted by recurrence, log scale."""
    n = len(scc_states)
    order = np.argsort([recurrence_scores[s] for s in scc_states])[::-1]
    sorted_states = [scc_states[i] for i in order]
    mfpt_sorted = mfpt[order, :][:, order]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Log scale for display (add 1 to avoid log(0))
    mfpt_log = np.log10(mfpt_sorted + 1)
    mask = mfpt_sorted == 0  # diagonal

    sns.heatmap(mfpt_log, mask=mask, cmap='YlOrRd', robust=True,
                xticklabels=False, yticklabels=False, ax=ax)
    ax.set_xlabel('To State (sorted by recurrence)')
    ax.set_ylabel('From State (sorted by recurrence)')
    ax.set_title('Mean First Passage Time (log₁₀ scale)')

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'fig_A4a_mfpt_matrix.{ext}'),
                    dpi=300 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)


def plot_transition_landscape(mfpt, scc_states, recurrence_scores,
                              dominant_networks, state_summary, out_dir):
    """MDS embedding of MFPT → 2D 'state space map'."""
    n = len(scc_states)
    if n < 4:
        logger.warning("A4: Too few states for MDS embedding")
        return

    # Symmetrize MFPT for MDS
    D_sym = (mfpt + mfpt.T) / 2.0
    np.fill_diagonal(D_sym, 0)

    # MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42,
              normalized_stress='auto')
    coords = mds.fit_transform(D_sym)
    stress = mds.stress_

    fig, ax = plt.subplots(figsize=(12, 10))

    # Node properties
    rec_vals = np.array([recurrence_scores[s] for s in scc_states])

    # Size from total occupancy (from state summary table)
    if state_summary is not None and 'total_occupancy_s' in state_summary.columns:
        occ_map = dict(zip(state_summary['state_id'], state_summary['total_occupancy_s']))
        sizes = np.array([occ_map.get(s, 100) for s in scc_states])
        sizes = 100 + 900 * (sizes / max(sizes.max(), 1))
    else:
        sizes = 100 + 900 * rec_vals

    # Draw top-3 transitions per state as directed edges
    # Use the model transmat restricted to SCC
    for ii, i_state in enumerate(scc_states):
        # Get exit probabilities to other SCC states
        exit_probs = mfpt[ii, :].copy()
        exit_probs[ii] = np.inf  # exclude self
        # Find top-3 closest (lowest MFPT)
        top3 = np.argsort(exit_probs)[:3]
        for jj in top3:
            if exit_probs[jj] == np.inf:
                continue
            ax.annotate("", xy=coords[jj], xytext=coords[ii],
                        arrowprops=dict(arrowstyle='->', color='gray',
                                        lw=0.5, alpha=0.2,
                                        connectionstyle='arc3,rad=0.1'))

    # Draw nodes
    for ii, s in enumerate(scc_states):
        net = dominant_networks.get(int(s), "Unknown")
        color = NETWORK_COLORS.get(net, '#888888')
        # Border darkness proportional to recurrence
        border_alpha = 0.3 + 0.7 * rec_vals[ii]
        ax.scatter(coords[ii, 0], coords[ii, 1], s=sizes[ii],
                   c=color, edgecolors='black', linewidth=1.5,
                   alpha=0.85, zorder=3)
        ax.annotate(str(s), (coords[ii, 0], coords[ii, 1]),
                    fontsize=7, ha='center', va='center', zorder=4,
                    fontweight='bold' if rec_vals[ii] > 0.7 else 'normal')

    # Legend
    present_nets = set(dominant_networks.get(int(s), "Unknown")
                       for s in scc_states)
    handles = [mpatches.Patch(color=NETWORK_COLORS.get(net, '#888888'), label=net)
               for net in NETWORK_ORDER if net in present_nets]
    ax.legend(handles=handles, loc='upper left', fontsize=8, framealpha=0.8)

    ax.set_title(f'Transition Landscape (MDS on MFPT, stress={stress:.3f})')
    ax.set_xlabel('MDS Dimension 1')
    ax.set_ylabel('MDS Dimension 2')

    # Post-hoc: correlate MDS dims with recurrence
    for dim in range(2):
        rho, p = sp_stats.spearmanr(coords[:, dim], rec_vals)
        ax.text(0.02, 0.02 + dim * 0.04,
                f'Dim{dim + 1} × recurrence: ρ={rho:.2f}, p={p:.2e}',
                transform=ax.transAxes, fontsize=8, va='bottom')

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'fig_A4b_transition_landscape.{ext}'),
                    dpi=300 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze transition structure of brain states.")
    parser.add_argument('--sub_id', type=str, required=True,
                        help="Subject ID (e.g., sub-01)")
    parser.add_argument('--parcellation', type=str, default='atlas-4S156Parcels')
    parser.add_argument('--vt', type=str, default=None,
                        help="Variance threshold subdirectory (e.g., 0.95)")
    args = parser.parse_args()

    parc = normalize_parcellation_name(args.parcellation)
    parc_full = f'atlas-{parc}' if not args.parcellation.startswith('atlas-') else args.parcellation
    sub_id = args.sub_id

    logger.info("==============================================")
    logger.info("06b - Transition Structure Analysis")
    logger.info("==============================================")
    logger.info(f"Subject: {sub_id}, Parcellation: {parc}")

    # --- Paths ---
    vt_suffix = f'vt{args.vt}' if args.vt else None

    def _path(stage, *parts):
        base = os.path.join(SCRATCH_DIR, 'output', stage, parc, sub_id)
        if vt_suffix:
            base = os.path.join(base, vt_suffix)
        return os.path.join(base, *parts)

    hmm_base = os.path.join(SCRATCH_DIR, 'output', '04_combined_hdphmm', parc,
                            sub_id, 'final')
    if vt_suffix:
        hmm_base = os.path.join(hmm_base, vt_suffix)

    out_dir = _path('06b_transition_structure')
    os.makedirs(out_dir, exist_ok=True)

    # --- Load data ---
    logger.info("Loading inputs...")

    # Model and decoded states (from 04)
    model_path = os.path.join(hmm_base, 'best_model.pkl')
    decoded_path = os.path.join(hmm_base, 'decoded_states.pkl')
    for fpath, label in [(model_path, 'best model'), (decoded_path, 'decoded states')]:
        if not os.path.exists(fpath):
            logger.error(f"Missing {label}: {fpath}")
            sys.exit(1)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    model_transmat = model.transmat_.copy()

    # Try to load state means for network annotation
    state_means_path = os.path.join(hmm_base, 'state_means_parcel.npy')
    state_means = np.load(state_means_path) if os.path.exists(state_means_path) else None
    del model

    with open(decoded_path, 'rb') as f:
        decoded_states = pickle.load(f)

    # Recurrence scores (from 05a)
    recur_path = _path('05a_recurrence_analysis', 'recurrence_summary.json')
    if not os.path.exists(recur_path):
        logger.error(f"Missing recurrence summary: {recur_path}")
        sys.exit(1)
    with open(recur_path) as f:
        recurrence_summary = json.load(f)

    n_states = recurrence_summary['n_states']
    recurrence_scores = np.array(recurrence_summary['recurrence_scores'])
    active_states = [i for i in range(n_states) if recurrence_scores[i] > 0]
    if not active_states:
        logger.error("No active states found (all recurrence_scores = 0). Exiting.")
        sys.exit(1)

    # Empirical transition probabilities (from 06a)
    emp_P_path = _path('06a_state_temp_dynamics', 'transition_probabilities.npy')
    if os.path.exists(emp_P_path):
        P_empirical = np.load(emp_P_path)
        logger.info(f"Loaded empirical transition matrix from 06a")
    else:
        logger.warning("06a transition_probabilities.npy not found; computing from decoded states")
        P_counts = np.zeros((n_states, n_states))
        for seq in decoded_states.values():
            if len(seq) < 2:
                continue
            for t in range(len(seq) - 1):
                P_counts[seq[t], seq[t + 1]] += 1
        row_sums = P_counts.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            P_empirical = np.nan_to_num(P_counts / row_sums, nan=0.0)

    # FC similarity (from 05f, optional)
    fc_path = _path('05f_state_fc', 'fc_similarity_corr_rv.npy')
    rv_matrix = None
    if os.path.exists(fc_path):
        rv_matrix = np.load(fc_path)
        logger.info(f"Loaded FC similarity matrix from 05f (shape={rv_matrix.shape})")
    else:
        logger.info("05f FC similarity not found — A3 FC-transition test will be skipped")

    # State summary table (from 06a, for occupancy data)
    summary_path = _path('06a_state_temp_dynamics', 'state_summary_table.csv')
    state_summary = pd.read_csv(summary_path) if os.path.exists(summary_path) else None

    # Dominant networks
    dominant_networks = {}
    if state_means is not None:
        parcel_networks = load_parcel_networks(parc)
        if parcel_networks is not None:
            dominant_networks = compute_dominant_networks(
                state_means, np.array(active_states), parcel_networks)
            logger.info(f"Computed dominant networks for {len(dominant_networks)} states")

    # ===================== A1: Graph Topology =====================
    logger.info("=" * 50)
    logger.info("A1: Transition Graph Topology")

    G = build_transition_graph(model_transmat, active_states, recurrence_scores,
                               dominant_networks)

    df_metrics = compute_graph_metrics(G)
    df_metrics.to_csv(os.path.join(out_dir, 'graph_metrics.csv'), index=False)

    communities, consensus, consensus_nodes = detect_communities_with_stability(
        G, decoded_states, model_transmat, active_states, n_bootstrap=200)

    with open(os.path.join(out_dir, 'community_assignments.json'), 'w') as f:
        json.dump({str(k): int(v) for k, v in communities.items()}, f, indent=2)

    # Save graph
    nx.write_graphml(G, os.path.join(out_dir, 'transition_graph.graphml'))

    plot_transition_graph(G, recurrence_scores, dominant_networks,
                          communities, out_dir)
    plot_degree_centrality(df_metrics, out_dir)

    # ===================== A2: Selectivity & Asymmetry =====================
    logger.info("=" * 50)
    logger.info("A2: Transition Selectivity & Asymmetry")

    asym_matrix = compute_asymmetry_matrix(P_empirical, active_states)
    np.save(os.path.join(out_dir, 'asymmetry_matrix.npy'), asym_matrix)

    df_select = compute_selectivity_metrics(P_empirical, active_states,
                                            recurrence_scores)
    df_select.to_csv(os.path.join(out_dir, 'selectivity_metrics.csv'), index=False)

    bidir = compute_bidirectionality(P_empirical, active_states)
    logger.info(f"A2: Bidirectionality index = {bidir['bidirectionality_index']:.3f}")

    plot_asymmetry_heatmap(asym_matrix, active_states, recurrence_scores, out_dir)

    # ===================== A3: Assortativity =====================
    logger.info("=" * 50)
    logger.info("A3: Transition ↔ State Properties")

    # Build empirical graph for assortativity (use empirical P, not model)
    G_emp = build_transition_graph(P_empirical, active_states, recurrence_scores,
                                   dominant_networks, edge_threshold=_EDGE_THRESH_ASSORTATIVITY)

    assort_result = compute_recurrence_assortativity(
        G_emp, decoded_states, active_states, recurrence_scores)
    with open(os.path.join(out_dir, 'recurrence_assortativity.json'), 'w') as f:
        json.dump(assort_result, f, indent=2)

    fc_result = {'rho': np.nan, 'p_value': np.nan, 'n_pairs': 0}
    if rv_matrix is not None:
        fc_result = test_fc_transition_correlation(P_empirical, rv_matrix,
                                                   active_states)
    with open(os.path.join(out_dir, 'fc_transition_correlation.json'), 'w') as f:
        json.dump(fc_result, f, indent=2)

    homophily_result = test_network_homophily(P_empirical, active_states,
                                             dominant_networks)
    with open(os.path.join(out_dir, 'network_homophily.json'), 'w') as f:
        json.dump(homophily_result, f, indent=2)

    plot_assortativity_panel(assort_result, fc_result, homophily_result, out_dir)

    # ===================== A4: MFPT & Landscape =====================
    logger.info("=" * 50)
    logger.info("A4: Transition Distance & Landscape")

    mfpt, stationary, scc_states = compute_mfpt(model_transmat, active_states)

    if mfpt is not None:
        np.save(os.path.join(out_dir, 'mfpt_matrix.npy'), mfpt)
        np.save(os.path.join(out_dir, 'stationary_distribution.npy'), stationary)

        plot_mfpt_matrix(mfpt, scc_states, recurrence_scores, out_dir)
        plot_transition_landscape(mfpt, scc_states, recurrence_scores,
                                 dominant_networks, state_summary, out_dir)

        # MFPT ↔ FC correlation (Mantel test)
        # NaN handling: see test_fc_transition_correlation docstring — same
        # permutation NaN-mask bug, same fix (per-permutation mask intersection).
        mfpt_fc_result = {'rho': np.nan, 'p_value': np.nan}
        if rv_matrix is not None and len(scc_states) >= 5:
            # Symmetrize MFPT as distance
            D_mfpt = (mfpt + mfpt.T) / 2.0
            # FC dissimilarity = 1 - RV
            scc_idx = np.array(scc_states)
            rv_sub = rv_matrix[np.ix_(scc_idx, scc_idx)]
            D_fc = 1.0 - rv_sub

            n_scc = len(scc_states)
            triu_idx = np.triu_indices(n_scc, k=1)
            mfpt_vec_full = D_mfpt[triu_idx]
            fc_vec_full = D_fc[triu_idx]

            # Observed-side mask: drop pairs with NaN in either vector.
            valid = ~np.isnan(fc_vec_full) & ~np.isnan(mfpt_vec_full)
            mfpt_vec_f = mfpt_vec_full[valid]
            fc_vec_f = fc_vec_full[valid]

            rng = np.random.default_rng(42)
            if len(mfpt_vec_f) < 10:
                logger.warning("A4: Too few valid MFPT-FC pairs — skipping correlation")
            else:
                rho_obs, _ = sp_stats.spearmanr(mfpt_vec_f, fc_vec_f)
                if np.isnan(rho_obs):
                    logger.warning("A4: MFPT-FC rho is NaN (constant input?) — skipping")
                else:
                    n_perm = 5000
                    rho_null = np.full(n_perm, np.nan)
                    n_skipped = 0
                    n_eff_min = None
                    for i in range(n_perm):
                        perm = rng.permutation(n_scc)
                        fc_perm_full = D_fc[np.ix_(perm, perm)][triu_idx]
                        mask_i = valid & ~np.isnan(fc_perm_full)
                        n_eff = int(mask_i.sum())
                        if n_eff_min is None or n_eff < n_eff_min:
                            n_eff_min = n_eff
                        if n_eff < 10:
                            n_skipped += 1
                            continue
                        rho_null[i], _ = sp_stats.spearmanr(
                            mfpt_vec_full[mask_i], fc_perm_full[mask_i]
                        )
                    if n_skipped == n_perm:
                        logger.warning(
                            "A4: ALL %d permutations skipped (n_eff < 10 in every permuted matrix). "
                            "Permutation null has no support; p_value will be NaN.",
                            n_perm,
                        )
                    p_val = permutation_pvalue(rho_obs, rho_null, alternative='two-sided')
                    mfpt_fc_result = {
                        'rho': float(rho_obs), 'p_value': p_val,
                        'n_states': n_scc, 'n_perm': n_perm,
                        'n_perm_skipped': n_skipped,
                        'n_eff_min': int(n_eff_min) if n_eff_min is not None else None,
                    }
                    logger.info(
                        f"A4: MFPT ↔ FC dissimilarity rho={rho_obs:.3f}, p={p_val:.4f} "
                        f"(n_pairs_obs={int(valid.sum())}, n_perm_skipped={n_skipped}, n_eff_min={n_eff_min})"
                    )

        with open(os.path.join(out_dir, 'mfpt_fc_correlation.json'), 'w') as f:
            json.dump(mfpt_fc_result, f, indent=2)
    else:
        logger.warning("A4: MFPT computation failed — skipping landscape plots")

    # ===================== Summary =====================
    summary = {
        'sub_id': sub_id,
        'parcellation': parc,
        'vt': args.vt,
        'n_active_states': len(active_states),
        'A1_n_nodes': G.number_of_nodes(),
        'A1_n_edges': G.number_of_edges(),
        'A1_n_communities': len(set(communities.values())),
        'A2_bidirectionality': bidir,
        'A2_mean_concentration_ratio': float(df_select['concentration_ratio'].mean()),
        'A3_assortativity': assort_result,
        'A3_fc_transition': fc_result,
        'A3_network_homophily': homophily_result,
        'A4_scc_size': len(scc_states) if scc_states else 0,
        'A4_mfpt_fc': mfpt_fc_result if mfpt is not None else None,
    }
    with open(os.path.join(out_dir, 'transition_structure_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("=" * 50)
    logger.info(f"Done! All outputs saved to {out_dir}")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
