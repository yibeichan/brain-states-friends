#!/usr/bin/env python3
"""
05d_state_similarity.py - Assess distinctness of discovered brain states (heuristic diagnostic).

This is a review aid for manual inspection and threshold sensitivity checks,
NOT a pass/fail test for HMM correctness or a basis for biological labels.

Computes pairwise similarity across three complementary metrics and flags
high-similarity pairs for inspection:

1. Activation similarity:   Pearson correlation of state mean vectors.
2. Transition similarity:   Pearson correlation of outgoing transition rows.
3. Combined similarity:     Mean of [0,1]-normalised activation + transition (heuristic).
4. Flagged-pair diagnosis:  Episode-level overlap.

Episode-level aggregation: overlap metrics use episodes (not runs) as the unit of
observation after multipart aggregation from 05a.

FC similarity was removed (see 05f_state_fc.py for empirical state-conditioned FC).

Prerequisites:
    - 04_combined_hdphmm.py (mode: select) completed
    - state_means_parcel.npy, best_model.pkl, decoded_states.pkl
    - 05a_recurrence_analysis.py completed (recurrence_summary.json, fractional_occupancy.pkl)

Outputs:
    Saves similarity matrices, flagged pairs, summary JSON and diagnostic plots to:
    {SCRATCH_DIR}/output/05d_state_similarity/{parcellation}/{sub_id}/
"""

import os
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths and logger
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from utils.plot_style import recurrence_color, make_recurrence_colorbar, apply_publication_style
from utils.common import normalize_parcellation_name, get_episode_base
from utils.state_blocks import load_eligible_states

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


# ===================================================================
# Similarity metrics
# ===================================================================

def compute_activation_similarity(means: np.ndarray) -> np.ndarray:
    """Pairwise Pearson correlation of state mean vectors.

    Parameters
    ----------
    means : (K, n_parcels)

    Returns
    -------
    sim : (K, K) correlation matrix with diagonal = 1.
    """
    K = means.shape[0]
    # Centre each row then correlate
    centered = means - means.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = centered / norms
    sim = normed @ normed.T
    # Clip to [-1, 1] for numerical safety
    return np.clip(sim, -1.0, 1.0)


def compute_transition_similarity(transmat: np.ndarray) -> np.ndarray:
    """Pearson correlation of outgoing transition probability rows.

    Parameters
    ----------
    transmat : (K, K) row-stochastic transition matrix.

    Returns
    -------
    sim : (K, K) correlation matrix.
    """
    K = transmat.shape[0]
    centered = transmat - transmat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = centered / norms
    sim = normed @ normed.T
    return np.clip(sim, -1.0, 1.0)


def normalize_to_01(r: np.ndarray) -> np.ndarray:
    """Map Pearson r in [-1, 1] to [0, 1] via (1+r)/2."""
    return (1.0 + r) / 2.0


def compute_combined_similarity(
    act_sim: np.ndarray,
    trans_sim: np.ndarray,
) -> np.ndarray:
    """Average of [0,1]-normalised activation and transition similarities.

    Activation and transition correlations are mapped via (1+r)/2.
    """
    act_01 = normalize_to_01(act_sim)
    trans_01 = normalize_to_01(trans_sim)
    combined = (act_01 + trans_01) / 2.0
    return combined


# ===================================================================
# Flagged-pair diagnosis
# ===================================================================

def compute_episode_sets(decoded_states: dict, n_states: int) -> dict:
    """Return {state_id: set(episode_bases)} for each state.

    Aggregates at the episode level (multipart runs -> single episode).
    """
    episode_sets = {s: set() for s in range(n_states)}
    for run_id, seq in decoded_states.items():
        ep_base = get_episode_base(run_id)
        for s in np.unique(seq):
            episode_sets[int(s)].add(ep_base)
    return episode_sets


def compute_episode_overlap_jaccard(episode_sets: dict, n_states: int) -> np.ndarray:
    """Pairwise Jaccard overlap of episode sets (binary co-occurrence).

    J(i, j) = |episodes_i ∩ episodes_j| / |episodes_i ∪ episodes_j|

    Returns (K, K) matrix in [0, 1].
    """
    K = n_states
    jaccard = np.zeros((K, K))
    for i in range(K):
        for j in range(i, K):
            eps_i = episode_sets.get(i, set())
            eps_j = episode_sets.get(j, set())
            union = eps_i | eps_j
            if len(union) > 0:
                jaccard[i, j] = len(eps_i & eps_j) / len(union)
            jaccard[j, i] = jaccard[i, j]
    return jaccard


def compute_fo_weighted_overlap(fo_dict: dict, n_states: int) -> np.ndarray:
    """FO-weighted overlap: continuous generalization of Jaccard.

    fo_weighted_overlap[i,j] = sum_e min(FO_i[e], FO_j[e]) / sum_e max(FO_i[e], FO_j[e])

    Captures whether co-occurring states have similar occupancy strength,
    not just binary co-occurrence.

    Args:
        fo_dict: dict episode_id -> np.array(n_states,) (episode-level FO)
        n_states: number of HMM states

    Returns:
        (K, K) matrix in [0, 1].
    """
    K = n_states
    fo_matrix = np.stack(list(fo_dict.values()))  # (n_episodes, K)
    # Vectorized: (n_ep, K, 1) vs (n_ep, 1, K) → (K, K) via sum over episodes
    fo_i = fo_matrix[:, :, None]  # (n_ep, K, 1)
    fo_j = fo_matrix[:, None, :]  # (n_ep, 1, K)
    num = np.sum(np.minimum(fo_i, fo_j), axis=0)  # (K, K)
    den = np.sum(np.maximum(fo_i, fo_j), axis=0)  # (K, K)
    overlap = np.divide(num, den, out=np.zeros((K, K)), where=den > 0)
    return overlap


def diagnose_flagged_pairs(
    flagged: list,
    episode_sets: dict,
    recurrence_scores: np.ndarray,
    overlap_threshold: float = 0.50,
) -> list:
    """For each flagged pair determine likely explanation.

    Logic:
    - High episode overlap -> possible split state
    - Low episode overlap -> distinct states

    Flagged pairs are for manual inspection, not automatic merging or pruning.
    For genuine FC-based diagnosis, see 05f_state_fc.py.
    """
    diagnosed = []
    for pair in flagged:
        i, j = pair['state_i'], pair['state_j']
        eps_i = episode_sets.get(i, set())
        eps_j = episode_sets.get(j, set())
        union = eps_i | eps_j
        overlap = len(eps_i & eps_j) / max(len(union), 1)

        if overlap >= overlap_threshold:
            explanation = 'possible_split'
        else:
            explanation = 'distinct_states'

        diagnosed.append({
            **pair,
            'recurrence_score_i': round(float(recurrence_scores[i]), 4),
            'recurrence_score_j': round(float(recurrence_scores[j]), 4),
            'episode_overlap_jaccard': round(overlap, 4),
            'diagnosis': explanation,
        })
    return diagnosed


# ===================================================================
# Plotting helpers
# ===================================================================

def plot_combined_heatmap(combined: np.ndarray, recurrence_scores: np.ndarray, out_dir: str):
    """Heatmap of the combined similarity matrix ordered by descending recurrence score."""
    K = combined.shape[0]
    order = sorted(range(K), key=lambda s: -recurrence_scores[s])

    mat = combined[np.ix_(order, order)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 8),
                              gridspec_kw={'width_ratios': [20, 1]})
    ax = axes[0]
    im = ax.imshow(mat, vmin=0, vmax=1, cmap='viridis', aspect='equal')
    fig.colorbar(im, ax=ax, label='Combined Similarity')

    ax.set_xlabel('State')
    ax.set_ylabel('State')
    ax.set_title('Combined State Similarity Matrix (ordered by recurrence score)')

    # Recurrence score margin colorbar
    ax_margin = axes[1]
    scores_ordered = np.array([recurrence_scores[s] for s in order])
    ax_margin.imshow(
        scores_ordered[:, None], aspect='auto', cmap='viridis', vmin=0, vmax=1,
        interpolation='nearest',
    )
    ax_margin.set_xticks([])
    ax_margin.set_yticks([])
    ax_margin.set_title('Rec.\nscore', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'combined_similarity_heatmap.png'))
    plt.savefig(os.path.join(out_dir, 'combined_similarity_heatmap.pdf'))
    plt.close(fig)


def plot_dendrogram(combined: np.ndarray, recurrence_scores: np.ndarray, out_dir: str):
    """Dendrogram of states using 1-combined similarity as distance."""
    import matplotlib.colors as mcolors

    K = combined.shape[0]
    # Convert similarity to distance, ensure non-negative
    dist_mat = np.clip(1.0 - combined, 0, None)
    np.fill_diagonal(dist_mat, 0.0)

    # Make symmetric (should already be, but guard against float drift)
    dist_mat = (dist_mat + dist_mat.T) / 2.0

    condensed = squareform(dist_mat, checks=False)
    Z = linkage(condensed, method='average')

    # Leaf colours by recurrence score (continuous)
    leaf_colors = {
        i: mcolors.to_hex(recurrence_color(recurrence_scores[i]))
        for i in range(K)
    }

    def _color_func(k):
        # scipy leaf IDs below K are original observations
        if k < K:
            return leaf_colors.get(k, '#999999')
        return '#333333'

    fig_height = max(6, K * 0.3)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    dendrogram(
        Z,
        orientation='left',
        labels=[f'S{i}' for i in range(K)],
        leaf_font_size=max(6, min(10, 200 // K)),
        link_color_func=_color_func,
        ax=ax,
    )
    ax.set_xlabel('Distance (1 - Combined Similarity)')
    ax.set_title('State Dendrogram by Combined Similarity')
    make_recurrence_colorbar(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'state_dendrogram.png'))
    plt.savefig(os.path.join(out_dir, 'state_dendrogram.pdf'))
    plt.close(fig)


def plot_flagged_pair_summary(diagnosed: list, act_sim: np.ndarray,
                               trans_sim: np.ndarray, out_dir: str):
    """Table-like summary figure for flagged high-similarity pairs."""
    if not diagnosed:
        logger.info("No flagged pairs to plot.")
        return

    n_pairs = len(diagnosed)
    fig, ax = plt.subplots(figsize=(12, max(3, 0.5 * n_pairs + 2)))
    ax.axis('off')

    col_labels = ['Pair', 'Rec_i', 'Rec_j', 'Act r', 'Trans r',
                  'Combined', 'Ep Overlap', 'Diagnosis']
    table_data = []
    for d in diagnosed:
        i, j = d['state_i'], d['state_j']
        table_data.append([
            f'S{i}-S{j}',
            f"{d.get('recurrence_score_i', 0):.2f}",
            f"{d.get('recurrence_score_j', 0):.2f}",
            f"{act_sim[i, j]:.3f}",
            f"{trans_sim[i, j]:.3f}",
            f"{d['combined_similarity']:.3f}",
            f"{d['episode_overlap_jaccard']:.2f}",
            d['diagnosis'],
        ])

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    ax.set_title('Flagged High-Similarity State Pairs', fontsize=13, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'flagged_pairs_summary.png'))
    plt.savefig(os.path.join(out_dir, 'flagged_pairs_summary.pdf'))
    plt.close(fig)


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Assess distinctness of discovered brain states via multi-metric similarity."
    )
    parser.add_argument('--sub_id', type=str, required=True,
                        help="Subject ID (e.g., sub-01)")
    parser.add_argument('--parcellation', type=str, default='atlas-4S156Parcels',
                        help="Parcellation scheme")
    parser.add_argument('--similarity_threshold', type=float, default=0.85,
                        help="Combined similarity threshold for flagging pairs (default 0.85)")
    parser.add_argument('--vt', type=str, default=None,
                        help="Variance threshold subdirectory under final/ (e.g., 0.99). "
                             "Reads from final/vt{VT}/. If omitted, reads from final/ directly "
                             "(legacy path).")
    parser.add_argument('--exclude_sub_hrf', action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Exclude sub-HRF states from flagged-pair analysis "
                             "(default: True). Use --no_exclude_sub_hrf to include all.")
    args = parser.parse_args()

    parc = normalize_parcellation_name(args.parcellation)
    sub_id = args.sub_id
    threshold = args.similarity_threshold

    logger.info("==============================================")
    logger.info("05d - State Similarity Analysis")
    logger.info("==============================================")
    logger.info(f"Subject: {sub_id}, Parcellation: {parc}")
    logger.info(f"Combined similarity threshold: {threshold}")

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    if args.vt is not None:
        hmm_base = os.path.join(SCRATCH_DIR, 'output', '04_combined_hdphmm', parc, sub_id,
                                'final', f'vt{args.vt}')
    else:
        hmm_base = os.path.join(SCRATCH_DIR, 'output', '04_combined_hdphmm', parc, sub_id, 'final')
    means_path = os.path.join(hmm_base, 'state_means_parcel.npy')
    model_path = os.path.join(hmm_base, 'best_model.pkl')
    decoded_path = os.path.join(hmm_base, 'decoded_states.pkl')

    recur_base = os.path.join(SCRATCH_DIR, 'output', '05a_recurrence_analysis', parc, sub_id)
    if args.vt is not None:
        recur_base = os.path.join(recur_base, f'vt{args.vt}')
    summary_path = os.path.join(recur_base, 'recurrence_summary.json')

    out_dir = os.path.join(SCRATCH_DIR, 'output', '05d_state_similarity', parc, sub_id)
    if args.vt is not None:
        out_dir = os.path.join(out_dir, f'vt{args.vt}')
    if not args.exclude_sub_hrf:
        out_dir = os.path.join(out_dir, 'all_states')
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    for label, path in [('state_means_parcel', means_path),
                        ('best_model', model_path),
                        ('decoded_states', decoded_path),
                        ('recurrence_summary', summary_path)]:
        if not os.path.exists(path):
            logger.error(f"Missing {label}: {path}")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logger.info("Loading inputs...")
    means = np.load(means_path)          # (K, n_parcels)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    transmat = model.transmat_.copy()
    del model  # free memory

    with open(decoded_path, 'rb') as f:
        decoded_states = pickle.load(f)

    with open(summary_path, 'r') as f:
        recurrence_summary = json.load(f)

    K = means.shape[0]
    n_parcels = means.shape[1]
    n_states = recurrence_summary['n_states']
    logger.info(f"Loaded {K} states with {n_parcels} parcels (recurrence n_states={n_states})")

    # ------------------------------------------------------------------
    # Shape validation
    # ------------------------------------------------------------------
    if K != n_states:
        logger.error(
            f"State count mismatch: means has {K} states but "
            f"recurrence_summary has n_states={n_states}"
        )
        sys.exit(1)
    if transmat.shape[0] != K or transmat.shape[1] != K:
        logger.error(f"Transition matrix shape {transmat.shape} does not match {K} states")
        sys.exit(1)

    # ------------------------------------------------------------------
    # K < 2 guard
    # ------------------------------------------------------------------
    if K < 2:
        logger.warning(f"Only {K} state(s); pairwise similarity is not meaningful.")
        summary = {
            'sub_id': sub_id,
            'parcellation': parc,
            'n_states': int(K),
            'n_parcels': int(n_parcels),
            'similarity_threshold': threshold,
            'n_flagged_pairs': 0,
            'note': 'K < 2: pairwise analysis skipped',
        }
        with open(os.path.join(out_dir, 'similarity_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Done (K < 2). Outputs saved to {out_dir}")
        return

    # Load recurrence scores (continuous)
    recurrence_scores = np.array(recurrence_summary['recurrence_scores'], dtype=float)

    # ------------------------------------------------------------------
    # Load eligible states (sub-HRF filter)
    # ------------------------------------------------------------------
    excluded_sub_hrf = set()
    if args.exclude_sub_hrf:
        try:
            eligible_ids, excluded_ids, _ = load_eligible_states(recur_base)
            excluded_sub_hrf = set(excluded_ids)
            logger.info(
                "Sub-HRF exclusion ON: %d states excluded from flagged-pair analysis",
                len(excluded_sub_hrf),
            )
        except FileNotFoundError:
            logger.warning(
                "eligible_states.json not found in %s; sub-HRF filtering skipped. "
                "Re-run 05a to generate it.",
                recur_base,
            )

    # ------------------------------------------------------------------
    # Load episode-level FO for overlap metrics
    # ------------------------------------------------------------------
    fo_path = os.path.join(recur_base, 'fractional_occupancy.pkl')
    if os.path.exists(fo_path):
        with open(fo_path, 'rb') as f:
            fo_dict = pickle.load(f)
        logger.info(f"Loaded episode-level FO ({len(fo_dict)} episodes)")
    else:
        fo_dict = None
        logger.warning(f"fractional_occupancy.pkl not found at {fo_path}; "
                       "FO-weighted overlap will be skipped")

    # ------------------------------------------------------------------
    # 1. Activation similarity
    # ------------------------------------------------------------------
    logger.info("Computing activation similarity (Pearson)...")
    act_sim = compute_activation_similarity(means)
    np.save(os.path.join(out_dir, 'activation_similarity.npy'), act_sim)

    n_high_act = int(np.sum(np.triu(act_sim, k=1) > threshold))
    logger.info(f"  Activation pairs with r > {threshold}: {n_high_act}")

    # ------------------------------------------------------------------
    # 2. Transition similarity
    # ------------------------------------------------------------------
    logger.info("Computing transition similarity (Pearson on outgoing rows)...")
    trans_sim = compute_transition_similarity(transmat)
    np.save(os.path.join(out_dir, 'transition_similarity.npy'), trans_sim)

    n_high_trans = int(np.sum(np.triu(trans_sim, k=1) > threshold))
    logger.info(f"  Transition pairs with r > {threshold}: {n_high_trans}")

    # ------------------------------------------------------------------
    # 3. Episode overlap metrics
    # ------------------------------------------------------------------
    logger.info("Computing episode overlap metrics...")
    episode_sets = compute_episode_sets(decoded_states, n_states)
    jaccard_mat = compute_episode_overlap_jaccard(episode_sets, n_states)
    np.save(os.path.join(out_dir, 'episode_overlap_jaccard.npy'), jaccard_mat)

    if fo_dict is not None:
        fo_overlap = compute_fo_weighted_overlap(fo_dict, n_states)
        np.save(os.path.join(out_dir, 'fo_weighted_overlap.npy'), fo_overlap)
        logger.info("  Saved episode_overlap_jaccard and fo_weighted_overlap")
    else:
        fo_overlap = None
        logger.info("  Saved episode_overlap_jaccard (FO-weighted skipped)")

    # ------------------------------------------------------------------
    # 4. Combined similarity (heuristic)
    # ------------------------------------------------------------------
    logger.info("Computing heuristic combined similarity...")
    combined = compute_combined_similarity(act_sim, trans_sim)
    np.save(os.path.join(out_dir, 'heuristic_combined_similarity.npy'), combined)

    # Flag pairs above threshold (upper triangle only)
    flagged = []
    n_skipped_sub_hrf = 0
    for i in range(K):
        for j in range(i + 1, K):
            c_val = combined[i, j]
            if np.isfinite(c_val) and c_val > threshold:
                if i in excluded_sub_hrf or j in excluded_sub_hrf:
                    n_skipped_sub_hrf += 1
                    continue
                _safe = lambda v: round(float(v), 4) if np.isfinite(v) else None
                flagged.append({
                    'state_i': int(i),
                    'state_j': int(j),
                    'combined_similarity': _safe(c_val),
                    'activation_r': _safe(act_sim[i, j]),
                    'transition_r': _safe(trans_sim[i, j]),
                })
    if n_skipped_sub_hrf:
        logger.info("  Skipped %d flagged pairs involving sub-HRF states", n_skipped_sub_hrf)

    logger.info(f"  Flagged pairs (combined > {threshold}): {len(flagged)}")

    # ------------------------------------------------------------------
    # 5. Diagnose flagged pairs
    # ------------------------------------------------------------------
    diagnosed = diagnose_flagged_pairs(flagged, episode_sets, recurrence_scores)

    # Save flagged pairs
    with open(os.path.join(out_dir, 'flagged_pairs.json'), 'w') as f:
        json.dump(diagnosed, f, indent=2)

    # ------------------------------------------------------------------
    # Summary JSON
    # ------------------------------------------------------------------
    # Compute mean over strict upper triangle (real pairwise entries only)
    triu_indices = np.triu_indices(K, k=1)
    n_pairs = len(triu_indices[0])
    triu_vals = combined[triu_indices]
    mean_combined = float(np.nanmean(triu_vals)) if n_pairs > 0 else 0.0
    max_combined = float(np.nanmax(triu_vals)) if n_pairs > 0 else 0.0

    summary = {
        'sub_id': sub_id,
        'parcellation': parc,
        'n_states': int(K),
        'n_parcels': int(n_parcels),
        'similarity_threshold': threshold,
        'n_flagged_pairs': len(flagged),
        'n_possible_split': sum(1 for d in diagnosed if d['diagnosis'] == 'possible_split'),
        'n_distinct': sum(1 for d in diagnosed if d['diagnosis'] == 'distinct_states'),
        'activation_pairs_above_threshold': n_high_act,
        'transition_pairs_above_threshold': n_high_trans,
        'mean_heuristic_combined_similarity': round(mean_combined, 4),
        'max_heuristic_combined_similarity': round(max_combined, 4),
        'n_pairwise_comparisons': n_pairs,
        'combined_metric_note': (
            'heuristic_combined_similarity is the mean of [0,1]-normalised '
            'activation and transition similarities. FC was removed - see '
            '05f_state_fc.py for empirical state-conditioned FC.'
        ),
        'exclude_sub_hrf': args.exclude_sub_hrf,
        'n_excluded_sub_hrf_states': len(excluded_sub_hrf),
        'excluded_sub_hrf_states': sorted(excluded_sub_hrf),
        'n_skipped_flagged_pairs_sub_hrf': n_skipped_sub_hrf,
    }
    with open(os.path.join(out_dir, 'similarity_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary: {json.dumps(summary, indent=2)}")

    # ------------------------------------------------------------------
    # Diagnostic plots
    # ------------------------------------------------------------------
    logger.info("Generating diagnostic plots...")

    plot_combined_heatmap(combined, recurrence_scores, out_dir)
    logger.info("  Saved combined_similarity_heatmap")

    plot_dendrogram(combined, recurrence_scores, out_dir)
    logger.info("  Saved state_dendrogram")

    plot_flagged_pair_summary(diagnosed, act_sim, trans_sim, out_dir)
    logger.info("  Saved flagged_pairs_summary")

    logger.info(f"Done! Outputs saved to {out_dir}")
    logger.info("==============================================")


if __name__ == '__main__':
    main()
