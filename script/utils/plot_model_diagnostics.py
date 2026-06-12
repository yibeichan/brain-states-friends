#!/usr/bin/env python3
"""
plot_model_diagnostics.py - Post-hoc model diagnostic plots from pickled HDP-HMM models.

Generates diagnostic figures B1–B5 from already-computed outputs (no re-fitting):

  B1. LL Convergence Traces    - per-seed log-likelihood per EM iteration
  B2. Active State Count       - violin/strip of n_active_states across final seeds
  B3. Covariance Health        - eigenvalue spectra and condition numbers
  B4. Seed Stability           - Hungarian-matched Pearson r heatmap across seeds
  B5. Per-Season Test LL       - bar plot from final_results.json

Prerequisites:
    - 04_combined_hdphmm.py mode=select completed for this subject
    - Outputs at {SCRATCH_DIR}/output/04_combined_hdphmm/{parcellation}/{sub_id}/

Usage:
    python script/utils/plot_model_diagnostics.py \\
        --sub_id sub-01 --parcellation atlas-4S156Parcels

    # Only specific plots:
    python script/utils/plot_model_diagnostics.py \\
        --sub_id sub-01 --plots B1 B4 B5
"""

import os
import sys
import json
import pickle
import logging
import argparse
import re
from pathlib import Path
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plot_style import apply_publication_style
from utils.common import normalize_parcellation_name

from dotenv import load_dotenv
load_dotenv()

SCRATCH_DIR = os.getenv('SCRATCH_DIR')
if SCRATCH_DIR is None:
    raise ValueError("SCRATCH_DIR must be set in .env")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

apply_publication_style()


# =============================================================================
# Path helpers
# =============================================================================

def get_output_base(sub_id, parcellation, output_root=None):
    """Resolve root output dir for 04 combined HMM results.

    Priority:
      1. Explicit --output_root if provided
      2. Auto-detect best root based on available artifacts:
         - `final/seeds/seed_*_model.pkl`
         - `final/final_results.json`
      3. Fallback to existing directory preference
      4. Default to 04_combined_hdphmm path
    """
    if output_root is not None:
        return os.path.join(SCRATCH_DIR, 'output', output_root, parcellation, sub_id)

    v2_path = os.path.join(
        SCRATCH_DIR, 'output', '04_combined_hdphmm', parcellation, sub_id
    )
    legacy_path = os.path.join(
        SCRATCH_DIR, 'output', '04_combined_hdphmm', parcellation, sub_id
    )

    def _artifact_score(base_path):
        final_dir = os.path.join(base_path, 'final')
        seeds_dir = os.path.join(final_dir, 'seeds')
        has_results = os.path.isfile(os.path.join(final_dir, 'final_results.json'))
        n_seed_pickles = len(glob(os.path.join(seeds_dir, 'seed_*_model.pkl')))
        score = 0
        if has_results:
            score += 1
        if n_seed_pickles > 0:
            score += 2
        return score, has_results, n_seed_pickles

    v2_score, v2_has_results, v2_n_seeds = _artifact_score(v2_path)
    legacy_score, legacy_has_results, legacy_n_seeds = _artifact_score(legacy_path)

    if (v2_score > 0) or (legacy_score > 0):
        if v2_score >= legacy_score:
            logger.info(
                f"Auto-selected output root '04_combined_hdphmm' "
                f"(seed_pickles={v2_n_seeds}, final_results={v2_has_results})"
            )
            return v2_path
        logger.info(
            f"Auto-selected output root '04_combined_hdphmm' "
            f"(seed_pickles={legacy_n_seeds}, final_results={legacy_has_results})"
        )
        return legacy_path

    if os.path.isdir(v2_path):
        return v2_path
    if os.path.isdir(legacy_path):
        logger.warning(
            'Using legacy output root 04_combined_hdphmm (v2 directory not found).'
        )
        return legacy_path
    return v2_path


def _parse_seed_idx_from_path(path):
    """Extract integer seed index from seed model pickle path."""
    match = re.search(r'seed_(\d+)_model\.pkl$', os.path.basename(path))
    if match is None:
        return None
    return int(match.group(1))


def load_final_seed_models(output_base, n_seeds=None, final_dir=None):
    """Load final seed model pickles sequentially; extract only needed attributes.

    Loads each pickle, copies the needed arrays, then deletes the model object to
    avoid holding all models in memory simultaneously.

    Args:
        output_base: Root output directory for the subject.
        n_seeds: Max number of seeds to load (default: all).
        final_dir: Override for final directory path. If None, uses
            ``{output_base}/final/``.

    Returns:
        list of dicts with keys:
            seed, means, covars, history, covariance_type, n_components, min_state_usage
    """
    if final_dir is None:
        final_dir = os.path.join(output_base, 'final')
    seeds_dir = os.path.join(final_dir, 'seeds')
    unsorted_paths = glob(os.path.join(seeds_dir, 'seed_*_model.pkl'))

    parsed = []
    for path in unsorted_paths:
        seed_idx = _parse_seed_idx_from_path(path)
        if seed_idx is None:
            logger.warning(f"Skipping unrecognized seed filename: {Path(path).name}")
            continue
        parsed.append((seed_idx, path))

    parsed.sort(key=lambda item: item[0])
    seed_items = parsed

    if not seed_items:
        logger.warning(f"No seed model pickles found in {seeds_dir}")
        return []
    if n_seeds is not None:
        seed_items = seed_items[:n_seeds]

    records = []
    for seed_idx, path in seed_items:
        logger.info(f"  Loading seed {seed_idx}: {Path(path).name}")

        with open(path, 'rb') as f:
            model = pickle.load(f)

        history = getattr(model, 'history', {}) or {}

        record = {
            'seed': seed_idx,
            'means': model.means_.copy(),          # (K, D)
            'covars': model.covars_.copy(),         # (K, D, D) full | (K, D) diag
            'covariance_type': model.covariance_type,
            'n_components': model.n_components,
            'min_state_usage': float(getattr(model, 'min_state_usage', 0.01)),
            'history': {
                'log_likelihood': list(history.get('log_likelihood', [])),
                'active_states': list(history.get('active_states', [])),
                'state_usage': [
                    arr.tolist() if hasattr(arr, 'tolist') else list(arr)
                    for arr in history.get('state_usage', [])
                ],
            },
        }
        del model  # release memory before loading next pickle
        records.append(record)

    logger.info(f"Loaded {len(records)} seed model records")
    return records


# =============================================================================
# B1: LL Convergence Traces
# =============================================================================

def plot_ll_convergence(seed_records, output_dir):
    """Plot per-seed LL per EM iteration (multiple curves, same axes)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.tab10
    any_plotted = False

    for i, rec in enumerate(seed_records):
        ll_trace = rec['history']['log_likelihood']
        if not ll_trace:
            logger.warning(f"  Seed {rec['seed']}: empty LL history - skipping")
            continue
        ax.plot(ll_trace, color=cmap(i % 10), alpha=0.85,
                label=f"Seed {rec['seed']}", linewidth=1.5)
        any_plotted = True

    if not any_plotted:
        logger.warning("B1: No LL traces available to plot")
        plt.close(fig)
        return

    ax.set_xlabel('EM Iteration')
    ax.set_ylabel('Log-Likelihood')
    ax.set_title('B1. LL Convergence Traces (Final Seeds)')
    ax.legend(loc='lower right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, 'B1_ll_convergence.png')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# =============================================================================
# B2: Active State Count Stability
# =============================================================================

def _compute_n_active(rec):
    """Extract n_active_states from a model record.

    Priority:
      1. model.history['active_states'][-1]  - count of states above threshold per iter
      2. Recompute from model.history['state_usage'][-1]
      3. Return -1 if neither is available
    """
    active_hist = rec['history']['active_states']
    if active_hist:
        return int(active_hist[-1])

    usage_hist = rec['history']['state_usage']
    if usage_hist:
        last_usage = np.array(usage_hist[-1])
        return int(np.sum(last_usage > rec['min_state_usage']))

    return -1


def plot_active_state_count(seed_records, output_dir):
    """Violin + strip plot of n_active_states across seeds."""
    n_active_list = []
    for rec in seed_records:
        n = _compute_n_active(rec)
        if n >= 0:
            n_active_list.append(n)
        else:
            logger.warning(f"  Seed {rec['seed']}: no active-state history, skipping")

    if not n_active_list:
        logger.warning("B2: No n_active_states data available")
        return

    fig, ax = plt.subplots(figsize=(5, 5))

    if len(n_active_list) >= 2:
        ax.violinplot(n_active_list, positions=[0], widths=0.5, showmedians=True)
    else:
        # Only 1 point - violin would fail
        ax.axhline(y=n_active_list[0], color='steelblue', linewidth=2)

    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.1, 0.1, size=len(n_active_list))
    ax.scatter(jitter, n_active_list, color='steelblue', alpha=0.85, s=70, zorder=3)

    ax.set_xticks([0])
    ax.set_xticklabels(['Final Seeds'])
    ax.set_ylabel('N Active States')
    ax.set_title(f'B2. Active State Count Across {len(n_active_list)} Seeds\n'
                 f'Mean = {np.mean(n_active_list):.1f} ± {np.std(n_active_list):.1f}')
    ax.grid(True, axis='y', alpha=0.3)

    out_path = os.path.join(output_dir, 'B2_active_state_count.png')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# =============================================================================
# B3: Covariance Health - Eigenvalue Spectra
# =============================================================================

def _eigenvalues_for_record(rec, active_mask=None):
    """Compute eigenvalues and condition numbers per active state.

    Handles both full covariance (K, D, D) and diagonal covariance (K, D):
    - Full:     eigvalsh gives all eigenvalues in ascending order
    - Diagonal: diagonal values ARE the eigenvalues

    Returns:
        min_eigvals, max_eigvals, cond_nums: np.array(n_active,)
        active_state_ids: np.array(n_active,), indices into [0, K)
    """
    covars = rec['covars']
    K = covars.shape[0]
    cov_type = rec['covariance_type']

    if active_mask is None:
        active_mask = np.ones(K, dtype=bool)
    active_ids = np.where(active_mask)[0]

    min_eigvals, max_eigvals, cond_nums = [], [], []

    for k in active_ids:
        if cov_type == 'full':
            try:
                eigvals = np.linalg.eigvalsh(covars[k])  # ascending
                min_e, max_e = float(eigvals[0]), float(eigvals[-1])
            except np.linalg.LinAlgError:
                logger.warning(f"  eigvalsh failed for state {k}")
                min_e, max_e = np.nan, np.nan
        elif cov_type == 'diag':
            diag = covars[k]
            min_e, max_e = float(np.min(diag)), float(np.max(diag))
        else:
            logger.warning(f"  Unsupported covariance_type '{cov_type}' for state {k}")
            min_e, max_e = np.nan, np.nan

        # Condition number = max / |min|, with explicit floor to avoid /0
        denom = max(abs(min_e), 1e-10) if np.isfinite(min_e) else 1e-10
        cond = (max_e / denom) if np.isfinite(max_e) else np.nan

        min_eigvals.append(min_e)
        max_eigvals.append(max_e)
        cond_nums.append(cond)

    return (np.array(min_eigvals), np.array(max_eigvals),
            np.array(cond_nums), active_ids)


def plot_covariance_health(seed_records, output_dir):
    """Plot eigenvalue spectra for the best-fit seed (longest LL history)."""
    if not seed_records:
        logger.warning("B3: No seed records")
        return

    # Use seed with most EM iterations as the reference
    best_rec = max(seed_records, key=lambda r: len(r['history']['log_likelihood']))
    logger.info(f"B3: Using seed {best_rec['seed']} as reference "
                f"({len(best_rec['history']['log_likelihood'])} iters)")

    # Determine active states from last state_usage snapshot
    usage_hist = best_rec['history']['state_usage']
    if usage_hist:
        last_usage = np.array(usage_hist[-1])
        active_mask = last_usage > best_rec['min_state_usage']
    else:
        logger.warning("B3: No state_usage history; treating all states as active")
        active_mask = np.ones(best_rec['n_components'], dtype=bool)

    n_active = int(active_mask.sum())
    logger.info(f"  Active states: {n_active} / {best_rec['n_components']}")

    min_eigvals, max_eigvals, cond_nums, active_ids = _eigenvalues_for_record(
        best_rec, active_mask
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Panel 1: Condition numbers (log10 scale, violin + strip) ---
    valid_conds = cond_nums[np.isfinite(cond_nums) & (cond_nums > 0)]
    if len(valid_conds) > 0:
        log_conds = np.log10(np.maximum(valid_conds, 1.0))
        if len(log_conds) >= 2:
            axes[0].violinplot(log_conds, positions=[0], widths=0.5, showmedians=True)
        rng = np.random.default_rng(2)
        jitter = rng.uniform(-0.08, 0.08, size=len(log_conds))
        sc = axes[0].scatter(jitter, log_conds,
                             c=log_conds, cmap='RdYlGn_r',
                             vmin=0, vmax=6,
                             alpha=0.75, s=55, zorder=3)
        axes[0].axhline(y=3, color='red', linestyle='--', alpha=0.7,
                        linewidth=1.5, label='cond = 1000 (warning threshold)')
        axes[0].set_xticks([0])
        axes[0].set_xticklabels(['Active States'])
        axes[0].set_ylabel('log₁₀(Condition Number)')
        axes[0].set_title(f'B3a. Condition Numbers\n'
                          f'Seed {best_rec["seed"]}, n_active={n_active}')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, axis='y', alpha=0.3)
        plt.colorbar(sc, ax=axes[0], label='log₁₀(cond)')
    else:
        axes[0].text(0.5, 0.5, 'No valid condition numbers',
                     transform=axes[0].transAxes, ha='center', va='center')
        axes[0].axis('off')

    # --- Panel 2: Min eigenvalue per state (sorted), highlight regularization floor ---
    sort_idx = np.argsort(min_eigvals)
    min_sorted = min_eigvals[sort_idx]
    reg_floor = 1e-3
    hit_floor = min_sorted <= reg_floor * 1.01  # tolerance for float comparison

    if len(min_sorted) > 0:
        bar_colors = np.where(hit_floor, '#d62728', '#1f77b4')
        axes[1].bar(range(len(min_sorted)), min_sorted, color=bar_colors, alpha=0.82)
        axes[1].axhline(y=reg_floor, color='red', linestyle='--', alpha=0.7,
                        linewidth=1.5, label=f'Regularization floor ({reg_floor})')
        axes[1].set_xlabel('State (sorted by min eigenvalue)')
        axes[1].set_ylabel('Min Eigenvalue')
        axes[1].set_title(f'B3b. Min Eigenvalues\n'
                          f'Red bars: at regularization floor ({hit_floor.sum()}/{n_active})')
        axes[1].legend(fontsize=9)
        axes[1].grid(True, axis='y', alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No eigenvalue data', transform=axes[1].transAxes,
                     ha='center', va='center')
        axes[1].axis('off')

    fig.suptitle(f'B3. Covariance Health - Eigenvalue Spectra\n'
                 f'({best_rec["covariance_type"]} covariance)',
                 fontsize=13)
    out_path = os.path.join(output_dir, 'B3_covariance_health.png')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# =============================================================================
# B4: Seed Stability - Hungarian-Matched Pearson r
# =============================================================================

def _pearson_r_matrix(A, B):
    """Vectorized Pearson r between all row pairs of A (K, D) and B (K, D).

    Equivalent to computing corr(A[i], B[j]) for all i, j.
    Zero-variance rows (constant means) yield 0.0 in the correlation matrix
    so they are excluded from matching via the -1 nan_to_num penalty.

    Returns:
        corr: (K, K) matrix, values in [-1, 1]
    """
    K = A.shape[0]
    eps = 1e-12

    # Center rows
    A_c = A - A.mean(axis=1, keepdims=True)
    B_c = B - B.mean(axis=1, keepdims=True)

    # L2 norm per row
    A_norm = np.linalg.norm(A_c, axis=1)  # (K,)
    B_norm = np.linalg.norm(B_c, axis=1)  # (K,)

    # Normalize (zero-norm rows → zero vector)
    safe_A = np.where(A_norm[:, None] > eps, A_c / np.maximum(A_norm[:, None], eps), 0.0)
    safe_B = np.where(B_norm[:, None] > eps, B_c / np.maximum(B_norm[:, None], eps), 0.0)

    corr = safe_A @ safe_B.T  # (K, K); dot of normalized centered vectors = Pearson r
    corr = np.clip(corr, -1.0, 1.0)  # numerical guard

    # Mark zero-variance rows as NaN so Hungarian avoids matching them
    zero_A = A_norm < eps
    zero_B = B_norm < eps
    corr[zero_A, :] = np.nan
    corr[:, zero_B] = np.nan

    return corr


def _hungarian_match(means_i, means_j):
    """Hungarian matching of states across two seeds by Pearson r.

    Returns:
        matched_r: float, mean Pearson r of matched pairs (NaN if no valid match)
    """
    corr = _pearson_r_matrix(means_i, means_j)
    # Replace NaN with -1 so Hungarian avoids matching invalid states
    cost = -np.nan_to_num(corr, nan=-1.0)
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_values = corr[row_ind, col_ind]
    finite = matched_values[np.isfinite(matched_values)]
    if finite.size == 0:
        return np.nan
    matched_r = float(finite.mean())
    return matched_r


def plot_seed_stability(seed_records, output_dir):
    """n_seeds × n_seeds heatmap of pairwise Hungarian-matched Pearson r."""
    n = len(seed_records)
    if n < 2:
        logger.warning("B4: Need at least 2 seeds for stability heatmap")
        return

    match_matrix = np.full((n, n), np.nan)
    np.fill_diagonal(match_matrix, 1.0)

    for i in range(n):
        for j in range(i + 1, n):
            r = _hungarian_match(seed_records[i]['means'], seed_records[j]['means'])
            match_matrix[i, j] = r
            match_matrix[j, i] = r
            logger.info(
                f"  Seeds {seed_records[i]['seed']} ↔ {seed_records[j]['seed']}: r = {r:.3f}"
            )

    offdiag = match_matrix[~np.eye(n, dtype=bool)]
    finite_offdiag = offdiag[np.isfinite(offdiag)]
    mean_offdiag = float(finite_offdiag.mean()) if finite_offdiag.size > 0 else np.nan
    seed_labels = [f"S{rec['seed']}" for rec in seed_records]

    fig, ax = plt.subplots(figsize=(max(6, n + 1), max(5, n)))
    im = ax.imshow(match_matrix, vmin=0.0, vmax=1.0, cmap='RdYlGn', aspect='auto')
    plt.colorbar(im, ax=ax, label='Mean Matched Pearson r')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(seed_labels, rotation=45, ha='right')
    ax.set_yticklabels(seed_labels)

    for i in range(n):
        for j in range(n):
            val = match_matrix[i, j]
            if np.isfinite(val):
                txt_color = 'black' if val > 0.5 else 'white'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=8, color=txt_color)

    mean_text = f'{mean_offdiag:.3f}' if np.isfinite(mean_offdiag) else 'N/A'
    ax.set_title(f'B4. Seed Stability (Hungarian-Matched Pearson r)\n'
                 f'Mean off-diagonal r = {mean_text} '
                 f'(>0.80 = stable, <0.50 = unstable)')
    ax.set_xlabel('Seed')
    ax.set_ylabel('Seed')

    out_path = os.path.join(output_dir, 'B4_seed_stability.png')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# =============================================================================
# B5: Per-Season Test LL
# =============================================================================

def plot_test_ll_per_season(final_results, output_dir):
    """Bar plot of test LL/sample per season from final_results.json."""
    test_ll_per_season = (
        final_results.get('final_refit', {}).get('test_ll_per_season', {})
    )
    if not test_ll_per_season:
        logger.warning("B5: 'test_ll_per_season' not found in final_results.json")
        return

    seasons = sorted(test_ll_per_season.keys())
    lls = [test_ll_per_season[s] for s in seasons]
    overall_ll = final_results.get('final_refit', {}).get('test_ll_per_sample')

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(seasons)))
    bars = ax.bar(seasons, lls, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=0.7)

    for bar, ll in zip(bars, lls):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + abs(max(lls) - min(lls)) * 0.01,
                f'{ll:.4f}', ha='center', va='bottom', fontsize=9)

    if overall_ll is not None:
        ax.axhline(y=overall_ll, color='red', linestyle='--', alpha=0.8, linewidth=1.5,
                   label=f'Overall test LL/s = {overall_ll:.4f}')
        ax.legend(fontsize=10)

    ax.set_xlabel('Season')
    ax.set_ylabel('Test LL / Sample (held-out test episodes)')
    ax.set_title('B5. Per-Season Test Log-Likelihood\n'
                 '(Higher = better generalization to held-out episodes in that season)')
    ax.grid(True, axis='y', alpha=0.3)

    out_path = os.path.join(output_dir, 'B5_test_ll_per_season.png')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Post-hoc model diagnostics from HDP-HMM pickles.'
    )
    parser.add_argument('--sub_id', type=str, required=True,
                        help='Subject ID (e.g., sub-01)')
    parser.add_argument('--parcellation', type=str, default='atlas-4S156Parcels',
                        help='Parcellation name (default: atlas-4S156Parcels)')
    parser.add_argument('--output_root', type=str, default=None,
                        help='04 output root under SCRATCH_DIR/output '
                             '(e.g., 04_combined_hdphmm)')
    parser.add_argument('--n_seeds', type=int, default=None,
                        help='Max number of seeds to load (default: all available)')
    parser.add_argument('--plots', nargs='+',
                        default=['B1', 'B2', 'B3', 'B4', 'B5'],
                        choices=['B1', 'B2', 'B3', 'B4', 'B5'],
                        help='Which diagnostic plots to generate (default: all)')
    parser.add_argument('--vt', type=str, default=None,
                        help='Variance threshold subdirectory under final/ (e.g., 0.99). '
                             'Reads from final/vt{VT}/. If omitted, reads from final/ directly '
                             '(legacy path).')
    return parser.parse_args()


def main():
    args = parse_args()
    parc = normalize_parcellation_name(args.parcellation)
    output_base = get_output_base(args.sub_id, parc, output_root=args.output_root)
    if args.vt is not None:
        final_dir = os.path.join(output_base, 'final', f'vt{args.vt}')
    else:
        final_dir = os.path.join(output_base, 'final')

    out_dir = os.path.join(SCRATCH_DIR, 'output', 'diagnostics', parc, args.sub_id)
    os.makedirs(out_dir, exist_ok=True)

    logger.info('=' * 60)
    logger.info(f'MODEL DIAGNOSTICS: {args.sub_id} / {parc}')
    logger.info(f'Plots requested: {args.plots}')
    logger.info(f'Output: {out_dir}')
    logger.info('=' * 60)

    # Load final_results.json (needed for B5; harmless if absent)
    final_results_path = os.path.join(final_dir, 'final_results.json')
    final_results = {}
    if os.path.exists(final_results_path):
        with open(final_results_path) as f:
            final_results = json.load(f)
    else:
        logger.warning(f'final_results.json not found: {final_results_path}')

    # Load model pickles for B1–B4
    need_models = set(args.plots) & {'B1', 'B2', 'B3', 'B4'}
    seed_records = []
    if need_models:
        logger.info('Loading final seed model pickles...')
        seed_records = load_final_seed_models(output_base, n_seeds=args.n_seeds,
                                                final_dir=final_dir)
        if not seed_records:
            logger.error(
                f'No seed model pickles found under {output_base}/final/seeds/. '
                'Run 04_combined_hdphmm.py --mode select first.'
            )

    if 'B1' in args.plots:
        logger.info('--- B1: LL Convergence Traces ---')
        plot_ll_convergence(seed_records, out_dir)

    if 'B2' in args.plots:
        logger.info('--- B2: Active State Count Stability ---')
        plot_active_state_count(seed_records, out_dir)

    if 'B3' in args.plots:
        logger.info('--- B3: Covariance Health ---')
        plot_covariance_health(seed_records, out_dir)

    if 'B4' in args.plots:
        logger.info('--- B4: Seed Stability (Hungarian Matching) ---')
        plot_seed_stability(seed_records, out_dir)

    if 'B5' in args.plots:
        logger.info('--- B5: Per-Season Test LL ---')
        plot_test_ll_per_season(final_results, out_dir)

    logger.info(f'\nAll requested diagnostics saved to: {out_dir}')


if __name__ == '__main__':
    main()
