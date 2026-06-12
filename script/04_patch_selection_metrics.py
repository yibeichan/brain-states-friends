#!/usr/bin/env python3
"""
04_patch_selection_metrics.py - Compare model selection metrics across HDP-HMM configs.

Reads all config_summary.json files for a given subject + vt, computes a battery
of selection metrics beyond BIC, and produces a CSV + JSON + multi-panel figure
for side-by-side comparison.

No model pickles are loaded - only JSON files. Fast, runs on a login node.

Metrics computed (all "higher is better"):
  M1. valid_ll           - Raw validation LL/sample (baseline reference)
  M2. bic                - BIC on training LL with effective-K (existing 04 logic)
  M3. reflected_valid    - 2*valid_ll - train_ll, penalizes overfitting (0 free params)
  M4. ll_per_active      - valid_ll / K_active, rewards efficiency per state
  M5. gap_penalized_ll   - valid_ll - λ*(train_ll - valid_ll), tunable λ
  M6. season_worst       - min(valid_ll_per_season), robust to seasonal outliers

Elbow analysis: marginal LL gain per additional active state (per gamma group).

Outputs (to {SCRATCH_DIR}/output/diagnostics/{parcellation}/selection_metrics/{sub_id}/):
  - selection_metrics_vt{VT}.csv
  - selection_metrics_vt{VT}.json
  - M1_selection_metrics_vt{VT}.png

Usage:
    python script/04_patch_selection_metrics.py \\
        --sub_id sub-01 --vt 0.95

    # Custom overfitting penalty strength:
    python script/04_patch_selection_metrics.py \\
        --sub_id sub-01 --vt 0.95 --gap_lambda 1.0

    # All subjects at once:
    python script/utils/04_patch_selection_metrics.py --vt 0.95
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils.plot_style import apply_publication_style
from utils.common import normalize_parcellation_name

from dotenv import load_dotenv
load_dotenv()

SCRATCH_DIR = os.getenv('SCRATCH_DIR')
if not SCRATCH_DIR:
    raise RuntimeError("SCRATCH_DIR not set in environment.")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

apply_publication_style()
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# =============================================================================
# Data collection
# =============================================================================

def collect_configs(parcellation, sub_id, vt):
    """Load all config_summary.json files for a subject, filtered by vt prefix.

    Returns list of parsed dicts.
    """
    base = os.path.join(SCRATCH_DIR, 'output', '04_combined_hdphmm',
                        parcellation, sub_id, 'configs')
    if not os.path.isdir(base):
        logger.error(f"Configs directory not found: {base}")
        return []

    vt_prefix = f"vt{vt}_"
    summaries = []

    for cfg_name in sorted(os.listdir(base)):
        if not cfg_name.startswith(vt_prefix):
            continue
        summary_path = os.path.join(base, cfg_name, 'config_summary.json')
        if not os.path.isfile(summary_path):
            continue
        try:
            with open(summary_path, 'r') as f:
                s = json.load(f)
            summaries.append(s)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Skipping {cfg_name}: {e}")

    logger.info(f"  {sub_id}: loaded {len(summaries)} configs at vt={vt}")
    return summaries


# =============================================================================
# Metric computation
# =============================================================================

def _best_seed_by_train(summary):
    """Return the best seed result by training LL (matching 04 BIC logic)."""
    successful = [r for r in summary.get('seed_results', [])
                  if r.get('status') == 'success'
                  and r.get('train_ll_per_sample') is not None
                  and r.get('n_active_states', -1) > 0]
    if not successful:
        return None
    return max(successful, key=lambda r: r['train_ll_per_sample'])


def _best_seed_by_valid(summary):
    """Return the best seed result by validation LL."""
    successful = [r for r in summary.get('seed_results', [])
                  if r.get('status') == 'success'
                  and r.get('valid_ll_per_sample') is not None
                  and r.get('n_active_states', -1) > 0]
    if not successful:
        return None
    return max(successful, key=lambda r: r['valid_ll_per_sample'])


def compute_bic(summary):
    """Replicate BIC logic from 04_combined_hdphmm.py lines 926-967."""
    best = _best_seed_by_train(summary)
    if best is None:
        return np.nan

    train_ll = float(best['train_ll_per_sample'])
    K = int(best['n_active_states'])
    n_bic = int(best.get('n_train_samples', 0))
    if n_bic <= 1:
        return np.nan

    D = summary['n_pcs']
    cov = summary['config']['covariance_type']
    if cov == 'diag':
        n_params = 2 * K * D + K * (K - 1) + (K - 1)
    else:
        n_params = K * D + K * D * (D + 1) // 2 + K * (K - 1) + (K - 1)

    bic_penalty = n_params * np.log(n_bic) / (2 * n_bic)
    return train_ll - bic_penalty


def compute_metrics(summaries, gap_lambda=0.5):
    """Compute all selection metrics for a list of config summaries.

    Returns a DataFrame with one row per config, all metrics as columns.
    """
    rows = []

    for s in summaries:
        cfg = s.get('config', {})
        nc = cfg.get('n_components')
        gamma = cfg.get('gamma')
        cov = cfg.get('covariance_type', 'diag')
        n_pcs = s.get('n_pcs')
        config_name = s.get('config_name', '')

        # Short label for plots (strip vt prefix)
        parts = config_name.split('_')
        short_label = '_'.join(p for p in parts if not p.startswith('vt')
                               and not p.startswith('cov'))
        # e.g., "nc60_g5"

        # Per-seed stats
        successful = [r for r in s.get('seed_results', [])
                      if r.get('status') == 'success']
        if not successful:
            continue

        active_counts = [r['n_active_states'] for r in successful
                         if 'n_active_states' in r]
        valid_lls = [r['valid_ll_per_sample'] for r in successful
                     if r.get('valid_ll_per_sample') is not None]
        train_lls = [r['train_ll_per_sample'] for r in successful
                     if r.get('train_ll_per_sample') is not None]

        if not active_counts or not valid_lls or not train_lls:
            continue

        # Best seed by validation LL (used for most metrics)
        best_valid = _best_seed_by_valid(s)
        best_train = _best_seed_by_train(s)
        if best_valid is None or best_train is None:
            continue

        valid_ll = float(best_valid['valid_ll_per_sample'])
        train_ll_of_best_valid = float(best_valid['train_ll_per_sample'])
        K_valid = int(best_valid['n_active_states'])
        K_train = int(best_train['n_active_states'])

        # Generalization gap (from the same seed, for consistency)
        gap = train_ll_of_best_valid - valid_ll

        # Season LLs from best valid seed
        season_lls = best_valid.get('valid_ll_per_season', {})
        season_values = [float(v) for v in season_lls.values()] if season_lls else []

        # --- Metrics (all "higher is better") ---

        # M1: Raw validation LL
        m_valid_ll = valid_ll

        # M2: BIC (uses best train seed, matches 04 logic)
        m_bic = compute_bic(s)

        # M3: Reflected validation LL
        # 2*valid - train = valid - gap. If no overfit (gap=0), equals valid_ll.
        # If train exceeds valid by δ, score drops by δ. Additive, scale-invariant,
        # zero free parameters.
        m_reflected = 2.0 * valid_ll - train_ll_of_best_valid

        # M4: Validation LL per active state
        m_ll_per_active = valid_ll / K_valid if K_valid > 0 else np.nan

        # M5: Gap-penalized LL with tunable lambda
        # valid_ll - lambda * gap
        m_gap_penalized = valid_ll - gap_lambda * gap

        # M6: Season worst-case
        m_season_worst = min(season_values) if season_values else np.nan

        # --- Summary statistics ---
        mean_active = np.mean(active_counts)
        std_active = np.std(active_counts)
        inactive_frac = 1.0 - mean_active / nc if nc > 0 else np.nan
        seed_valid_std = np.std(valid_lls) if len(valid_lls) > 1 else 0.0
        season_std = np.std(season_values) if len(season_values) > 1 else np.nan
        season_range = (max(season_values) - min(season_values)
                        if len(season_values) > 1 else np.nan)

        rows.append({
            'config_name': config_name,
            'short_label': short_label,
            'nc': nc,
            'gamma': gamma,
            'cov_type': cov,
            'n_pcs': n_pcs,
            # Active state stats
            'K_best_valid': K_valid,
            'K_best_train': K_train,
            'K_mean': mean_active,
            'K_std': std_active,
            'inactive_frac': inactive_frac,
            # Raw LL stats
            'valid_ll': valid_ll,
            'train_ll': train_ll_of_best_valid,
            'overfit_gap': gap,
            'seed_valid_std': seed_valid_std,
            'season_std': season_std,
            'season_range': season_range,
            'n_seeds': len(successful),
            # --- Metrics ---
            'M1_valid_ll': m_valid_ll,
            'M2_bic': m_bic,
            'M3_reflected': m_reflected,
            'M4_ll_per_active': m_ll_per_active,
            'M5_gap_penalized': m_gap_penalized,
            'M6_season_worst': m_season_worst,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Sort by nc, gamma for consistent ordering
    df = df.sort_values(['nc', 'gamma']).reset_index(drop=True)
    return df


def compute_rankings(df):
    """Add rank columns for each metric (1 = best). Higher metric = better."""
    metric_cols = [c for c in df.columns if c.startswith('M')]

    for col in metric_cols:
        rank_col = col.replace('M', 'R')  # e.g., M1_valid_ll -> R1_valid_ll
        df[rank_col] = df[col].rank(ascending=False, method='min').astype(int)

    # Mean rank across all metrics
    rank_cols = [c for c in df.columns if c.startswith('R')]
    df['mean_rank'] = df[rank_cols].mean(axis=1)

    return df


def compute_marginal_gains(df):
    """Compute marginal LL gain per additional active state, per gamma group.

    Returns a DataFrame with columns: gamma, nc_from, nc_to, delta_valid_ll,
    delta_K, marginal_gain.
    """
    gains = []
    for gamma_val in sorted(df['gamma'].unique()):
        group = df[df['gamma'] == gamma_val].sort_values('nc')
        if len(group) < 2:
            continue
        for i in range(len(group) - 1):
            r1 = group.iloc[i]
            r2 = group.iloc[i + 1]
            delta_ll = r2['valid_ll'] - r1['valid_ll']
            delta_K = r2['K_best_valid'] - r1['K_best_valid']
            marginal = delta_ll / max(delta_K, 1) if delta_K > 0 else np.nan
            gains.append({
                'gamma': gamma_val,
                'nc_from': int(r1['nc']),
                'nc_to': int(r2['nc']),
                'delta_valid_ll': delta_ll,
                'delta_K': delta_K,
                'marginal_gain': marginal,
            })
    return pd.DataFrame(gains)


# =============================================================================
# Plotting
# =============================================================================

GAMMA_COLORS = {1: '#1b9e77', 5: '#d95f02', 10: '#7570b3'}
GAMMA_MARKERS = {1: 'o', 5: 's', 10: '^'}


def plot_selection_panels(df, gains_df, out_path, sub_id, vt):
    """Create 6-panel diagnostic figure."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'{sub_id}  |  vt={vt}  |  Selection Metric Comparison',
                 fontsize=13, fontweight='bold')

    metric_cols = [c for c in df.columns if c.startswith('M')]
    rank_cols = [c for c in df.columns if c.startswith('R')]
    labels = df['short_label'].values

    # --- Panel A: BIC vs valid_ll bar comparison ---
    ax = axes[0, 0]
    x = np.arange(len(df))
    w = 0.35
    # Normalize both to [0,1] for visual comparison
    for i, (col, color, label) in enumerate([
        ('M1_valid_ll', '#2196F3', 'Valid LL'),
        ('M2_bic', '#FF9800', 'BIC'),
    ]):
        vals = df[col].values
        if np.ptp(vals) > 0:
            normed = (vals - vals.min()) / np.ptp(vals)
        else:
            normed = np.ones_like(vals) * 0.5
        ax.barh(x + i * w - w / 2, normed, height=w, color=color,
                alpha=0.8, label=label)
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Normalized score (0=worst, 1=best)')
    ax.set_title('A: Valid LL vs BIC')
    ax.legend(fontsize=7, loc='lower right')
    ax.invert_yaxis()

    # --- Panel B: LL vs K_active scatter with Pareto frontier ---
    ax = axes[0, 1]
    for gamma_val in sorted(df['gamma'].unique()):
        mask = df['gamma'] == gamma_val
        ax.scatter(df.loc[mask, 'K_best_valid'], df.loc[mask, 'valid_ll'],
                   c=GAMMA_COLORS.get(gamma_val, 'gray'),
                   marker=GAMMA_MARKERS.get(gamma_val, 'o'),
                   s=60, label=f'γ={gamma_val}', zorder=3)
    # Label points
    for _, row in df.iterrows():
        ax.annotate(f"nc{int(row['nc'])}",
                    (row['K_best_valid'], row['valid_ll']),
                    fontsize=6, ha='left', va='bottom',
                    xytext=(3, 3), textcoords='offset points')
    # Pareto frontier: configs where no other dominates on both (fewer K, higher LL)
    pareto_idx = []
    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if (other['K_best_valid'] <= row['K_best_valid']
                    and other['valid_ll'] >= row['valid_ll']
                    and (other['K_best_valid'] < row['K_best_valid']
                         or other['valid_ll'] > row['valid_ll'])):
                dominated = True
                break
        if not dominated:
            pareto_idx.append(i)
    if pareto_idx:
        pareto = df.loc[pareto_idx].sort_values('K_best_valid')
        ax.plot(pareto['K_best_valid'], pareto['valid_ll'],
                'r--', alpha=0.5, linewidth=1.5, label='Pareto front')
    ax.set_xlabel('K active states')
    ax.set_ylabel('Valid LL / sample')
    ax.set_title('B: LL vs Active States (Pareto)')
    ax.legend(fontsize=7)

    # --- Panel C: Overfit gap bar chart ---
    ax = axes[0, 2]
    colors = [GAMMA_COLORS.get(g, 'gray') for g in df['gamma']]
    order = df['overfit_gap'].argsort().values  # ascending gap
    ax.barh(np.arange(len(df)), df['overfit_gap'].values[order],
            color=[colors[i] for i in order], alpha=0.8)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(labels[order], fontsize=7)
    ax.set_xlabel('Train − Valid LL (lower = less overfit)')
    ax.set_title('C: Generalization Gap')
    ax.invert_yaxis()

    # --- Panel D: Season LL heatmap ---
    ax = axes[1, 0]
    # Collect season LLs from best valid seed for each config
    season_keys = ['s1', 's2', 's3', 's4', 's5', 's6']
    season_matrix = np.full((len(df), 6), np.nan)
    for i, (_, row) in enumerate(df.iterrows()):
        # Re-read the summary to get season data
        # (we stored valid_ll but not season breakdown in the DataFrame)
        pass

    # Rebuild season matrix from the summaries - need to pass summaries through
    # For now, show marginal LL gain curves instead
    if not gains_df.empty:
        for gamma_val in sorted(gains_df['gamma'].unique()):
            g = gains_df[gains_df['gamma'] == gamma_val]
            midpoints = [(r['nc_from'] + r['nc_to']) / 2 for _, r in g.iterrows()]
            ax.plot(midpoints, g['marginal_gain'],
                    color=GAMMA_COLORS.get(gamma_val, 'gray'),
                    marker=GAMMA_MARKERS.get(gamma_val, 'o'),
                    label=f'γ={gamma_val}', linewidth=1.5)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(0.01, color='red', linestyle='--', alpha=0.4,
                    label='threshold=0.01')
        ax.set_xlabel('nc midpoint')
        ax.set_ylabel('ΔLL / ΔK (marginal gain)')
        ax.set_title('D: Marginal LL Gain per State')
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, 'Insufficient data\nfor marginal gains',
                ha='center', va='center', transform=ax.transAxes, fontsize=9)
        ax.set_title('D: Marginal LL Gain per State')

    # --- Panel E: Rank heatmap ---
    ax = axes[1, 1]
    if rank_cols:
        rank_matrix = df[rank_cols].values.astype(float)
        n_configs = len(df)
        im = ax.imshow(rank_matrix, aspect='auto', cmap='RdYlGn_r',
                        vmin=1, vmax=n_configs)
        ax.set_xticks(np.arange(len(rank_cols)))
        ax.set_xticklabels([c.split('_', 1)[1] for c in rank_cols],
                           fontsize=7, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(df)))
        ax.set_yticklabels(labels, fontsize=7)
        # Annotate cells with rank number
        for i in range(rank_matrix.shape[0]):
            for j in range(rank_matrix.shape[1]):
                ax.text(j, i, f'{int(rank_matrix[i, j])}',
                        ha='center', va='center', fontsize=6,
                        color='white' if rank_matrix[i, j] > n_configs * 0.6
                        else 'black')
        # Add mean rank column annotation
        for i, mr in enumerate(df['mean_rank']):
            ax.text(len(rank_cols) - 0.3, i, f'  μ={mr:.1f}',
                    ha='left', va='center', fontsize=6, fontweight='bold')
        ax.set_title('E: Rank Heatmap (1=best)')
        fig.colorbar(im, ax=ax, shrink=0.6, label='Rank')

    # --- Panel F: Seed stability (K spread across seeds) ---
    ax = axes[1, 2]
    x = np.arange(len(df))
    ax.barh(x, df['K_mean'], xerr=df['K_std'], height=0.6,
            color=[GAMMA_COLORS.get(g, 'gray') for g in df['gamma']],
            alpha=0.8, capsize=3)
    # Overlay nc as a reference line per config
    for i, nc_val in enumerate(df['nc']):
        ax.plot(nc_val, i, '|', color='red', markersize=12, markeredgewidth=2)
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Active states (red | = nc truncation)')
    ax.set_title('F: Seed Stability (K ± std)')
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved figure: {out_path}")


# =============================================================================
# JSON recommendation
# =============================================================================

def build_recommendation(df, sub_id, vt, gap_lambda):
    """Build recommendation JSON with best config per metric."""
    metric_cols = [c for c in df.columns if c.startswith('M')]

    recs = {}
    for col in metric_cols:
        best_idx = df[col].idxmax()
        best_row = df.loc[best_idx]
        recs[col] = {
            'config_name': best_row['config_name'],
            'short_label': best_row['short_label'],
            'value': float(best_row[col]),
            'K_active': int(best_row['K_best_valid']),
            'nc': int(best_row['nc']),
        }

    # Consensus: lowest mean rank
    best_consensus_idx = df['mean_rank'].idxmin()
    best_consensus = df.loc[best_consensus_idx]
    recs['consensus'] = {
        'config_name': best_consensus['config_name'],
        'short_label': best_consensus['short_label'],
        'mean_rank': float(best_consensus['mean_rank']),
        'K_active': int(best_consensus['K_best_valid']),
        'nc': int(best_consensus['nc']),
    }

    return {
        'sub_id': sub_id,
        'vt': float(vt),
        'gap_lambda': gap_lambda,
        'n_configs_evaluated': len(df),
        'recommendations': recs,
        'timestamp': datetime.now().isoformat(),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare model selection metrics across HDP-HMM configs.'
    )
    parser.add_argument('--sub_id', default=None,
                        help='Subject ID (e.g., sub-01). If omitted, run all.')
    parser.add_argument('--parcellation', default='atlas-4S156Parcels')
    parser.add_argument('--vt', required=True,
                        help='Variance threshold to filter configs (e.g., 0.95)')
    parser.add_argument('--gap_lambda', type=float, default=0.5,
                        help='Lambda for M5 gap-penalized LL (default: 0.5)')
    args = parser.parse_args()

    parcellation = normalize_parcellation_name(args.parcellation)
    vt = args.vt

    # Determine subjects
    base = os.path.join(SCRATCH_DIR, 'output', '04_combined_hdphmm', parcellation)
    if args.sub_id:
        sub_ids = [args.sub_id]
    else:
        sub_ids = sorted([
            d for d in os.listdir(base)
            if d.startswith('sub-') and os.path.isdir(os.path.join(base, d))
        ])

    for sub_id in sub_ids:
        logger.info(f"Processing {sub_id} at vt={vt}...")

        # Collect configs
        summaries = collect_configs(parcellation, sub_id, vt)
        if not summaries:
            logger.warning(f"  No configs found for {sub_id} at vt={vt}")
            continue

        # Compute metrics
        df = compute_metrics(summaries, gap_lambda=args.gap_lambda)
        if df.empty:
            logger.warning(f"  No valid configs for {sub_id}")
            continue

        # Compute rankings
        df = compute_rankings(df)

        # Compute marginal gains
        gains_df = compute_marginal_gains(df)

        # Output directory
        out_dir = os.path.join(SCRATCH_DIR, 'output', 'diagnostics',
                               parcellation, 'selection_metrics', sub_id)
        os.makedirs(out_dir, exist_ok=True)

        # Save CSV
        csv_path = os.path.join(out_dir, f'selection_metrics_vt{vt}.csv')
        df.to_csv(csv_path, index=False, float_format='%.6f')
        logger.info(f"  Saved CSV: {csv_path}")

        # Save figure
        fig_path = os.path.join(out_dir, f'M1_selection_metrics_vt{vt}.png')
        plot_selection_panels(df, gains_df, fig_path, sub_id, vt)

        # Save JSON recommendation
        rec = build_recommendation(df, sub_id, vt, args.gap_lambda)
        json_path = os.path.join(out_dir, f'selection_metrics_vt{vt}.json')
        with open(json_path, 'w') as f:
            json.dump(rec, f, indent=2)
        logger.info(f"  Saved JSON: {json_path}")

        # Print summary table
        logger.info(f"\n  === {sub_id} vt={vt} - Metric Rankings ===")
        metric_cols = [c for c in df.columns if c.startswith('M')]
        rank_cols = [c for c in df.columns if c.startswith('R')]
        summary_cols = ['short_label', 'nc', 'K_best_valid', 'inactive_frac',
                        'overfit_gap'] + metric_cols + ['mean_rank']
        print(df[summary_cols].to_string(index=False, float_format='%.4f'))
        print()

        # Print recommendations
        for metric, rec_info in rec['recommendations'].items():
            if metric == 'consensus':
                logger.info(f"  CONSENSUS: {rec_info['short_label']} "
                            f"(mean_rank={rec_info['mean_rank']:.1f}, "
                            f"K={rec_info['K_active']})")
            else:
                logger.info(f"  {metric}: best = {rec_info['short_label']} "
                            f"(value={rec_info['value']:.4f}, "
                            f"K={rec_info['K_active']})")


if __name__ == '__main__':
    main()
