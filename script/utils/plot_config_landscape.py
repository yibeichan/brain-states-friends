#!/usr/bin/env python3
"""
plot_config_landscape.py - Tabulate and visualize HDP-HMM config grid results.

Reads all config_summary.json and stage1_result.json files across subjects
to produce a comprehensive view of the variance threshold x model capacity
landscape. Decomposes the Stage 1 metric (ll_per_dim) into eigenvalue-bias
and genuine HMM-gain components.

No model pickles are loaded - only JSON files. Fast, no GPU, no large memory.

Outputs (to {SCRATCH_DIR}/output/diagnostics/{parcellation}/config_landscape/):
  - config_landscape_summary.csv   Full DataFrame
  - C1_stage1_vt_comparison.png    ll_per_dim vs vt, per subject
  - C2_ll_decomposition.png        null_LL/dim + HMM_gain/dim = ll_per_dim
  - C3_stage2_heatmap_{sub}.png    Validation LL heatmap: nc x gamma at vt=0.95
  - C4_active_states_vs_params.png n_active vs nc, colored by gamma

Usage:
    python script/utils/plot_config_landscape.py --parcellation atlas-4S156Parcels

    # Single subject:
    python script/utils/plot_config_landscape.py --sub_id sub-01
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
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


# =============================================================================
# Stage 1 fixed parameters (used to label stage1 vs stage2 configs)
# =============================================================================
STAGE1_NC = 60
STAGE1_GAMMA = 5


# =============================================================================
# Data collection
# =============================================================================

def collect_config_summaries(parcellation, sub_id=None):
    """Walk all subjects' config directories and parse config_summary.json.

    Returns a list of dicts (one per config per subject).
    """
    base = os.path.join(SCRATCH_DIR, 'output', '04_combined_hdphmm', parcellation)
    if not os.path.isdir(base):
        logger.error(f"Output base not found: {base}")
        sys.exit(1)

    if sub_id:
        sub_dirs = [os.path.join(base, sub_id)]
    else:
        sub_dirs = sorted(glob(os.path.join(base, 'sub-*')))

    rows = []
    n_unpatched = 0

    for sub_dir in sub_dirs:
        sid = os.path.basename(sub_dir)
        configs_dir = os.path.join(sub_dir, 'configs')
        if not os.path.isdir(configs_dir):
            logger.warning(f"No configs/ for {sid}")
            continue

        # Load stage1 result to mark selected vt
        stage1_path = os.path.join(sub_dir, 'stage1_result.json')
        selected_vt = None
        if os.path.exists(stage1_path):
            with open(stage1_path, 'r') as f:
                stage1 = json.load(f)
            selected_vt = stage1.get('selected_vt')

        for cfg_name in sorted(os.listdir(configs_dir)):
            cfg_dir = os.path.join(configs_dir, cfg_name)
            summary_path = os.path.join(cfg_dir, 'config_summary.json')
            if not os.path.isfile(summary_path):
                continue

            try:
                with open(summary_path, 'r') as f:
                    s = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Skipping {sid}/{cfg_name}: {e}")
                continue

            # Check patch status
            if not s.get('patched_timestamp'):
                n_unpatched += 1

            cfg = s.get('config', {})
            vt = cfg.get('variance_threshold')
            nc = cfg.get('n_components')
            gamma = cfg.get('gamma')
            cov = cfg.get('covariance_type', 'diag')
            n_pcs = s.get('n_pcs')

            # Extract per-seed active state counts
            seed_results = s.get('seed_results', [])
            active_counts = [
                sr['n_active_states'] for sr in seed_results
                if sr.get('status') == 'success' and 'n_active_states' in sr
            ]

            # Classify stage
            is_stage1 = (nc == STAGE1_NC and gamma == STAGE1_GAMMA)

            row = {
                'sub_id': sid,
                'config_name': cfg_name,
                'vt': vt,
                'nc': nc,
                'gamma': gamma,
                'cov_type': cov,
                'n_pcs': n_pcs,
                'mean_valid_ll': s.get('mean_valid_ll_per_sample'),
                'std_valid_ll': s.get('std_valid_ll_per_sample'),
                'best_train_ll': s.get('best_train_ll_per_sample'),
                'best_valid_ll': s.get('best_seed_valid_ll_per_sample'),
                'mean_n_active': np.mean(active_counts) if active_counts else np.nan,
                'std_n_active': np.std(active_counts) if active_counts else np.nan,
                'n_successful': s.get('n_seeds_successful', 0),
                'stage': 'stage1' if is_stage1 else 'stage2',
                'is_selected_vt': (vt == selected_vt) if selected_vt else False,
                'patched': bool(s.get('patched_timestamp')),
            }

            # Derived metrics
            if row['mean_valid_ll'] is not None and n_pcs:
                row['ll_per_dim'] = row['mean_valid_ll'] / n_pcs
            else:
                row['ll_per_dim'] = np.nan

            if row['best_train_ll'] is not None and row['best_valid_ll'] is not None:
                row['overfit_gap'] = row['best_train_ll'] - row['best_valid_ll']
            else:
                row['overfit_gap'] = np.nan

            # Null LL per dim: single isotropic Gaussian baseline = -0.5*(log(2π) + 1)
            row['null_ll_per_dim'] = -0.5 * (np.log(2 * np.pi) + 1)
            row['hmm_gain_per_dim'] = row['ll_per_dim'] - row['null_ll_per_dim']

            rows.append(row)

    if n_unpatched > 0:
        logger.warning(
            f"{n_unpatched} config_summary.json files lack patched_timestamp. "
            f"Run script/utils/patch_config_summaries.py to fix "
            f"best_seed_valid_ll_per_sample values."
        )

    return pd.DataFrame(rows)


# =============================================================================
# Plotting functions
# =============================================================================

def plot_c1_stage1_vt_comparison(df, out_dir):
    """C1: ll_per_dim vs vt, one line per subject."""
    stage1 = df[df['stage'] == 'stage1'].copy()
    if stage1.empty:
        logger.warning("No stage1 configs found - skipping C1")
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    subjects = sorted(stage1['sub_id'].unique())
    for sid in subjects:
        sub_data = stage1[stage1['sub_id'] == sid].sort_values('vt')
        ax.plot(sub_data['vt'], sub_data['ll_per_dim'],
                marker='o', markersize=5, label=sid, linewidth=1.5)

        # Mark selected vt
        selected = sub_data[sub_data['is_selected_vt']]
        if not selected.empty:
            ax.scatter(selected['vt'].values, selected['ll_per_dim'].values,
                       s=120, facecolors='none', edgecolors='red',
                       linewidths=2, zorder=5)

    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Variance threshold')
    ax.set_ylabel('LL per dimension (valid)')
    ax.set_title('Stage 1: vt selection metric (ll_per_dim)\nRed circle = selected vt')
    ax.legend(fontsize=7, loc='upper left')
    fig.tight_layout()

    path = os.path.join(out_dir, 'C1_stage1_vt_comparison.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_c2_ll_decomposition(df, out_dir):
    """C2: Stacked bar showing null_LL/dim + HMM_gain/dim = ll_per_dim."""
    stage1 = df[df['stage'] == 'stage1'].copy()
    if stage1.empty:
        logger.warning("No stage1 configs found - skipping C2")
        return

    subjects = sorted(stage1['sub_id'].unique())
    n_subs = len(subjects)
    vt_vals = sorted(stage1['vt'].unique())
    n_vt = len(vt_vals)

    fig, axes = plt.subplots(1, n_subs, figsize=(3.5 * n_subs, 4), sharey=True)
    if n_subs == 1:
        axes = [axes]

    x = np.arange(n_vt)
    width = 0.6

    for ax, sid in zip(axes, subjects):
        sub_data = stage1[stage1['sub_id'] == sid].sort_values('vt')
        null_vals = sub_data['null_ll_per_dim'].values
        gain_vals = sub_data['hmm_gain_per_dim'].values

        ax.bar(x, null_vals, width, label='Null LL/dim (eigenvalue bias)',
               color='#D55E00', alpha=0.8)
        ax.bar(x, gain_vals, width, bottom=null_vals, label='HMM gain/dim',
               color='#0072B2', alpha=0.8)

        # Total = ll_per_dim
        totals = null_vals + gain_vals
        for i, (t, g) in enumerate(zip(totals, gain_vals)):
            ax.text(i, t + 0.02, f'{g:.3f}', ha='center', va='bottom',
                    fontsize=7, color='#0072B2')

        ax.set_xticks(x)
        ax.set_xticklabels([f'{v:.2f}' for v in vt_vals], fontsize=8)
        ax.set_xlabel('Variance threshold')
        ax.set_title(sid, fontsize=10)
        ax.axhline(0, color='grey', linestyle='-', linewidth=0.5)

    axes[0].set_ylabel('LL per dimension')
    axes[0].legend(fontsize=7, loc='lower right')
    fig.suptitle('LL decomposition: eigenvalue bias vs HMM gain', fontsize=11, y=1.02)
    fig.tight_layout()

    path = os.path.join(out_dir, 'C2_ll_decomposition.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_c3_stage2_heatmap(df, out_dir):
    """C3: Validation LL heatmap nc x gamma at vt=0.95, per subject."""
    stage2 = df[df['stage'] == 'stage2'].copy()
    if stage2.empty:
        logger.warning("No stage2 configs found - skipping C3")
        return

    # Filter to vt=0.95 stage2 configs (where the grid was run)
    stage2_95 = stage2[np.isclose(stage2['vt'], 0.95)]
    if stage2_95.empty:
        logger.warning("No stage2 configs at vt=0.95 - skipping C3")
        return

    subjects = sorted(stage2_95['sub_id'].unique())

    for sid in subjects:
        sub_data = stage2_95[stage2_95['sub_id'] == sid]
        if sub_data.empty:
            continue

        nc_vals = sorted(sub_data['nc'].unique())
        gamma_vals = sorted(sub_data['gamma'].unique())

        # Build 2D grid
        grid = np.full((len(gamma_vals), len(nc_vals)), np.nan)
        for _, row in sub_data.iterrows():
            gi = gamma_vals.index(row['gamma'])
            ni = nc_vals.index(row['nc'])
            grid[gi, ni] = row['mean_valid_ll']

        fig, ax = plt.subplots(figsize=(5, 3.5))
        im = ax.imshow(grid, cmap='viridis', aspect='auto',
                        origin='lower')
        ax.set_xticks(range(len(nc_vals)))
        ax.set_xticklabels(nc_vals)
        ax.set_yticks(range(len(gamma_vals)))
        ax.set_yticklabels(gamma_vals)
        ax.set_xlabel('n_components')
        ax.set_ylabel('gamma')
        ax.set_title(f'{sid} - Stage 2 validation LL at vt=0.95')

        # Annotate cells
        for gi in range(len(gamma_vals)):
            for ni in range(len(nc_vals)):
                val = grid[gi, ni]
                if not np.isnan(val):
                    ax.text(ni, gi, f'{val:.2f}', ha='center', va='center',
                            fontsize=8, color='white' if val < np.nanmedian(grid) else 'black')

        fig.colorbar(im, ax=ax, label='Mean valid LL/sample', shrink=0.8)
        fig.tight_layout()

        path = os.path.join(out_dir, f'C3_stage2_heatmap_{sid}.png')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved {path}")


def plot_c4_active_states(df, out_dir):
    """C4: n_active_states vs nc, colored by gamma."""
    stage2 = df[df['stage'] == 'stage2'].copy()
    if stage2.empty:
        logger.warning("No stage2 configs found - skipping C4")
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    gamma_vals = sorted(stage2['gamma'].unique())
    colors = {1: '#E69F00', 5: '#56B4E9', 10: '#009E73'}

    for g in gamma_vals:
        g_data = stage2[stage2['gamma'] == g]
        color = colors.get(g, '#999999')
        ax.scatter(g_data['nc'], g_data['mean_n_active'],
                   c=color, label=f'gamma={g}', alpha=0.6, s=30, edgecolors='none')

    ax.set_xlabel('n_components (model capacity)')
    ax.set_ylabel('Mean active states')
    ax.set_title('Active states vs model capacity (Stage 2 configs)')
    ax.legend(fontsize=8)
    ax.plot([0, stage2['nc'].max() * 1.1], [0, stage2['nc'].max() * 1.1],
            'k--', alpha=0.3, linewidth=0.8, label='y=x')
    fig.tight_layout()

    path = os.path.join(out_dir, 'C4_active_states_vs_params.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved {path}")


# =============================================================================
# Console output
# =============================================================================

def print_summary_table(df):
    """Print a formatted summary table to console."""
    cols = ['sub_id', 'stage', 'vt', 'nc', 'gamma', 'cov_type', 'n_pcs',
            'mean_valid_ll', 'll_per_dim', 'hmm_gain_per_dim',
            'mean_n_active', 'overfit_gap', 'is_selected_vt']
    display = df[cols].sort_values(['sub_id', 'stage', 'vt', 'nc', 'gamma'])
    display = display.round({
        'mean_valid_ll': 3, 'll_per_dim': 4, 'hmm_gain_per_dim': 4,
        'mean_n_active': 1, 'overfit_gap': 4,
    })
    print("\n" + "=" * 120)
    print("CONFIG LANDSCAPE SUMMARY")
    print("=" * 120)
    print(display.to_string(index=False))
    print("=" * 120 + "\n")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Tabulate and visualize HDP-HMM config grid results.'
    )
    parser.add_argument('--parcellation', default='atlas-4S156Parcels',
                        help='Parcellation name (default: atlas-4S156Parcels)')
    parser.add_argument('--sub_id', default=None,
                        help='Single subject (default: all subjects)')
    parser.add_argument('--plots', nargs='*', default=['C1', 'C2', 'C3', 'C4'],
                        help='Which plots to generate (default: C1 C2 C3 C4)')
    args = parser.parse_args()

    parcellation = normalize_parcellation_name(args.parcellation)
    logger.info(f"Collecting configs for {parcellation}")

    df = collect_config_summaries(parcellation, sub_id=args.sub_id)
    if df.empty:
        logger.error("No config summaries found.")
        sys.exit(1)

    logger.info(f"Collected {len(df)} configs across "
                f"{df['sub_id'].nunique()} subject(s)")

    # Output directory
    out_dir = os.path.join(
        SCRATCH_DIR, 'output', 'diagnostics', parcellation, 'config_landscape'
    )
    os.makedirs(out_dir, exist_ok=True)

    # Save CSV
    csv_path = os.path.join(out_dir, 'config_landscape_summary.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {csv_path}")

    # Console table
    print_summary_table(df)

    # Plots
    plot_funcs = {
        'C1': plot_c1_stage1_vt_comparison,
        'C2': plot_c2_ll_decomposition,
        'C3': plot_c3_stage2_heatmap,
        'C4': plot_c4_active_states,
    }
    for plot_name in args.plots:
        if plot_name in plot_funcs:
            plot_funcs[plot_name](df, out_dir)
        else:
            logger.warning(f"Unknown plot: {plot_name}")

    logger.info(f"All outputs saved to {out_dir}")


if __name__ == '__main__':
    main()
