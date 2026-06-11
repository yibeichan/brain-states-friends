#!/usr/bin/env python3
"""
03b_pca_loadings.py - Visualize and diagnose PCA loadings from fitted PCA models.

Generates diagnostic figures A1-A7 from already-computed PCA outputs:

  A1. Loadings Heatmap        - top PCs x parcels, grouped by network
  A2. Top-Loading Parcels     - horizontal bar per PC (one fig per PC)
  A3. Residual Variance       - per-parcel & per-network residual fraction
  A4. Network Variance per PC - stacked bar of per-network loading energy
  A5. LOSO Residual Stability - scatter of primary vs LOSO residual fractions
  A6. Cross-Subject PC Comp.  - per-network PC composition across subjects
  A7. Residual vs Threshold   - residual variance sweep across variance thresholds

Also produces:
  - Motion artifact flags (SomMot/subcortical dominance in early PCs)
  - pca_loadings_flags.json with automated red-flag detection results

Prerequisites:
    - 03a_pca4combined_hmm.py completed for this subject
    - pca_model.pkl at {SCRATCH_DIR}/output/03a_pca4combined_hmm/{parcellation}/{sub_id}/
    - Parcel labels at {SCRATCH_DIR}/data/parcellation_labels/{parcellation}_labels.csv
    - For A5: LOSO PCA models from 03a (auto-detected)
    - For A6: PCA models for all 6 subjects from 03a

Usage:
    # Per-subject diagnostics (A1-A5, A7):
    python script/03b_pca_loadings.py --sub_id sub-01

    # Only specific plots:
    python script/03b_pca_loadings.py --sub_id sub-01 --plots A1 A3

    # Cross-subject comparison (A6, writes to shared dir):
    python script/03b_pca_loadings.py --sub_id sub-01 --cross_subject
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
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
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
# Constants
# =============================================================================

ALL_SUBJECTS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06']

# Motion artifact flag thresholds
SOMMOT_FLAG_THRESHOLD = 0.30   # SomMotA + SomMotB fraction in PC1-3
SUBCORT_FLAG_THRESHOLD = 0.25  # All subcortical fraction in PC1-2
SOMMOT_NETWORKS = {'SomMotA', 'SomMotB'}
SUBCORT_NETWORKS = {'BG', 'Midbrain-DA', 'Midbrain-Diencephalic',
                    'Thalamus', 'Hipp/Amyg', 'Cerebellum'}

# LOSO residual stability: CV threshold for flagging unstable parcels
LOSO_CV_THRESHOLD = 0.20


# =============================================================================
# Network definitions and label utilities
# =============================================================================

# Canonical ordering: Yeo-17 cortical sub-networks (visual -> DMN) + 6-bin
# canonical subcortical partition.
#
# IMPORTANT: subcortical bins MUST stay in sync with
# script/utils/plot_style.py:_SUBCORT_GROUPS (v2 canonical BG circuit per
# Alexander-DeLong-Strick 1986; Haber & Knutson 2010). Cortical bins here
# use Yeo-17 sub-network names (VisCent/VisPeri/etc.) — a separate local
# convention from plot_style.py's Yeo-7 names; unifying that is out of
# scope.
NETWORK_ORDER = [
    'VisCent', 'VisPeri',
    'SomMotA', 'SomMotB',
    'DorsAttnA', 'DorsAttnB',
    'SalVentAttnA', 'SalVentAttnB',
    'ContA', 'ContB', 'ContC',
    'TempPar',
    'DefaultA', 'DefaultB', 'DefaultC',
    'LimbicA', 'LimbicB',
    # Subcortical groups (v2 canonical partition; 6 bins)
    'BG', 'Midbrain-DA', 'Midbrain-Diencephalic',
    'Thalamus', 'Hipp/Amyg', 'Cerebellum',
]

# Colors for the 17 Yeo networks + subcortical groups
NETWORK_COLORS = {
    'VisCent': '#781286', 'VisPeri': '#DC143C',
    'SomMotA': '#4682B4', 'SomMotB': '#6495ED',
    'DorsAttnA': '#006400', 'DorsAttnB': '#00FF7F',
    'SalVentAttnA': '#C71585', 'SalVentAttnB': '#FF69B4',
    'ContA': '#E69422', 'ContB': '#F0C080', 'ContC': '#D4A574',
    'TempPar': '#A0522D',
    'DefaultA': '#CD3333', 'DefaultB': '#FF6347', 'DefaultC': '#FFA07A',
    'LimbicA': '#ADFF2F', 'LimbicB': '#7FFF00',
    # Subcortical (v2 canonical; matches plot_style.py)
    'BG': '#708090', 'Midbrain-DA': '#722F37',
    'Midbrain-Diencephalic': '#A9A9A9',
    'Thalamus': '#B0C4DE', 'Hipp/Amyg': '#DDA0DD', 'Cerebellum': '#BC8F8F',
}

# Subcortical structure-to-group mapping (v2 canonical BG circuit).
# Mirrors script/utils/plot_style.py:_SUBCORT_GROUPS — keep in sync.
_SUBCORT_GROUPS = {
    'Pu': 'BG', 'Ca': 'BG', 'NAC': 'BG',
    'GPe': 'BG', 'GPi': 'BG',
    'STH': 'BG', 'SNr': 'BG', 'VeP': 'BG',
    'SNc_PBP_VTA': 'Midbrain-DA',
    'RN': 'Midbrain-Diencephalic',
    'HN': 'Midbrain-Diencephalic',
    'HTH': 'Midbrain-Diencephalic',
    'MN': 'Midbrain-Diencephalic',
    'Pulvinar': 'Thalamus', 'Anterior': 'Thalamus', 'Medio_Dorsal': 'Thalamus',
    'Ventral_Latero_Dorsal': 'Thalamus',
    'Central_Lateral-Lateral_Posterior-Medial_Pulvinar': 'Thalamus',
    'Ventral_Anterior': 'Thalamus', 'Ventral_Latero_Ventral': 'Thalamus',
    'Hippocampus': 'Hipp/Amyg', 'Amygdala': 'Hipp/Amyg',
    'EXA': 'Hipp/Amyg',
}


def extract_network(label):
    """Parse network name from parcel label.

    Cortical: '17Networks_{hemi}_{network}_{subregion}_{index}' -> network
    Subcortical: mapped via _SUBCORT_GROUPS
    Cerebellar: 'Cerebellar_Region*' -> 'Cerebellum'
    """
    if label.startswith('17Networks_'):
        parts = label.split('_')
        # parts: ['17Networks', hemi, network, ...]
        if len(parts) >= 3:
            return parts[2]
        return 'Unknown'

    if label.startswith('Cerebellar'):
        return 'Cerebellum'

    # Subcortical: parse structure name
    # Formats: 'LH-Pu', 'RH-SNc_PBP_VTA', 'LH_Hippocampus', 'LH-Pulvinar'
    for sep in ['-', '_']:
        if sep in label:
            idx = label.index(sep)
            prefix = label[:idx]
            if prefix in ('LH', 'RH'):
                structure = label[idx + 1:]
                if structure in _SUBCORT_GROUPS:
                    return _SUBCORT_GROUPS[structure]

    return 'Unknown'


def abbreviate_parcel_label(label):
    """Shorten parcel label for axis display.

    '17Networks_LH_DefaultC_PHC_1' -> 'L.DefaultC_PHC_1'
    'LH-Pulvinar' -> 'L.Pulvinar'
    'Cerebellar_Region6' -> 'Cereb_6'
    """
    if label.startswith('17Networks_'):
        label = label.replace('17Networks_', '')
        label = label.replace('LH_', 'L.').replace('RH_', 'R.')
        return label

    if label.startswith('Cerebellar_Region'):
        num = label.replace('Cerebellar_Region', '')
        return f'Cereb_{num}'

    # Subcortical
    label = label.replace('LH-', 'L.').replace('RH-', 'R.')
    label = label.replace('LH_', 'L.').replace('RH_', 'R.')
    return label


def group_parcels_by_network(labels):
    """Group parcel indices by network, ordered by NETWORK_ORDER.

    Parameters
    ----------
    labels : list of str
        Parcel labels (length = n_parcels, already offset-corrected).

    Returns
    -------
    groups : list of (network_name, list_of_indices)
        Ordered by NETWORK_ORDER. Networks not present are omitted.
    network_per_parcel : list of str
        Network name for each parcel (same length as labels).
    """
    network_per_parcel = [extract_network(lab) for lab in labels]

    # Build groups
    network_to_indices = {}
    for i, net in enumerate(network_per_parcel):
        network_to_indices.setdefault(net, []).append(i)

    # Order by canonical list, append any extras
    groups = []
    seen = set()
    for net in NETWORK_ORDER:
        if net in network_to_indices:
            groups.append((net, network_to_indices[net]))
            seen.add(net)
    for net, indices in network_to_indices.items():
        if net not in seen:
            groups.append((net, indices))

    return groups, network_per_parcel


# =============================================================================
# Data loading
# =============================================================================

def _pca_output_dir(sub_id, parcellation):
    """Return 03a output directory for a subject."""
    return os.path.join(
        SCRATCH_DIR, 'output', '03a_pca4combined_hmm', parcellation, sub_id
    )


def load_pca_model(sub_id, parcellation):
    """Load PCA model and extract key arrays.

    Returns
    -------
    components : np.ndarray, shape (n_components, n_features)
    explained_variance : np.ndarray, shape (n_components,)
    explained_variance_ratio : np.ndarray, shape (n_components,)
    """
    pca_path = os.path.join(_pca_output_dir(sub_id, parcellation), 'pca_model.pkl')
    logger.info(f"Loading PCA model: {pca_path}")
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)

    components = pca.components_.copy()
    explained_variance = pca.explained_variance_.copy()
    explained_variance_ratio = pca.explained_variance_ratio_.copy()
    del pca
    return components, explained_variance, explained_variance_ratio


def load_n_pcs_lookup(sub_id, parcellation):
    """Load n_pcs_lookup.json mapping variance thresholds to PC counts."""
    lookup_path = os.path.join(
        _pca_output_dir(sub_id, parcellation), 'n_pcs_lookup.json'
    )
    with open(lookup_path) as f:
        return json.load(f)


def load_parcel_labels(parcellation):
    """Load parcel labels from CSV, returning offset-corrected list.

    Returns list of n_parcels labels (index 0 = parcel_id 1).
    """
    csv_path = os.path.join(
        SCRATCH_DIR, 'data', 'parcellation_labels',
        f'{parcellation}_labels.csv'
    )
    logger.info(f"Loading parcel labels: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.sort_values('parcel_id')
    return list(df['label_name'])


def load_loso_pca_models(sub_id, parcellation):
    """Load all available LOSO PCA models.

    Returns
    -------
    loso_models : dict
        season (int) -> (components, explained_variance, explained_variance_ratio)
    """
    base = _pca_output_dir(sub_id, parcellation)
    loso_dir = os.path.join(base, 'loso')
    if not os.path.isdir(loso_dir):
        return {}

    models = {}
    for entry in sorted(os.listdir(loso_dir)):
        if not entry.startswith('season_'):
            continue
        season = int(entry.replace('season_', ''))
        pca_path = os.path.join(loso_dir, entry, 'pca_model.pkl')
        if not os.path.exists(pca_path):
            continue
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
        models[season] = (
            pca.components_.copy(),
            pca.explained_variance_.copy(),
            pca.explained_variance_ratio_.copy(),
        )
        del pca
        logger.info(f"  Loaded LOSO PCA: season {season}")

    return models


# =============================================================================
# Motion artifact flagging
# =============================================================================

def compute_motion_artifact_flags(components, groups):
    """Check for motion artifact contamination in early PCs.

    Flags:
    - SomMotA + SomMotB > SOMMOT_FLAG_THRESHOLD in PC1, PC2, or PC3
    - All subcortical networks > SUBCORT_FLAG_THRESHOLD in PC1 or PC2

    Returns
    -------
    flags : dict
        'sommot_fractions': {pc_index: fraction} for PC1-3
        'subcort_fractions': {pc_index: fraction} for PC1-2
        'sommot_flagged': list of flagged PC indices (1-based)
        'subcort_flagged': list of flagged PC indices (1-based)
        'any_flag': bool
    """
    # Build network index sets
    net_indices = {name: set(indices) for name, indices in groups}
    sommot_idx = set()
    subcort_idx = set()
    for name, indices in groups:
        if name in SOMMOT_NETWORKS:
            sommot_idx.update(indices)
        if name in SUBCORT_NETWORKS:
            subcort_idx.update(indices)

    sommot_idx = sorted(sommot_idx)
    subcort_idx = sorted(subcort_idx)

    n_pcs_check = min(3, components.shape[0])
    sommot_fracs = {}
    subcort_fracs = {}
    sommot_flagged = []
    subcort_flagged = []

    for pc_idx in range(n_pcs_check):
        total_sq = np.sum(components[pc_idx, :] ** 2)
        if total_sq == 0:
            continue

        # SomMot fraction
        sm_frac = float(np.sum(components[pc_idx, sommot_idx] ** 2) / total_sq)
        sommot_fracs[pc_idx + 1] = round(sm_frac, 4)
        if sm_frac > SOMMOT_FLAG_THRESHOLD:
            sommot_flagged.append(pc_idx + 1)

        # Subcortical fraction (only check PC1-2)
        if pc_idx < 2:
            sc_frac = float(np.sum(components[pc_idx, subcort_idx] ** 2) / total_sq)
            subcort_fracs[pc_idx + 1] = round(sc_frac, 4)
            if sc_frac > SUBCORT_FLAG_THRESHOLD:
                subcort_flagged.append(pc_idx + 1)

    any_flag = bool(sommot_flagged or subcort_flagged)

    if sommot_flagged:
        logger.warning(
            f"MOTION ARTIFACT FLAG: SomMotA+B > {SOMMOT_FLAG_THRESHOLD:.0%} "
            f"in PC(s) {sommot_flagged}. Fractions: {sommot_fracs}"
        )
    if subcort_flagged:
        logger.warning(
            f"SUSCEPTIBILITY FLAG: Subcortical > {SUBCORT_FLAG_THRESHOLD:.0%} "
            f"in PC(s) {subcort_flagged}. Fractions: {subcort_fracs}"
        )
    if not any_flag:
        logger.info("Motion artifact check: PASSED (no flags)")

    return {
        'sommot_fractions': sommot_fracs,
        'subcort_fractions': subcort_fracs,
        'sommot_flagged': sommot_flagged,
        'subcort_flagged': subcort_flagged,
        'sommot_threshold': SOMMOT_FLAG_THRESHOLD,
        'subcort_threshold': SUBCORT_FLAG_THRESHOLD,
        'any_flag': any_flag,
    }


# =============================================================================
# A1: Loadings Heatmap
# =============================================================================

def plot_A1_loadings_heatmap(components, var_ratio, labels, groups, n_top_pcs,
                             output_dir):
    """Heatmap of top PC loadings, parcels grouped by network."""
    n_pcs = min(n_top_pcs, components.shape[0])

    # Build reordered column indices
    reorder = []
    boundary_positions = []
    boundary_labels = []
    pos = 0
    for net_name, indices in groups:
        reorder.extend(indices)
        boundary_positions.append(pos + len(indices) / 2)
        boundary_labels.append(net_name)
        pos += len(indices)

    data = components[:n_pcs, :][:, reorder]
    vmax = np.max(np.abs(data))

    fig_height = max(4, 0.5 * n_pcs + 2)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    im = ax.imshow(data, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Loading weight', shrink=0.8)

    # Network boundary lines
    pos = 0
    for net_name, indices in groups:
        pos += len(indices)
        if pos < len(reorder):
            ax.axvline(x=pos - 0.5, color='black', linewidth=0.5, alpha=0.5)

    # Axes
    ax.set_xticks(boundary_positions)
    ax.set_xticklabels(boundary_labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(n_pcs))
    ax.set_yticklabels([f'PC {i+1} ({var_ratio[i]*100:.1f}%)' for i in range(n_pcs)],
                       fontsize=9)
    ax.set_xlabel('Parcels (grouped by network)')
    ax.set_title(f'A1. PCA Loadings Heatmap -- Top {n_pcs} PCs')

    out_path = os.path.join(output_dir, 'A1_pca_loadings_heatmap.png')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# =============================================================================
# A2: Top-Loading Parcels per PC
# =============================================================================

def plot_A2_top_parcels_per_pc(components, var_ratio, labels, network_per_parcel,
                                n_top_pcs, n_top_parcels, output_dir):
    """One horizontal bar chart per PC showing top-loading parcels."""
    n_pcs = min(n_top_pcs, components.shape[0])
    n_parcels = components.shape[1]
    n_show = min(n_top_parcels, n_parcels)

    for pc_idx in range(n_pcs):
        loadings = components[pc_idx, :]
        top_indices = np.argsort(np.abs(loadings))[::-1][:n_show]
        # Sort by signed loading for display (most negative at top, most positive at bottom)
        top_indices = top_indices[np.argsort(loadings[top_indices])]

        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * n_show + 1)))

        vals = loadings[top_indices]
        bar_labels = [abbreviate_parcel_label(labels[i]) for i in top_indices]
        bar_colors = [NETWORK_COLORS.get(network_per_parcel[i], '#888888')
                      for i in top_indices]

        ax.barh(range(n_show), vals, color=bar_colors, edgecolor='black',
                linewidth=0.3, alpha=0.85)
        ax.set_yticks(range(n_show))
        ax.set_yticklabels(bar_labels, fontsize=8)
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_xlabel('Loading weight')
        ax.set_title(f'A2. PC {pc_idx+1} ({var_ratio[pc_idx]*100:.1f}% variance) '
                     f'-- Top {n_show} parcels by |loading|')
        ax.grid(True, axis='x', alpha=0.3)

        out_path = os.path.join(output_dir, f'A2_pc{pc_idx+1}_top_parcels.png')
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved: {out_path}")


# =============================================================================
# A3: Residual Variance Fraction
# =============================================================================

def compute_residual_variance(components, explained_variance, k):
    """Compute per-parcel signal vs residual variance at cutoff k.

    Returns
    -------
    signal_var : np.ndarray, shape (n_parcels,)
    residual_var : np.ndarray, shape (n_parcels,)
    residual_frac : np.ndarray, shape (n_parcels,)
    """
    V = components              # (n_comp, n_parcels)
    lam = explained_variance    # (n_comp,)
    signal_var = (V[:k, :] ** 2 * lam[:k, np.newaxis]).sum(axis=0)
    residual_var = (V[k:, :] ** 2 * lam[k:, np.newaxis]).sum(axis=0)
    total = signal_var + residual_var
    residual_frac = np.where(total > 0, residual_var / total, 0.0)
    return signal_var, residual_var, residual_frac


def plot_A3_residual_variance(components, explained_variance, labels, groups,
                               network_per_parcel, k, variance_threshold,
                               output_dir, suffix=''):
    """Two-panel plot: per-parcel and per-network residual fraction."""
    signal_var, residual_var, residual_frac = compute_residual_variance(
        components, explained_variance, k
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # --- Top panel: per-parcel, sorted ---
    sort_idx = np.argsort(residual_frac)[::-1]
    sorted_frac = residual_frac[sort_idx]
    sorted_colors = [NETWORK_COLORS.get(network_per_parcel[i], '#888888')
                     for i in sort_idx]

    ax1.bar(range(len(sorted_frac)), sorted_frac, color=sorted_colors,
            edgecolor='none', width=1.0)
    ax1.set_xlabel('Parcels (sorted by residual fraction)')
    ax1.set_ylabel('Residual variance fraction')
    ax1.set_title(f'A3a. Per-Parcel Residual Variance (k={k} PCs, '
                  f'{variance_threshold*100:.0f}% variance threshold)')
    ax1.set_xlim(-0.5, len(sorted_frac) - 0.5)
    ax1.set_ylim(0, min(1.0, sorted_frac[0] * 1.15))
    ax1.axhline(y=np.median(residual_frac), color='red', linestyle='--',
                alpha=0.7, linewidth=1.5,
                label=f'Median = {np.median(residual_frac):.3f}')
    ax1.legend(fontsize=9)
    ax1.grid(True, axis='y', alpha=0.3)

    # Label top 5 parcels
    for rank in range(min(5, len(sort_idx))):
        pidx = sort_idx[rank]
        ax1.annotate(abbreviate_parcel_label(labels[pidx]),
                     xy=(rank, sorted_frac[rank]),
                     xytext=(rank + 3, sorted_frac[rank] + 0.01),
                     fontsize=7, rotation=30, ha='left',
                     arrowprops=dict(arrowstyle='-', color='grey', lw=0.5))

    # --- Bottom panel: per-network ---
    net_names = []
    net_means = []
    net_stds = []
    net_colors = []
    for net_name, indices in groups:
        fracs = residual_frac[indices]
        net_names.append(net_name)
        net_means.append(np.mean(fracs))
        net_stds.append(np.std(fracs))
        net_colors.append(NETWORK_COLORS.get(net_name, '#888888'))

    # Sort by mean residual fraction
    order = np.argsort(net_means)[::-1]
    net_names = [net_names[i] for i in order]
    net_means = [net_means[i] for i in order]
    net_stds = [net_stds[i] for i in order]
    net_colors = [net_colors[i] for i in order]

    ax2.barh(range(len(net_names)), net_means, xerr=net_stds,
             color=net_colors, edgecolor='black', linewidth=0.3, alpha=0.85,
             capsize=3)
    ax2.set_yticks(range(len(net_names)))
    ax2.set_yticklabels(net_names, fontsize=9)
    ax2.set_xlabel('Mean residual variance fraction')
    ax2.set_title(f'A3b. Per-Network Residual Variance (k={k} PCs)')
    ax2.grid(True, axis='x', alpha=0.3)
    ax2.invert_yaxis()

    fname = f'A3_residual_variance_{suffix}.png' if suffix else 'A3_residual_variance.png'
    out_path = os.path.join(output_dir, fname)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# =============================================================================
# A4: Network Variance Contribution per PC (Stacked Bar)
# =============================================================================

def plot_A4_network_variance_per_pc(components, groups, n_pcs_show, output_dir,
                                     motion_flags=None):
    """Stacked bar: fraction of squared loadings from each network per PC.

    If motion_flags is provided, annotates flagged PCs with warning text.
    """
    n_pcs = min(n_pcs_show, components.shape[0])

    # Compute per-network squared loading fraction for each PC
    pc_indices = list(range(n_pcs))
    net_names = [g[0] for g in groups]
    n_nets = len(net_names)

    fractions = np.zeros((n_nets, n_pcs))
    for g_idx, (net_name, indices) in enumerate(groups):
        for pc_idx in pc_indices:
            sq_load = components[pc_idx, indices] ** 2
            fractions[g_idx, pc_idx] = sq_load.sum()

    # Normalize each PC to sum to 1
    col_sums = fractions.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    fractions /= col_sums

    fig, ax = plt.subplots(figsize=(max(10, n_pcs * 0.5 + 2), 6))

    bottom = np.zeros(n_pcs)
    x = np.arange(n_pcs)
    for g_idx, net_name in enumerate(net_names):
        color = NETWORK_COLORS.get(net_name, '#888888')
        ax.bar(x, fractions[g_idx], bottom=bottom, color=color,
               edgecolor='white', linewidth=0.2, label=net_name, width=0.85)
        bottom += fractions[g_idx]

    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Fraction of squared loadings')
    ax.set_title(f'A4. Network Variance Contribution per PC (top {n_pcs})')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}' for i in x], fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', alpha=0.2)

    # Annotate flagged PCs
    if motion_flags and motion_flags['any_flag']:
        flagged_pcs = set(motion_flags.get('sommot_flagged', []))
        flagged_pcs.update(motion_flags.get('subcort_flagged', []))
        for pc in flagged_pcs:
            if pc <= n_pcs:
                ax.annotate('FLAG', xy=(pc - 1, 1.02), fontsize=7,
                            color='red', fontweight='bold', ha='center',
                            annotation_clip=False)

    # Legend outside plot
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7,
              ncol=1, frameon=True)

    out_path = os.path.join(output_dir, 'A4_network_variance_per_pc.png')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# =============================================================================
# A5: LOSO Residual Variance Stability
# =============================================================================

def compute_loso_residual_stability(components, explained_variance, k,
                                     loso_models, labels, network_per_parcel):
    """Compare per-parcel residual fraction across primary and LOSO PCA models.

    Returns
    -------
    stability : dict with keys:
        'primary_frac': np.ndarray (n_parcels,)
        'loso_fracs': np.ndarray (n_folds, n_parcels)
        'loso_mean': np.ndarray (n_parcels,)
        'loso_std': np.ndarray (n_parcels,)
        'loso_cv': np.ndarray (n_parcels,)
        'high_cv_parcels': list of parcel indices with CV > threshold
        'seasons': list of season ints
    """
    _, _, primary_frac = compute_residual_variance(components, explained_variance, k)

    seasons = sorted(loso_models.keys())
    n_parcels = components.shape[1]
    loso_fracs = np.zeros((len(seasons), n_parcels))

    for i, season in enumerate(seasons):
        loso_comp, loso_var, _ = loso_models[season]
        # Use same k cutoff
        k_loso = min(k, loso_comp.shape[0])
        _, _, frac = compute_residual_variance(loso_comp, loso_var, k_loso)
        loso_fracs[i, :] = frac

    loso_mean = loso_fracs.mean(axis=0)
    loso_std = loso_fracs.std(axis=0)
    # CV = std / mean; avoid division by zero
    loso_cv = np.where(loso_mean > 1e-8, loso_std / loso_mean, 0.0)

    high_cv = [int(i) for i in np.where(loso_cv > LOSO_CV_THRESHOLD)[0]]

    return {
        'primary_frac': primary_frac,
        'loso_fracs': loso_fracs,
        'loso_mean': loso_mean,
        'loso_std': loso_std,
        'loso_cv': loso_cv,
        'high_cv_parcels': high_cv,
        'seasons': seasons,
    }


def plot_A5_loso_residual_stability(stability, labels, network_per_parcel,
                                     groups, k, output_dir, suffix=''):
    """Scatter: primary residual_frac vs mean LOSO residual_frac."""
    primary = stability['primary_frac']
    loso_mean = stability['loso_mean']
    loso_std = stability['loso_std']
    loso_cv = stability['loso_cv']
    n_folds = stability['loso_fracs'].shape[0]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Color by network
    colors = [NETWORK_COLORS.get(network_per_parcel[i], '#888888')
              for i in range(len(primary))]

    ax.errorbar(primary, loso_mean, yerr=loso_std, fmt='none',
                ecolor='lightgrey', elinewidth=0.5, alpha=0.5, zorder=1)
    ax.scatter(primary, loso_mean, c=colors, s=30, edgecolors='black',
               linewidths=0.3, alpha=0.8, zorder=2)

    # Diagonal reference
    lim_max = max(primary.max(), loso_mean.max()) * 1.1
    ax.plot([0, lim_max], [0, lim_max], 'k--', alpha=0.4, linewidth=1)

    # Label high-CV parcels
    for pidx in stability['high_cv_parcels']:
        ax.annotate(
            abbreviate_parcel_label(labels[pidx]),
            xy=(primary[pidx], loso_mean[pidx]),
            xytext=(5, 5), textcoords='offset points',
            fontsize=6, color='red', fontweight='bold',
        )

    ax.set_xlabel('Primary PCA residual fraction')
    ax.set_ylabel(f'Mean LOSO residual fraction ({n_folds} folds)')
    ax.set_title(f'A5. LOSO Residual Stability (k={k} PCs, '
                 f'{len(stability["high_cv_parcels"])} high-CV parcels)')
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Network legend (compact)
    from matplotlib.lines import Line2D
    seen = set()
    handles = []
    for net_name, _ in groups:
        if net_name not in seen:
            handles.append(Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=NETWORK_COLORS.get(net_name, '#888'),
                                  markersize=6, label=net_name))
            seen.add(net_name)
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=6, ncol=1, frameon=True)

    fname = f'A5_loso_residual_stability_{suffix}.png' if suffix else 'A5_loso_residual_stability.png'
    out_path = os.path.join(output_dir, fname)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


def save_loso_residual_csv(stability, labels, network_per_parcel, output_dir):
    """Save CSV with per-parcel LOSO residual variance stability."""
    rows = []
    for i in range(len(labels)):
        rows.append({
            'parcel_id': i + 1,
            'parcel_label': labels[i],
            'network': network_per_parcel[i],
            'primary_residual_frac': round(float(stability['primary_frac'][i]), 6),
            'loso_mean': round(float(stability['loso_mean'][i]), 6),
            'loso_std': round(float(stability['loso_std'][i]), 6),
            'loso_cv': round(float(stability['loso_cv'][i]), 6),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values('loso_cv', ascending=False)
    out_path = os.path.join(output_dir, 'pca_residual_variance_loso.csv')
    df.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path}")


# =============================================================================
# A6: Cross-Subject PC Comparison
# =============================================================================

def compute_cross_subject_pc_composition(parcellation, groups, n_pcs=3):
    """Load PCA models for all subjects and compute per-network PC composition.

    Returns
    -------
    composition : dict
        'subjects': list of sub_ids that were loaded
        'fractions': np.ndarray (n_subjects, n_pcs, n_networks)
        'net_names': list of network names
        'cosine_sim': dict of pairwise cosine similarities per PC
    """
    net_names = [g[0] for g in groups]
    n_nets = len(net_names)

    subjects_loaded = []
    all_components = []  # list of (n_components, n_parcels) arrays

    for sub_id in ALL_SUBJECTS:
        pca_path = os.path.join(
            _pca_output_dir(sub_id, parcellation), 'pca_model.pkl'
        )
        if not os.path.exists(pca_path):
            logger.warning(f"  Skipping {sub_id}: PCA model not found")
            continue
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
        all_components.append(pca.components_.copy())
        del pca
        subjects_loaded.append(sub_id)
        logger.info(f"  Loaded PCA for {sub_id}")

    if len(subjects_loaded) < 2:
        logger.warning("Need at least 2 subjects for cross-subject comparison")
        return None

    n_subjects = len(subjects_loaded)
    n_pcs_actual = min(n_pcs, min(c.shape[0] for c in all_components))
    fractions = np.zeros((n_subjects, n_pcs_actual, n_nets))

    for s_idx, comp in enumerate(all_components):
        for pc_idx in range(n_pcs_actual):
            total_sq = np.sum(comp[pc_idx, :] ** 2)
            if total_sq == 0:
                continue
            for g_idx, (net_name, indices) in enumerate(groups):
                sq_load = np.sum(comp[pc_idx, indices] ** 2)
                fractions[s_idx, pc_idx, g_idx] = sq_load / total_sq

    # Pairwise cosine similarity per PC
    cosine_sim = {}
    for pc_idx in range(n_pcs_actual):
        sim_matrix = np.zeros((n_subjects, n_subjects))
        for i in range(n_subjects):
            for j in range(i, n_subjects):
                vi = all_components[i][pc_idx, :]
                vj = all_components[j][pc_idx, :]
                norm_prod = np.linalg.norm(vi) * np.linalg.norm(vj)
                if norm_prod > 0:
                    # Use absolute cosine similarity (PCs can flip sign)
                    sim = abs(float(np.dot(vi, vj) / norm_prod))
                else:
                    sim = 0.0
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
        cosine_sim[f'PC{pc_idx + 1}'] = {
            'matrix': [[round(v, 4) for v in row] for row in sim_matrix.tolist()],
            'mean_off_diagonal': round(float(
                (sim_matrix.sum() - np.trace(sim_matrix)) /
                (n_subjects * (n_subjects - 1))
            ), 4),
        }

    return {
        'subjects': subjects_loaded,
        'fractions': fractions,
        'net_names': net_names,
        'cosine_sim': cosine_sim,
        'n_pcs': n_pcs_actual,
    }


def plot_A6_cross_subject_pc_composition(cross_data, output_dir):
    """Grouped bar chart: per-network PC composition across subjects."""
    subjects = cross_data['subjects']
    fractions = cross_data['fractions']  # (n_sub, n_pcs, n_nets)
    net_names = cross_data['net_names']
    n_pcs = cross_data['n_pcs']
    n_subjects = len(subjects)

    fig, axes = plt.subplots(n_pcs, 1, figsize=(14, 4 * n_pcs + 2))
    if n_pcs == 1:
        axes = [axes]

    bar_width = 0.8 / n_subjects
    subject_colors = plt.cm.Set2(np.linspace(0, 0.8, n_subjects))

    for pc_idx, ax in enumerate(axes):
        x = np.arange(len(net_names))
        for s_idx, sub_id in enumerate(subjects):
            offset = (s_idx - n_subjects / 2 + 0.5) * bar_width
            ax.bar(x + offset, fractions[s_idx, pc_idx, :],
                   width=bar_width, label=sub_id, color=subject_colors[s_idx],
                   edgecolor='white', linewidth=0.3)

        cos_sim = cross_data['cosine_sim'].get(f'PC{pc_idx + 1}', {})
        mean_sim = cos_sim.get('mean_off_diagonal', 'N/A')

        ax.set_ylabel('Fraction of squared loadings')
        ax.set_title(f'PC{pc_idx + 1} -- Network Composition '
                     f'(mean cosine sim = {mean_sim})')
        ax.set_xticks(x)
        ax.set_xticklabels(net_names, rotation=45, ha='right', fontsize=7)
        ax.set_ylim(0, None)
        ax.grid(True, axis='y', alpha=0.2)

        if pc_idx == 0:
            ax.legend(fontsize=8, ncol=n_subjects, loc='upper right')

    fig.suptitle('A6. Cross-Subject PC Composition', fontsize=14, y=1.01)

    out_path = os.path.join(output_dir, 'A6_cross_subject_pc_composition.png')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


def save_cross_subject_csv(cross_data, output_dir):
    """Save CSV with per-network PC composition across subjects."""
    rows = []
    for s_idx, sub_id in enumerate(cross_data['subjects']):
        for pc_idx in range(cross_data['n_pcs']):
            for g_idx, net_name in enumerate(cross_data['net_names']):
                rows.append({
                    'pc_index': pc_idx + 1,
                    'subject': sub_id,
                    'network': net_name,
                    'fraction': round(float(
                        cross_data['fractions'][s_idx, pc_idx, g_idx]
                    ), 6),
                })
    df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, 'pca_cross_subject_composition.csv')
    df.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path}")


# =============================================================================
# A7: Residual Variance vs. Variance Threshold
# =============================================================================

def plot_A7_residual_vs_threshold(components, explained_variance, groups,
                                   network_per_parcel, n_pcs_lookup, output_dir):
    """Sweep residual variance across all variance thresholds in n_pcs_lookup.

    Panel A7a: Median per-parcel residual fraction vs. threshold (with IQR).
    Panel A7b: Per-network mean residual fraction across thresholds.
    """
    thresholds = sorted(n_pcs_lookup.keys(), key=float)
    if len(thresholds) < 2:
        logger.warning("A7: Need at least 2 variance thresholds; skipping")
        return

    vt_floats = [float(t) for t in thresholds]
    ks = [n_pcs_lookup[t] for t in thresholds]

    # Compute residual fractions at each threshold
    n_parcels = components.shape[1]
    all_residual_fracs = np.zeros((len(thresholds), n_parcels))
    for i, k in enumerate(ks):
        _, _, residual_frac = compute_residual_variance(
            components, explained_variance, k
        )
        all_residual_fracs[i, :] = residual_frac

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- A7a: Median residual with IQR shading ---
    medians = np.median(all_residual_fracs, axis=1)
    q25 = np.percentile(all_residual_fracs, 25, axis=1)
    q75 = np.percentile(all_residual_fracs, 75, axis=1)

    ax1.plot(vt_floats, medians, 'o-', color='steelblue', linewidth=2,
             markersize=8, zorder=3)
    ax1.fill_between(vt_floats, q25, q75, alpha=0.25, color='steelblue',
                     label='IQR (25th-75th)')
    for i, (vt, med, k) in enumerate(zip(vt_floats, medians, ks)):
        ax1.annotate(f'k={k}', (vt, med), textcoords='offset points',
                     xytext=(0, 10), ha='center', fontsize=8, color='grey')
    ax1.set_xlabel('Variance Threshold')
    ax1.set_ylabel('Residual Variance Fraction')
    ax1.set_title('A7a. Median Per-Parcel Residual vs. Variance Threshold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(min(vt_floats) - 0.02, max(vt_floats) + 0.02)
    ax1.set_ylim(bottom=0)

    # --- A7b: Per-network residual across thresholds ---
    for net_name, indices in groups:
        if len(indices) < 2:
            continue
        net_means = all_residual_fracs[:, indices].mean(axis=1)
        color = NETWORK_COLORS.get(net_name, '#888888')
        ax2.plot(vt_floats, net_means, 'o-', color=color, linewidth=1.5,
                 markersize=5, alpha=0.85, label=net_name)

    ax2.set_xlabel('Variance Threshold')
    ax2.set_ylabel('Mean Residual Variance Fraction')
    ax2.set_title('A7b. Per-Network Mean Residual Across Thresholds')
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=6,
               ncol=1, frameon=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(min(vt_floats) - 0.02, max(vt_floats) + 0.02)
    ax2.set_ylim(bottom=0)

    out_path = os.path.join(output_dir, 'A7_residual_vs_threshold.png')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# =============================================================================
# CSV outputs (A1-A4)
# =============================================================================

def save_top_parcels_csv(components, var_ratio, labels, network_per_parcel,
                          n_top_pcs, n_top_parcels, output_dir):
    """Save CSV with top-loading parcels per PC."""
    rows = []
    n_pcs = min(n_top_pcs, components.shape[0])
    n_show = min(n_top_parcels, components.shape[1])

    for pc_idx in range(n_pcs):
        loadings = components[pc_idx, :]
        top_indices = np.argsort(np.abs(loadings))[::-1][:n_show]
        for rank, pidx in enumerate(top_indices):
            rows.append({
                'pc_index': pc_idx + 1,
                'pc_variance_pct': round(var_ratio[pc_idx] * 100, 2),
                'abs_rank': rank + 1,
                'parcel_id': pidx + 1,  # offset back to 1-based
                'parcel_label': labels[pidx],
                'network': network_per_parcel[pidx],
                'loading_value': round(float(loadings[pidx]), 6),
            })

    df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, 'pca_loadings_top_parcels.csv')
    df.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path}")


def save_residual_variance_csv(components, explained_variance, labels,
                                network_per_parcel, k, output_dir):
    """Save CSV with per-parcel residual variance breakdown."""
    signal_var, residual_var, residual_frac = compute_residual_variance(
        components, explained_variance, k
    )
    rows = []
    for i in range(len(labels)):
        rows.append({
            'parcel_id': i + 1,
            'parcel_label': labels[i],
            'network': network_per_parcel[i],
            'signal_var': round(float(signal_var[i]), 6),
            'residual_var': round(float(residual_var[i]), 6),
            'residual_fraction': round(float(residual_frac[i]), 6),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values('residual_fraction', ascending=False)
    out_path = os.path.join(output_dir, 'pca_residual_variance.csv')
    df.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path}")


def save_network_variance_csv(components, groups, n_pcs_show, output_dir):
    """Save CSV with per-network squared loading fraction per PC."""
    n_pcs = min(n_pcs_show, components.shape[0])
    rows = []
    for pc_idx in range(n_pcs):
        total_sq = np.sum(components[pc_idx, :] ** 2)
        if total_sq == 0:
            continue
        for net_name, indices in groups:
            sq_load = np.sum(components[pc_idx, indices] ** 2)
            rows.append({
                'pc_index': pc_idx + 1,
                'network': net_name,
                'sum_sq_loading': round(float(sq_load), 6),
                'fraction': round(float(sq_load / total_sq), 6),
            })
    df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, 'pca_network_variance.csv')
    df.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize and diagnose PCA loadings from fitted PCA models.'
    )
    parser.add_argument('--sub_id', type=str, required=True,
                        help='Subject ID (e.g., sub-01)')
    parser.add_argument('--parcellation', type=str, default='atlas-4S156Parcels',
                        help='Parcellation name (default: atlas-4S156Parcels)')
    parser.add_argument('--n_top_pcs', type=int, default=5,
                        help='Number of top PCs for A1/A2 (default: 5)')
    parser.add_argument('--n_top_parcels', type=int, default=15,
                        help='Number of top-loading parcels per PC in A2 (default: 15)')
    parser.add_argument('--variance_threshold', type=str, nargs='+', default=['0.90'],
                        help='Variance threshold(s) for A3/A5 (default: 0.90). '
                             'Accepts multiple values, e.g. --variance_threshold 0.80 0.85 0.90')
    parser.add_argument('--n_pcs_c4', type=int, default=20,
                        help='Number of PCs to show in A4 stacked bar (default: 20)')
    parser.add_argument('--plots', nargs='+',
                        default=['A1', 'A2', 'A3', 'A4', 'A5', 'A7'],
                        choices=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7'],
                        help='Which plots to generate (default: A1-A5, A7)')
    parser.add_argument('--cross_subject', action='store_true',
                        help='Generate A6 cross-subject PC comparison')
    parser.add_argument('--no_loso', action='store_true',
                        help='Skip LOSO residual stability (A5)')
    return parser.parse_args()


def main():
    args = parse_args()
    parc = normalize_parcellation_name(args.parcellation)

    # Per-subject output directory
    out_dir = os.path.join(SCRATCH_DIR, 'output', '03b_pca_loadings', parc, args.sub_id)
    os.makedirs(out_dir, exist_ok=True)

    logger.info('=' * 60)
    logger.info(f'PCA LOADINGS ANALYSIS: {args.sub_id} / {parc}')
    logger.info(f'Plots requested: {args.plots}')
    logger.info(f'Cross-subject: {args.cross_subject}')
    logger.info(f'Output: {out_dir}')
    logger.info('=' * 60)

    # Load data
    components, explained_variance, var_ratio = load_pca_model(args.sub_id, parc)
    labels = load_parcel_labels(parc)
    n_pcs_lookup = load_n_pcs_lookup(args.sub_id, parc)

    # Verify dimensions
    n_parcels = components.shape[1]
    if len(labels) != n_parcels:
        logger.error(f"Dimension mismatch: PCA has {n_parcels} features but "
                     f"found {len(labels)} labels")
        sys.exit(1)
    logger.info(f"PCA: {components.shape[0]} components x {n_parcels} parcels")

    # Group parcels by network
    groups, network_per_parcel = group_parcels_by_network(labels)
    logger.info(f"Network groups: {len(groups)}")
    for net_name, indices in groups:
        logger.info(f"  {net_name}: {len(indices)} parcels")

    # Resolve variance thresholds (may be multiple for A3/A5)
    vt_list = args.variance_threshold  # list of str
    vt_configs = []  # list of (vt_str, vt_float, k, suffix)
    for vt_str in vt_list:
        if vt_str in n_pcs_lookup:
            k = n_pcs_lookup[vt_str]
        else:
            logger.warning(f"Variance threshold '{vt_str}' not in n_pcs_lookup; "
                           f"available: {list(n_pcs_lookup.keys())}. Using 0.90.")
            k = n_pcs_lookup.get('0.90', 43)
        vt_float = float(vt_str)
        suffix = f"vt{int(vt_float * 100):03d}" if len(vt_list) > 1 else ''
        vt_configs.append((vt_str, vt_float, k, suffix))
        logger.info(f"Signal/residual cutoff: k={k} PCs (variance threshold={vt_str})")
    # Use first threshold as default for non-threshold-dependent operations
    _, _, k_default, _ = vt_configs[0]

    # =========================================================================
    # Motion artifact flagging (always runs)
    # =========================================================================
    logger.info('--- Motion Artifact Flagging ---')
    motion_flags = compute_motion_artifact_flags(components, groups)

    # Initialize flags JSON (will accumulate from all sections)
    flags_json = {'motion_artifact': motion_flags}

    # =========================================================================
    # Generate per-subject plots (A1-A4)
    # =========================================================================
    if 'A1' in args.plots:
        logger.info('--- A1: Loadings Heatmap ---')
        plot_A1_loadings_heatmap(components, var_ratio, labels, groups,
                                 args.n_top_pcs, out_dir)

    if 'A2' in args.plots:
        logger.info('--- A2: Top-Loading Parcels per PC ---')
        plot_A2_top_parcels_per_pc(components, var_ratio, labels,
                                    network_per_parcel, args.n_top_pcs,
                                    args.n_top_parcels, out_dir)

    if 'A3' in args.plots:
        for vt_str_i, vt_float_i, k_i, suffix_i in vt_configs:
            logger.info(f'--- A3: Residual Variance Fraction (vt={vt_str_i}) ---')
            plot_A3_residual_variance(components, explained_variance, labels,
                                      groups, network_per_parcel, k_i,
                                      vt_float_i, out_dir, suffix=suffix_i)

    if 'A4' in args.plots:
        logger.info('--- A4: Network Variance per PC ---')
        plot_A4_network_variance_per_pc(components, groups, args.n_pcs_c4,
                                         out_dir, motion_flags=motion_flags)

    # =========================================================================
    # A5: LOSO Residual Stability
    # =========================================================================
    if 'A5' in args.plots and not args.no_loso:
        loso_models = load_loso_pca_models(args.sub_id, parc)
        if loso_models:
            for vt_str_i, vt_float_i, k_i, suffix_i in vt_configs:
                logger.info(f'--- A5: LOSO Residual Stability (vt={vt_str_i}) ---')
                stability = compute_loso_residual_stability(
                    components, explained_variance, k_i,
                    loso_models, labels, network_per_parcel
                )
                plot_A5_loso_residual_stability(stability, labels,
                                                network_per_parcel, groups,
                                                k_i, out_dir, suffix=suffix_i)
                save_loso_residual_csv(stability, labels, network_per_parcel,
                                       out_dir)

            # Add to flags (use first threshold)
            _, _, k_first, _ = vt_configs[0]
            stability_first = compute_loso_residual_stability(
                components, explained_variance, k_first,
                loso_models, labels, network_per_parcel
            )
            flags_json['loso_stability'] = {
                'n_folds': len(stability_first['seasons']),
                'seasons': stability_first['seasons'],
                'n_high_cv_parcels': len(stability_first['high_cv_parcels']),
                'high_cv_parcel_ids': [i + 1 for i in stability_first['high_cv_parcels']],
                'high_cv_parcel_labels': [labels[i] for i in stability_first['high_cv_parcels']],
                'cv_threshold': LOSO_CV_THRESHOLD,
            }

            if stability_first['high_cv_parcels']:
                logger.warning(
                    f"LOSO STABILITY: {len(stability_first['high_cv_parcels'])} parcels "
                    f"with CV > {LOSO_CV_THRESHOLD}"
                )
            else:
                logger.info("LOSO stability check: PASSED (all parcels CV < "
                            f"{LOSO_CV_THRESHOLD})")
        else:
            logger.warning("No LOSO PCA models found; skipping A5")

    # =========================================================================
    # Save per-subject CSV outputs (always)
    # =========================================================================
    logger.info('--- Saving CSV outputs ---')
    save_top_parcels_csv(components, var_ratio, labels, network_per_parcel,
                          args.n_top_pcs, args.n_top_parcels, out_dir)
    save_residual_variance_csv(components, explained_variance, labels,
                                network_per_parcel, k_default, out_dir)
    save_network_variance_csv(components, groups, args.n_pcs_c4, out_dir)

    # Save flags JSON
    flags_path = os.path.join(out_dir, 'pca_loadings_flags.json')
    with open(flags_path, 'w') as f:
        json.dump(flags_json, f, indent=2)
    logger.info(f"Saved: {flags_path}")

    # =========================================================================
    # A6: Cross-Subject PC Comparison (optional, writes to shared dir)
    # =========================================================================
    if args.cross_subject or 'A6' in args.plots:
        logger.info('--- A6: Cross-Subject PC Comparison ---')
        cross_out_dir = os.path.join(
            SCRATCH_DIR, 'output', '03b_pca_loadings', parc, 'cross_subject'
        )
        os.makedirs(cross_out_dir, exist_ok=True)

        cross_data = compute_cross_subject_pc_composition(parc, groups, n_pcs=3)
        if cross_data is not None:
            plot_A6_cross_subject_pc_composition(cross_data, cross_out_dir)
            save_cross_subject_csv(cross_data, cross_out_dir)

            # Save cosine similarity to flags in cross_subject dir
            cross_flags = {'cosine_similarity': cross_data['cosine_sim']}
            cross_flags_path = os.path.join(cross_out_dir,
                                             'pca_cross_subject_flags.json')
            with open(cross_flags_path, 'w') as f:
                json.dump(cross_flags, f, indent=2)
            logger.info(f"Saved: {cross_flags_path}")

    # =========================================================================
    # A7: Residual Variance vs. Variance Threshold
    # =========================================================================
    if 'A7' in args.plots:
        logger.info('--- A7: Residual Variance vs. Threshold ---')
        plot_A7_residual_vs_threshold(components, explained_variance, groups,
                                      network_per_parcel, n_pcs_lookup, out_dir)

    logger.info(f'\nAll PCA loading diagnostics saved to: {out_dir}')


if __name__ == '__main__':
    main()
