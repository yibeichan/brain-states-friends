#!/usr/bin/env python3
"""
pp_03_project_pp_pca.py - Project Petit Prince parcel time series through Friends-trained PCA.

Loads the PCA model fitted on Friends training data (from 04_combined_hdphmm final/)
and projects each PP run's parcel time series into the same PCA space. Computes a
PCA transfer diagnostic (variance explained by Friends PCA on PP data).

Petit Prince is an audio-only stimulus (audiobook listening in French and English).
This tests whether the Friends PCA subspace - trained on audiovisual data - captures
meaningful variance in audio-only narrative listening. Visual cortex PCs are expected
to carry no stimulus-driven variance; auditory/language components should transfer well.

Prerequisites:
    - 00_postproc.py completed for petit-prince (cleaned CIFTIs)
    - 02_extract_parcel_ts.py completed for petit-prince episodes
    - 04_combined_hdphmm.py (mode: select) completed for this subject (Friends model)

Outputs:
    {SCRATCH_DIR}/output/pp_03_projected/{parcellation}/{sub_id}/[vt{VT}/]
        {run_id}.npy                  - PCA-projected PP run (n_trs, n_pcs)
        pp_run_ids.json               - Run IDs grouped by language type
        pca_transfer_diagnostic.json  - Variance explained diagnostics

Documentation: the design notes
"""

import os
import sys
import json
import glob
import pickle
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Add script directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils.common import normalize_parcellation_name
from utils.plot_style import (assign_network, NETWORK_ORDER, NETWORK_COLORS,
                              apply_publication_style)

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

# Petit Prince stimulus type prefixes (two languages of the same story)
PP_PREFIXES = {
    'lppFR': 'French',
    'lppEN': 'English',
}


def get_stimulus_type(run_id):
    """Extract stimulus type from a run ID string.

    Args:
        run_id: Full BIDS run ID (e.g., 'sub-01_ses-001_task-lppFR_run-1_space-fsLR_den-91k_part-mag')
                or just the task portion (e.g., 'lppFR_run-1')

    Returns:
        Stimulus type string ('lppFR' or 'lppEN') or None
    """
    for prefix in PP_PREFIXES:
        if f'task-{prefix}' in run_id or run_id.startswith(prefix):
            return prefix
    return None


def discover_pp_runs(parcel_ts_dir):
    """Discover Petit Prince run files in the parcel time series directory.

    Finds all .npy files with lppFR or lppEN task prefix, excluding Friends
    episodes (task-s*), Movie10 runs, and Harry Potter runs.

    Args:
        parcel_ts_dir: Path to 02_parcel_ts_avg/{parc}/{sub}/

    Returns:
        dict: stimulus_type -> list of (run_id, filepath) tuples
    """
    all_files = sorted(glob.glob(os.path.join(parcel_ts_dir, '*_parcel_avg.npy')))

    pp_runs = {stype: [] for stype in PP_PREFIXES}

    for fpath in all_files:
        fname = os.path.basename(fpath)
        run_id = fname.replace('_parcel_avg.npy', '')

        stype = get_stimulus_type(run_id)
        if stype is not None:
            pp_runs[stype].append((run_id, fpath))

    return pp_runs


def compute_pca_transfer_diagnostic_from_stats(pca, n_pcs, stats):
    """Compute PCA transfer diagnostic from pre-accumulated statistics.

    Avoids holding all data in memory simultaneously by using
    SSE/SST/sum/count accumulated per-run during projection.

    Args:
        pca: Fitted sklearn PCA model (from Friends training data)
        n_pcs: Number of PCs used in the HMM
        stats: dict with keys 'overall' and 'by_type', each containing
               'sst', 'sse_n', 'sse_full', 'parcel_sum', 'count'

    Returns:
        dict with diagnostic metrics
    """
    ov = stats['overall']
    n_parcels = len(pca.mean_)

    # Overall R²
    r_squared = 1.0 - ov['sse_n'] / ov['sst'] if ov['sst'] > 0 else 0.0
    r_squared_full = 1.0 - ov['sse_full'] / ov['sst'] if ov['sst'] > 0 else 0.0

    # Per-type R²
    r2_by_type = {}
    for stype, ts in stats['by_type'].items():
        if ts['count'] > 0 and ts['sst'] > 0:
            r2_by_type[stype] = {
                'r2_n_pcs': float(1.0 - ts['sse_n'] / ts['sst']),
                'r2_full': float(1.0 - ts['sse_full'] / ts['sst']),
                'n_trs': int(ts['count']),
            }

    # Per-parcel mean shift: PP parcel means vs Friends training means
    pp_parcel_means = ov['parcel_sum'] / ov['count'] if ov['count'] > 0 else np.zeros(n_parcels)
    mean_shift = pp_parcel_means - pca.mean_
    mean_shift_abs = np.abs(mean_shift)

    # Mean-corrected R²: removes mean shift contribution to isolate
    # covariance mismatch from baseline shift.
    if ov['count'] > 0:
        sum_x_sq = ov['sst'] + 2.0 * np.dot(pca.mean_, ov['parcel_sum']) \
            - ov['count'] * np.dot(pca.mean_, pca.mean_)
        sst_mc = sum_x_sq - ov['count'] * np.dot(pp_parcel_means, pp_parcel_means)

        shift = pca.mean_ - pp_parcel_means
        V_n = pca.components_[:n_pcs]
        shift_in_pca = (shift @ V_n.T) @ V_n
        delta = shift - shift_in_pca  # null-space component of mean shift
        sse_mc = ov['sse_n'] - ov['count'] * np.dot(delta, delta)

        r_squared_mc = 1.0 - sse_mc / sst_mc if sst_mc > 0 else 0.0
    else:
        r_squared_mc = 0.0

    # Friends training R² (from PCA explained_variance_ratio_)
    friends_r2_n_pcs = float(np.sum(pca.explained_variance_ratio_[:n_pcs]))
    friends_r2_full = float(np.sum(pca.explained_variance_ratio_))

    # Per-network R²
    r2_by_network = {}
    if 'by_network' in ov:
        for net in NETWORK_ORDER:
            nb = ov['by_network'].get(net)
            if nb and nb['sst'] > 0:
                r2_by_network[net] = {
                    'r2_n_pcs': float(1.0 - nb['sse_n'] / nb['sst']),
                    'n_parcels': int(len(ov['_network_indices'].get(net, []))),
                }

    diagnostic = {
        'stimulus': 'petit-prince',
        'stimulus_modality': 'audio_only',
        'pp_r2_n_pcs': float(r_squared),
        'pp_r2_n_pcs_mean_corrected': float(r_squared_mc),
        'pp_r2_full_pca': float(r_squared_full),
        'friends_r2_n_pcs': friends_r2_n_pcs,
        'friends_r2_full_pca': friends_r2_full,
        'n_pcs': n_pcs,
        'n_components_total': pca.n_components_,
        'n_pp_trs': int(ov['count']),
        'n_parcels': n_parcels,
        'transfer_gap': float(friends_r2_n_pcs - r_squared),
        'transfer_gap_mean_corrected': float(friends_r2_n_pcs - r_squared_mc),
        'flag_low_variance': bool(r_squared < 0.70),
        'r2_by_type': r2_by_type,
        'r2_by_network': r2_by_network,
        'parcel_mean_shift': {
            'mean_abs_shift': float(mean_shift_abs.mean()),
            'max_abs_shift': float(mean_shift_abs.max()),
            'n_parcels_shift_gt_1sd': int(np.sum(mean_shift_abs > 1.0)),
            'flag_mean_shift': bool(np.sum(mean_shift_abs > 1.0) > 0.1 * len(mean_shift)),
        },
        'note': ('Petit Prince is audio-only (audiobook listening). Visual cortex PCs '
                 'carry no stimulus-driven variance - low visual network R² is expected. '
                 'Auditory and language network components should transfer well.'),
    }

    return diagnostic


def load_network_labels(parcellation):
    """Load per-parcel network assignment using centralized definitions.

    Cortical parcels use the ``network_label`` column from the atlas TSV.
    Subcortical parcels are mapped via ``assign_network()`` from plot_style.py
    (13 groups: Yeo-7 + BG, Midbrain-DA, Midbrain-Diencephalic, Thalamus,
    Hipp/Amyg, Cerebellum).

    Args:
        parcellation: Full parcellation name (e.g., 'atlas-4S156Parcels')

    Returns:
        list of str: Network label per parcel (length = n_parcels), or None
                     if the atlas TSV is not found.
    """
    atlas_dir = os.environ.get('ATLAS_DIR', os.path.expanduser('~/atlases'))
    tsv_path = os.path.join(atlas_dir, parcellation, f'{parcellation}_dseg.tsv')
    if not os.path.exists(tsv_path):
        logger.warning(f"Atlas TSV not found: {tsv_path} - network-stratified R² will be skipped")
        return None

    df = pd.read_csv(tsv_path, sep='\t')
    networks = []
    for _, row in df.iterrows():
        net = row.get('network_label')
        if pd.isna(net):
            # Subcortical parcel - use assign_network() from plot_style
            subcort_net = assign_network(row['label'])
            networks.append(subcort_net if subcort_net else 'Unknown')
        else:
            networks.append(net)
    return networks


def _accumulate_r2_stats(pca, n_pcs, X, stats_bucket):
    """Accumulate SSE/SST/sum/count for online R² computation.

    Adds to stats_bucket in-place. Each bucket has keys:
    'sst', 'sse_n', 'sse_full', 'parcel_sum', 'count'.
    If stats_bucket has 'by_network', also accumulates per-network SSE/SST.

    Returns:
        X_proj: np.ndarray (n_trs, n_pcs) - truncated PCA projection, for
                reuse by caller (avoids redundant pca.transform call).
    """
    X_centered = X - pca.mean_
    stats_bucket['sst'] += np.sum(X_centered ** 2)

    # Single PCA transform - reused for truncated and full reconstruction
    X_full_proj = pca.transform(X)
    X_proj = X_full_proj[:, :n_pcs]
    X_recon = X_proj @ pca.components_[:n_pcs] + pca.mean_
    residuals_n = X - X_recon
    stats_bucket['sse_n'] += np.sum(residuals_n ** 2)

    # Full reconstruction (reuse cached transform)
    X_recon_full = pca.inverse_transform(X_full_proj)
    stats_bucket['sse_full'] += np.sum((X - X_recon_full) ** 2)

    stats_bucket['parcel_sum'] += X.sum(axis=0)
    stats_bucket['count'] += X.shape[0]

    # Per-network accumulation
    if 'by_network' in stats_bucket:
        for net, indices in stats_bucket['_network_indices'].items():
            net_bucket = stats_bucket['by_network'][net]
            net_bucket['sst'] += np.sum(X_centered[:, indices] ** 2)
            net_bucket['sse_n'] += np.sum(residuals_n[:, indices] ** 2)
            net_bucket['count'] += X.shape[0]

    return X_proj


def parse_args():
    parser = argparse.ArgumentParser(
        description='Project Petit Prince parcel time series through Friends-trained PCA.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script/pp_03_project_pp_pca.py --sub_id sub-01
  python script/pp_03_project_pp_pca.py --sub_id sub-01 --parcellation atlas-4S456Parcels
        """
    )
    parser.add_argument('--sub_id', type=str, required=True,
                        help='Subject ID (e.g., "sub-01")')
    parser.add_argument('--parcellation', type=str, default='atlas-4S156Parcels',
                        help='Parcellation name (default: atlas-4S156Parcels)')
    parser.add_argument('--vt', type=str, default=None,
                        help='Variance threshold subdirectory under final/ (e.g., 0.99). '
                             'Reads from final/vt{VT}/. If omitted, reads from final/ directly '
                             '(legacy path).')
    return parser.parse_args()


def main():
    args = parse_args()
    sub_id = args.sub_id
    parc = normalize_parcellation_name(args.parcellation)

    # =========================================================================
    # Input paths
    # =========================================================================

    # Friends model outputs
    if args.vt is not None:
        hmm_final_dir = os.path.join(SCRATCH_DIR, 'output', '04_combined_hdphmm', parc, sub_id,
                                     'final', f'vt{args.vt}')
    else:
        hmm_final_dir = os.path.join(SCRATCH_DIR, 'output', '04_combined_hdphmm', parc, sub_id, 'final')
    pca_model_path = os.path.join(hmm_final_dir, 'pca_model.pkl')
    results_path = os.path.join(hmm_final_dir, 'final_results.json')

    # PP parcel time series
    parcel_ts_dir = os.path.join(SCRATCH_DIR, 'output', '02_parcel_ts_avg', parc, sub_id)

    # Validate inputs
    for path, label in [(pca_model_path, 'PCA model'), (results_path, 'final_results.json')]:
        if not os.path.exists(path):
            logger.error(f"Missing {label}: {path}")
            sys.exit(1)

    if not os.path.isdir(parcel_ts_dir):
        logger.error(f"Missing parcel time series directory: {parcel_ts_dir}")
        logger.error("Run pp_00_postproc.sh and pp_02_extract_parcel_ts.sh first.")
        sys.exit(1)

    # =========================================================================
    # Load Friends PCA and metadata
    # =========================================================================

    with open(pca_model_path, 'rb') as f:
        pca = pickle.load(f)

    with open(results_path, 'r') as f:
        final_results = json.load(f)
    n_pcs = final_results['data_info']['n_pcs']

    logger.info(f"Loaded Friends PCA: {pca.n_components_} components, using {n_pcs} PCs")

    # =========================================================================
    # Discover and group PP runs
    # =========================================================================

    pp_runs = discover_pp_runs(parcel_ts_dir)

    total_runs = sum(len(runs) for runs in pp_runs.values())
    if total_runs == 0:
        logger.error(f"No PP runs found in {parcel_ts_dir}")
        logger.error("Expected files matching *task-lppFR*_parcel_avg.npy or *task-lppEN*_parcel_avg.npy")
        sys.exit(1)

    logger.info(f"Discovered {total_runs} PP runs:")
    for stype, runs in pp_runs.items():
        logger.info(f"  {PP_PREFIXES[stype]} ({stype}): {len(runs)} runs")

    # =========================================================================
    # Output directory
    # =========================================================================

    out_dir = os.path.join(SCRATCH_DIR, 'output', 'pp_03_projected', parc, sub_id)
    if args.vt is not None:
        out_dir = os.path.join(out_dir, f'vt{args.vt}')
    os.makedirs(out_dir, exist_ok=True)

    # =========================================================================
    # Project each PP run and collect data for diagnostic
    # =========================================================================

    def _new_stats_bucket(n_parcels, network_indices=None):
        bucket = {'sst': 0.0, 'sse_n': 0.0, 'sse_full': 0.0,
                  'parcel_sum': np.zeros(n_parcels), 'count': 0}
        if network_indices is not None:
            bucket['by_network'] = {
                net: {'sst': 0.0, 'sse_n': 0.0, 'count': 0}
                for net in network_indices
            }
            bucket['_network_indices'] = network_indices
        return bucket

    n_parcels = pca.n_features_in_

    # Build per-network parcel index mapping
    network_labels = load_network_labels(parc)
    network_indices = None
    if network_labels is not None and len(network_labels) == n_parcels:
        network_indices = {}
        for i, net in enumerate(network_labels):
            network_indices.setdefault(net, []).append(i)
        logger.info(f"Network mapping loaded: {', '.join(f'{n}({len(idx)})' for n, idx in network_indices.items())}")
    elif network_labels is not None:
        logger.warning(f"Network labels length ({len(network_labels)}) != n_parcels ({n_parcels}); skipping network R²")

    diag_overall = _new_stats_bucket(n_parcels, network_indices)
    diag_by_type = {stype: _new_stats_bucket(n_parcels) for stype in PP_PREFIXES}
    pp_run_ids = {}   # stimulus_type -> list of run_ids
    projected_count = 0

    for stype, runs in pp_runs.items():
        run_id_list = []
        for run_id, fpath in runs:
            # Load parcel time series
            X = np.load(fpath)  # (n_trs, n_parcels)

            # Drop background column (column 0) - same as 03a_pca4combined_hmm.py
            original_cols = X.shape[1]
            if original_cols in (157, 457, 557, 657, 757, 857, 957, 1057):
                X = X[:, 1:]

            # Validate finite values - same as 03a_pca4combined_hmm.py
            if not np.all(np.isfinite(X)):
                n_bad = (~np.isfinite(X)).sum()
                pct = n_bad / X.size * 100
                if pct < 0.1:
                    logger.warning(
                        f"{run_id}: {n_bad} non-finite values ({pct:.4f}%), replacing with 0"
                    )
                    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    logger.error(
                        f"{run_id}: {n_bad} non-finite values ({pct:.2f}%) - too many, skipping"
                    )
                    continue

            if X.shape[1] != pca.n_features_in_:
                logger.error(
                    f"{run_id}: n_parcels={X.shape[1]} != PCA n_features={pca.n_features_in_}. "
                    "Parcellation mismatch between PP data and Friends PCA model."
                )
                sys.exit(1)
            logger.info(f"  {run_id}: shape {X.shape}")

            # Accumulate R² statistics online (avoids large concatenation)
            X_proj = _accumulate_r2_stats(pca, n_pcs, X, diag_overall)
            _accumulate_r2_stats(pca, n_pcs, X, diag_by_type[stype])

            # Save projected data
            out_path = os.path.join(out_dir, f'{run_id}.npy')
            np.save(out_path, X_proj)
            projected_count += 1

            run_id_list.append(run_id)

        pp_run_ids[stype] = run_id_list

    logger.info(f"Projected {projected_count} PP runs to {n_pcs} PCs")

    # =========================================================================
    # Save PP run ID grouping
    # =========================================================================

    run_ids_path = os.path.join(out_dir, 'pp_run_ids.json')
    with open(run_ids_path, 'w') as f:
        json.dump(pp_run_ids, f, indent=2)

    # =========================================================================
    # PCA transfer diagnostic (from online-accumulated statistics)
    # =========================================================================

    logger.info(f"Computing PCA transfer diagnostic on {diag_overall['count']} total PP TRs")

    diag_stats = {'overall': diag_overall, 'by_type': diag_by_type}
    diagnostic = compute_pca_transfer_diagnostic_from_stats(pca, n_pcs, diag_stats)

    diag_path = os.path.join(out_dir, 'pca_transfer_diagnostic.json')
    with open(diag_path, 'w') as f:
        json.dump(diagnostic, f, indent=2)

    # =========================================================================
    # Report
    # =========================================================================

    print(f"\n{'='*60}")
    print(f"PCA TRANSFER DIAGNOSTIC (Petit Prince Audio)")
    print(f"{'='*60}")
    print(f"Subject:            {sub_id}")
    print(f"Parcellation:       {parc}")
    print(f"PP runs:            {projected_count}")
    print(f"PCs used:           {n_pcs}")
    print(f"Friends R² (n_pcs): {diagnostic['friends_r2_n_pcs']:.4f}")
    print(f"PP R² (n_pcs):      {diagnostic['pp_r2_n_pcs']:.4f}")
    print(f"PP R² (mean-corr):  {diagnostic['pp_r2_n_pcs_mean_corrected']:.4f}")
    print(f"Transfer gap:       {diagnostic['transfer_gap']:.4f}")
    print(f"Transfer gap (mc):  {diagnostic['transfer_gap_mean_corrected']:.4f}")
    if diagnostic['flag_low_variance']:
        print(f"*** WARNING: PP R² < 0.70 - PCA subspace may not capture PP variance ***")
    print(f"\nPer-type R² (n_pcs):")
    for stype, r2_info in diagnostic['r2_by_type'].items():
        label = PP_PREFIXES.get(stype, stype)
        print(f"  {label:15s}: {r2_info['r2_n_pcs']:.4f}  ({r2_info['n_trs']} TRs)")
    if diagnostic.get('r2_by_network'):
        print(f"\nPer-network R² (n_pcs):")
        for net in NETWORK_ORDER:
            if net in diagnostic['r2_by_network']:
                info = diagnostic['r2_by_network'][net]
                print(f"  {net:15s}: {info['r2_n_pcs']:.4f}  ({info['n_parcels']} parcels)")
    ms = diagnostic['parcel_mean_shift']
    print(f"\nParcel mean shift (PP vs Friends):")
    print(f"  Mean |shift|:     {ms['mean_abs_shift']:.4f}")
    print(f"  Max |shift|:      {ms['max_abs_shift']:.4f}")
    print(f"  Parcels > 1 SD:   {ms['n_parcels_shift_gt_1sd']}")
    if ms['flag_mean_shift']:
        print(f"*** WARNING: >10% of parcels have mean shift > 1 SD - check preprocessing ***")
    print(f"{'='*60}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")

    # =========================================================================
    # Diagnostic figures
    # =========================================================================

    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        import matplotlib.colors as mcolors
        from utils.viz_yabplot import (
            setup_yabplot_headless, load_parcel_labels,
            pattern_to_cortical_dict, pattern_to_subcortical_dict,
            render_cortical_to_image, render_subcortical_to_image,
            get_subcortical_atlas_dir,
        )
        setup_yabplot_headless()
        apply_publication_style()
        labels_df = load_parcel_labels(parc)
        subcort_atlas_dir = get_subcortical_atlas_dir()
        vt_str = f', vt={args.vt}' if args.vt else ''

        def _render_pca_diagnostic_figure(
            r2_by_network, overall_r2, scope_label, r2_bar_items, out_path,
        ):
            """Render a PCA transfer diagnostic figure."""
            # Build parcel-level R² array from per-network values
            r2_pattern = np.full(n_parcels, np.nan)
            if network_labels is not None and r2_by_network:
                for i, nl in enumerate(network_labels):
                    net_info = r2_by_network.get(nl)
                    if net_info:
                        r2_pattern[i] = net_info['r2_n_pcs']

            # Render brain surfaces
            cort_dict = pattern_to_cortical_dict(r2_pattern, labels_df, parc)
            sub_dict = pattern_to_subcortical_dict(r2_pattern, labels_df, parc)
            cort_img = render_cortical_to_image(cort_dict, color_range=(0.0, 1.0), cmap='viridis')
            sub_img = render_subcortical_to_image(
                sub_dict, color_range=(0.0, 1.0),
                atlas_dir=subcort_atlas_dir, cmap='viridis')

            # Compose figure
            fig = plt.figure(figsize=(14, 10))
            gs = GridSpec(3, 2, figure=fig,
                          height_ratios=[3, 2, 1.2],
                          hspace=0.30, wspace=0.3)

            # Row 0: brain surfaces
            ax_cort = fig.add_subplot(gs[0, 0])
            ax_cort.imshow(cort_img)
            ax_cort.set_title('Cortical PCA R² by Network')
            ax_cort.axis('off')
            ax_sub = fig.add_subplot(gs[0, 1])
            ax_sub.imshow(sub_img)
            ax_sub.set_title('Subcortical PCA R² by Network')
            ax_sub.axis('off')

            # Horizontal colorbar
            ax_cb = fig.add_axes([0.15, 0.635, 0.7, 0.015])
            sm = plt.cm.ScalarMappable(cmap='viridis',
                                       norm=mcolors.Normalize(vmin=0.0, vmax=1.0))
            sm.set_array([])
            fig.colorbar(sm, cax=ax_cb, orientation='horizontal')
            ax_cb.set_title('PCA R² (network-level)', fontsize=9, pad=2)

            # Row 1: per-network R² bar chart
            ax_net = fig.add_subplot(gs[1, :])
            nets_present = [n for n in NETWORK_ORDER if n in (r2_by_network or {})]
            r2_vals = [r2_by_network[n]['r2_n_pcs'] for n in nets_present]
            colors = [NETWORK_COLORS.get(n, '#888888') for n in nets_present]
            y_pos = np.arange(len(nets_present))
            ax_net.barh(y_pos, r2_vals, color=colors, edgecolor='white', linewidth=0.5)
            ax_net.set_yticks(y_pos)
            ax_net.set_yticklabels(nets_present)
            ax_net.set_xlabel('R² (n PCs)')
            ax_net.set_xlim(0, 1.05)
            ax_net.axvline(overall_r2, color='black', ls='--', lw=0.8,
                           label=f"Overall R²={overall_r2:.3f}")
            ax_net.axvline(0.70, color='red', ls=':', lw=0.8, alpha=0.6, label='Threshold 0.70')
            for i, v in enumerate(r2_vals):
                ax_net.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=7)
            ax_net.legend(fontsize=7, loc='lower right')
            ax_net.set_title(f'Per-Network PCA Transfer R² - {scope_label}')
            ax_net.invert_yaxis()

            # Row 2: R² comparison bars
            ax_ov = fig.add_subplot(gs[2, :])
            bar_labels = [item[0] for item in r2_bar_items]
            bar_vals = [item[1] for item in r2_bar_items]
            bar_colors = [item[2] for item in r2_bar_items]
            bars = ax_ov.bar(bar_labels, bar_vals, color=bar_colors,
                             edgecolor='white', width=0.5)
            ax_ov.set_ylim(0, 1.05)
            ax_ov.axhline(0.70, color='red', ls=':', lw=0.8, alpha=0.6)
            for bar, v in zip(bars, bar_vals):
                ax_ov.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f'{v:.3f}',
                           ha='center', va='bottom', fontsize=8)
            ax_ov.set_ylabel('R²')
            gap = diagnostic['friends_r2_n_pcs'] - overall_r2
            ax_ov.set_title(f"Transfer gap: {gap:.4f}")

            fig.suptitle(
                f'PCA Transfer: Friends → PP - {scope_label} ({sub_id}{vt_str})',
                fontsize=12, y=0.98)
            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved diagnostic figure: {os.path.basename(out_path)}")

        # -- Overall diagnostic figure --
        _render_pca_diagnostic_figure(
            r2_by_network=diagnostic.get('r2_by_network', {}),
            overall_r2=diagnostic['pp_r2_n_pcs'],
            scope_label='Overall',
            r2_bar_items=[
                ('Friends R²', diagnostic['friends_r2_n_pcs'], '#4682B4'),
                ('PP R²', diagnostic['pp_r2_n_pcs'], '#CD3E4E'),
                ('PP R² (mc)', diagnostic['pp_r2_n_pcs_mean_corrected'], '#E69422'),
            ],
            out_path=os.path.join(out_dir, 'pca_transfer_diagnostic.png'),
        )

    except Exception as e:
        logger.warning(f"Could not generate diagnostic figures: {e}")


if __name__ == '__main__':
    main()
