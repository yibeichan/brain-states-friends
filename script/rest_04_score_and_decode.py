#!/usr/bin/env python3
"""
rest_04_score_and_decode.py - Score and decode hcptrt resting-state data with Friends-trained HMM.

Uses the Friends-trained weak-limit HMM to score (log-likelihood) and decode (Viterbi)
resting-state runs projected through the Friends PCA. Computes per-run metrics for
cross-stimulus comparison.

Resting state is a task-free condition (no stimulus, eyes open, ~15 min, 4-6 runs
per subject across all six subjects). This tests whether Friends-trained brain states
produce meaningful state sequences from spontaneous (unstimulated) brain activity.

Prerequisites:
    - rest_03_project_rest_pca.py completed (projected rest data + rest_run_ids.json)
    - 04_combined_hdphmm.py (mode: select) completed (Friends model)

Outputs:
    {SCRATCH_DIR}/output/rest_04_decoded/{parcellation}/{sub_id}/
        decoded_states.pkl         - dict: short run_id -> np.array(n_trs,) state indices
        fractional_occupancy.pkl   - dict: short run_id -> np.array(n_states,)
        run_id_map.json            - long BIDS <-> short run_id mapping (rest_05 input)
        rest_ll_summary.json       - Per-run LL, overall LL, baselines
        ll_diagnostic.png          - Per-run LL + active-states diagnostic figure

Documentation: the design notes
"""

import os
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path

import numpy as np

# Add script directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils.common import (canonicalize_and_save_decoded, normalize_parcellation_name,
                          short_run_label)

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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Score and decode hcptrt resting-state data with Friends-trained HMM.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script/rest_04_score_and_decode.py --sub_id sub-01
  python script/rest_04_score_and_decode.py --sub_id sub-01 --parcellation atlas-4S456Parcels
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

    # Friends model
    if args.vt is not None:
        hmm_final_dir = os.path.join(SCRATCH_DIR, 'output', '04_combined_hdphmm', parc, sub_id,
                                     'final', f'vt{args.vt}')
    else:
        hmm_final_dir = os.path.join(SCRATCH_DIR, 'output', '04_combined_hdphmm', parc, sub_id, 'final')
    model_path = os.path.join(hmm_final_dir, 'best_model.pkl')
    results_path = os.path.join(hmm_final_dir, 'final_results.json')

    # Projected rest data (from rest_03)
    proj_dir = os.path.join(SCRATCH_DIR, 'output', 'rest_03_projected', parc, sub_id)
    if args.vt is not None:
        proj_dir = os.path.join(proj_dir, f'vt{args.vt}')
    run_ids_path = os.path.join(proj_dir, 'rest_run_ids.json')

    # Validate inputs
    for path, label in [
        (model_path, 'Friends HMM model'),
        (results_path, 'final_results.json'),
        (run_ids_path, 'rest_run_ids.json'),
    ]:
        if not os.path.exists(path):
            logger.error(f"Missing {label}: {path}")
            sys.exit(1)

    # =========================================================================
    # Load Friends model and metadata
    # =========================================================================

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(results_path, 'r') as f:
        final_results = json.load(f)

    n_states = final_results['model_info']['n_states']
    n_pcs = final_results['data_info']['n_pcs']
    n_active_states = final_results['final_refit']['n_active_states']
    friends_test_ll = final_results['final_refit']['test_ll_per_sample']

    logger.info(f"Loaded Friends HMM: {n_states} states ({n_active_states} active), {n_pcs} PCs")
    logger.info(f"Friends test LL/sample: {friends_test_ll:.4f}")

    # Load rest run IDs
    with open(run_ids_path, 'r') as f:
        rest_run_ids = json.load(f)

    # =========================================================================
    # Output directory
    # =========================================================================

    if args.vt is not None:
        out_dir = os.path.join(SCRATCH_DIR, 'output', 'rest_04_decoded', parc, sub_id, f'vt{args.vt}')
    else:
        out_dir = os.path.join(SCRATCH_DIR, 'output', 'rest_04_decoded', parc, sub_id)
    os.makedirs(out_dir, exist_ok=True)

    # =========================================================================
    # Score and decode each rest run
    # =========================================================================

    decoded_states = {}
    per_run_ll = {}       # run_id -> ll_per_sample
    per_run_n_trs = {}    # run_id -> n_trs

    for stype, run_ids in rest_run_ids.items():
        logger.info(f"Processing {stype}: {len(run_ids)} runs")
        for run_id in run_ids:
            proj_path = os.path.join(proj_dir, f'{run_id}.npy')
            if not os.path.exists(proj_path):
                logger.warning(f"Missing projected data: {proj_path}, skipping")
                continue

            X = np.load(proj_path)  # (n_trs, n_pcs)
            if X.shape[1] != n_pcs:
                logger.error(f"{run_id}: expected {n_pcs} PCs but got {X.shape[1]}, skipping")
                continue
            n_trs = X.shape[0]

            # Score: model.score() returns total log-likelihood
            total_ll = model.score(X)
            ll_per_sample = total_ll / n_trs

            # Decode: Viterbi algorithm
            _, state_seq = model.decode(X)

            decoded_states[run_id] = state_seq
            per_run_ll[run_id] = float(ll_per_sample)
            per_run_n_trs[run_id] = int(n_trs)

            logger.info(f"  {run_id}: {n_trs} TRs, LL/sample={ll_per_sample:.4f}, "
                         f"active states={len(np.unique(state_seq))}")

    del model  # free memory

    total_runs_decoded = len(decoded_states)
    if total_runs_decoded == 0:
        logger.error("No rest runs were decoded - check rest_03 outputs")
        sys.exit(1)
    logger.info(f"Decoded {total_runs_decoded} rest runs")

    # =========================================================================
    # Aggregate LL metrics
    # =========================================================================

    # Overall rest LL (weighted average by n_trs)
    total_trs = sum(per_run_n_trs.values())
    overall_rest_ll = sum(
        per_run_ll[rid] * per_run_n_trs[rid] for rid in per_run_ll
    ) / total_trs if total_trs > 0 else 0.0

    # Per-type LL (rest has only one type, but keep structure for consistency)
    per_type_ll = {}
    for stype, run_ids in rest_run_ids.items():
        type_trs = sum(per_run_n_trs.get(rid, 0) for rid in run_ids)
        if type_trs > 0:
            type_ll = sum(
                per_run_ll.get(rid, 0) * per_run_n_trs.get(rid, 0) for rid in run_ids
            ) / type_trs
        else:
            type_ll = None
        per_type_ll[stype] = {
            'll_per_sample': float(type_ll) if type_ll is not None else None,
            'n_runs': len([rid for rid in run_ids if rid in per_run_ll]),
            'n_trs': int(type_trs),
        }

    # Per-run LL variance metrics (unweighted, treating each run equally)
    run_ll_values = np.array(list(per_run_ll.values()))
    run_ll_mean = float(np.mean(run_ll_values))
    run_ll_std = float(np.std(run_ll_values, ddof=1))
    run_ll_se = float(run_ll_std / np.sqrt(len(run_ll_values)))

    # Baseline: uniform state assignment (heuristic reference point only)
    baseline_ll = float(np.log(1.0 / n_active_states)) if n_active_states > 0 else float('-inf')

    # =========================================================================
    # Build summary
    # =========================================================================

    ll_summary = {
        'subject': sub_id,
        'parcellation': parc,
        'stimulus': 'restingstate',
        'stimulus_modality': 'task_free',
        'n_states': n_states,
        'n_active_states': n_active_states,
        'n_pcs': n_pcs,
        'friends_test_ll_per_sample': float(friends_test_ll),
        'rest_overall_ll_per_sample': float(overall_rest_ll),
        'rest_ll_per_run_mean': run_ll_mean,
        'rest_ll_per_run_std': run_ll_std,
        'rest_ll_per_run_se': run_ll_se,
        'rest_total_trs': int(total_trs),
        'rest_total_runs': total_runs_decoded,
        'baseline_ll_per_sample': baseline_ll,
        'baseline_note': ('Heuristic reference point: log(1/n_active_states) is '
                          'not on the same scale as Gaussian-emission HMM LL.'),
        'll_gap_friends_minus_rest': float(friends_test_ll - overall_rest_ll),
        'rest_above_baseline': bool(overall_rest_ll > baseline_ll),
        'per_type': per_type_ll,
        'per_run': {rid: {'ll_per_sample': per_run_ll[rid], 'n_trs': per_run_n_trs[rid]}
                    for rid in per_run_ll},
    }

    # =========================================================================
    # Save outputs
    # =========================================================================

    # Canonicalize keys to 08c-compatible short form ('rest_ses-NNN') and
    # save decoded states, FO, and run_id_map.json (required by rest_05).
    long_to_short, decoded_states_short, fo_short = canonicalize_and_save_decoded(
        decoded_states, out_dir, "restingstate", n_states)

    with open(os.path.join(out_dir, 'rest_ll_summary.json'), 'w') as f:
        json.dump(ll_summary, f, indent=2)

    # =========================================================================
    # Report
    # =========================================================================

    print(f"\n{'='*60}")
    print(f"RESTING STATE SCORE & DECODE SUMMARY")
    print(f"{'='*60}")
    print(f"Subject:                {sub_id}")
    print(f"Parcellation:           {parc}")
    print(f"Rest runs decoded:      {total_runs_decoded}")
    print(f"Total rest TRs:         {total_trs}")
    print(f"Friends test LL/sample: {friends_test_ll:.4f}")
    print(f"Rest overall LL/sample: {overall_rest_ll:.4f}")
    print(f"Baseline LL/sample:     {baseline_ll:.4f}")
    print(f"LL gap (Friends-Rest):  {friends_test_ll - overall_rest_ll:.4f}")
    print(f"Rest > baseline:        {overall_rest_ll > baseline_ll}")
    print(f"")
    print(f"Per-run LL/sample:")
    for rid in sorted(per_run_ll.keys()):
        print(f"  {rid}: {per_run_ll[rid]:.4f} ({per_run_n_trs[rid]} TRs)")
    print(f"{'='*60}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")

    # =========================================================================
    # Diagnostic figure
    # =========================================================================

    try:
        import matplotlib.pyplot as plt
        from utils.plot_style import apply_publication_style, NETWORK_COLORS
        apply_publication_style()

        sorted_rids = sorted(per_run_ll.keys())
        n_runs = len(sorted_rids)
        # Compact tick labels ('ses-NNN'; hcptrt rest runs are all run-1, the
        # session token is the run identity) from the canonical short id
        short_labels = [short_run_label(long_to_short.get(rid, rid))
                        for rid in sorted_rids]

        fig, (ax_ll, ax_st) = plt.subplots(1, 2, figsize=(10, 4))

        # Left: per-run LL dot chart
        ll_vals = [per_run_ll[rid] for rid in sorted_rids]
        ax_ll.scatter(range(n_runs), ll_vals, color=NETWORK_COLORS['Default'],
                      s=40, zorder=3, edgecolors='white', linewidths=0.5)
        ax_ll.axhline(friends_test_ll, color='#4682B4', ls='-', lw=1.2,
                      label=f'Friends test LL ({friends_test_ll:.2f})')
        ax_ll.axhline(overall_rest_ll, color=NETWORK_COLORS['Default'], ls='--', lw=1.0,
                      label=f'Rest mean LL ({overall_rest_ll:.2f})')
        ax_ll.set_xticks(range(n_runs))
        ax_ll.set_xticklabels(short_labels, rotation=45, ha='right')
        ax_ll.set_ylabel('LL / sample')
        ax_ll.set_title('Per-Run Log-Likelihood')
        ax_ll.legend(fontsize=7, loc='lower left')
        gap = friends_test_ll - overall_rest_ll
        ax_ll.text(0.97, 0.97, f'gap={gap:+.2f}', transform=ax_ll.transAxes,
                   ha='right', va='top', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Right: active states per run
        states_per_run = [len(np.unique(decoded_states[rid])) for rid in sorted_rids]
        ax_st.bar(range(n_runs), states_per_run, color=NETWORK_COLORS['Default'],
                  edgecolor='white', linewidth=0.5, alpha=0.8)
        ax_st.axhline(n_active_states, color='#4682B4', ls='--', lw=1.0,
                      label=f'Friends active ({n_active_states})')
        ax_st.set_xticks(range(n_runs))
        ax_st.set_xticklabels(short_labels, rotation=45, ha='right')
        ax_st.set_ylabel('# Active States')
        ax_st.set_title('States Used per Run')
        ax_st.legend(fontsize=7)

        vt_str = f', vt={args.vt}' if args.vt else ''
        fig.suptitle(f'Rest Score & Decode ({sub_id}{vt_str})', fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'll_diagnostic.png'), bbox_inches='tight')
        plt.close(fig)
        logger.info("Saved diagnostic figure: ll_diagnostic.png")
    except Exception as e:
        logger.warning(f"Could not generate diagnostic figure: {e}")


if __name__ == '__main__':
    main()
