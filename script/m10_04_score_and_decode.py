#!/usr/bin/env python3
"""
m10_04_score_and_decode.py - Score and decode movie10 data with Friends-trained HMM.

Uses the Friends-trained weak-limit HMM to score (log-likelihood) and decode (Viterbi)
movie runs projected through the Friends PCA. Computes per-run and per-movie-type
metrics for cross-stimulus comparison.

Prerequisites:
    - m10_03_project_movie_pca.py completed (projected movie data + movie_run_ids.json)
    - 04_combined_hdphmm.py (mode: select) completed (Friends model)

Outputs:
    {SCRATCH_DIR}/output/m10_04_decoded/{parcellation}/{sub_id}/
        decoded_states.pkl         - dict: run_id -> np.array(n_trs,) state indices
        fractional_occupancy.pkl   - dict: run_id -> np.array(n_states,)
        movie_ll_summary.json      - Per-run LL, per-type LL, overall LL, baselines

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
from utils.common import normalize_cross_stim_run_id, normalize_parcellation_name

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


def compute_fractional_occupancy(decoded_states, n_states):
    """Compute fractional occupancy per run.

    FO_k_e = count(state_seq == k) / len(state_seq)

    Args:
        decoded_states: dict run_id -> np.array(n_trs,)
        n_states: Total number of model states (n_components)

    Returns:
        fo: dict run_id -> np.array(n_states,)
    """
    fo = {}
    for run_id, state_seq in decoded_states.items():
        if len(state_seq) == 0:
            logger.warning(f"Empty state sequence for {run_id}")
            fo[run_id] = np.zeros(n_states)
            continue
        counts = np.bincount(state_seq, minlength=n_states).astype(float)
        fo[run_id] = counts / len(state_seq)
    return fo


def parse_args():
    parser = argparse.ArgumentParser(
        description='Score and decode movie10 data with Friends-trained HMM.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script/m10_04_score_and_decode.py --sub_id sub-01
  python script/m10_04_score_and_decode.py --sub_id sub-01 --parcellation atlas-4S456Parcels
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

    # Projected movie data (from m10_03)
    proj_dir = os.path.join(SCRATCH_DIR, 'output', 'm10_03_projected', parc, sub_id)
    if args.vt is not None:
        proj_dir = os.path.join(proj_dir, f'vt{args.vt}')
    run_ids_path = os.path.join(proj_dir, 'movie_run_ids.json')

    # Validate inputs
    for path, label in [
        (model_path, 'Friends HMM model'),
        (results_path, 'final_results.json'),
        (run_ids_path, 'movie_run_ids.json'),
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

    # Load movie run IDs
    with open(run_ids_path, 'r') as f:
        movie_run_ids = json.load(f)

    # =========================================================================
    # Output directory
    # =========================================================================

    if args.vt is not None:
        out_dir = os.path.join(SCRATCH_DIR, 'output', 'm10_04_decoded', parc, sub_id, f'vt{args.vt}')
    else:
        out_dir = os.path.join(SCRATCH_DIR, 'output', 'm10_04_decoded', parc, sub_id)
    os.makedirs(out_dir, exist_ok=True)

    # =========================================================================
    # Score and decode each movie run
    # =========================================================================

    decoded_states = {}
    per_run_ll = {}       # run_id -> ll_per_sample
    per_run_n_trs = {}    # run_id -> n_trs

    for mtype, run_ids in movie_run_ids.items():
        logger.info(f"Processing {mtype}: {len(run_ids)} runs")
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
        logger.error("No movie runs were decoded - check m10_03 outputs")
        sys.exit(1)
    logger.info(f"Decoded {total_runs_decoded} movie runs")

    # =========================================================================
    # Aggregate LL metrics
    # =========================================================================

    # Overall movie LL (weighted average by n_trs)
    total_trs = sum(per_run_n_trs.values())
    overall_movie_ll = sum(
        per_run_ll[rid] * per_run_n_trs[rid] for rid in per_run_ll
    ) / total_trs if total_trs > 0 else 0.0

    # Per-movie-type LL
    per_type_ll = {}
    for mtype, run_ids in movie_run_ids.items():
        type_trs = sum(per_run_n_trs.get(rid, 0) for rid in run_ids)
        if type_trs > 0:
            type_ll = sum(
                per_run_ll.get(rid, 0) * per_run_n_trs.get(rid, 0) for rid in run_ids
            ) / type_trs
        else:
            type_ll = None
        per_type_ll[mtype] = {
            'll_per_sample': float(type_ll) if type_ll is not None else None,
            'n_runs': len([rid for rid in run_ids if rid in per_run_ll]),
            'n_trs': int(type_trs),
        }

    # Per-run LL variance metrics (unweighted, treating each run equally)
    run_ll_values = np.array(list(per_run_ll.values()))
    run_ll_mean = float(np.mean(run_ll_values))
    run_ll_std = float(np.std(run_ll_values, ddof=1))
    run_ll_se = float(run_ll_std / np.sqrt(len(run_ll_values)))

    # Baseline: uniform state assignment (heuristic reference point only).
    # This is log(1/K), not on the same scale as Gaussian-emission HMM LL
    # which includes emission log-probabilities. A single-state Gaussian
    # fitted to the data would be a principled null on the correct scale.
    baseline_ll = float(np.log(1.0 / n_active_states)) if n_active_states > 0 else float('-inf')

    # =========================================================================
    # Build summary
    # =========================================================================

    ll_summary = {
        'subject': sub_id,
        'parcellation': parc,
        'n_states': n_states,
        'n_active_states': n_active_states,
        'n_pcs': n_pcs,
        'friends_test_ll_per_sample': float(friends_test_ll),
        'movie_overall_ll_per_sample': float(overall_movie_ll),
        'movie_ll_per_run_mean': run_ll_mean,
        'movie_ll_per_run_std': run_ll_std,
        'movie_ll_per_run_se': run_ll_se,
        'movie_total_trs': int(total_trs),
        'movie_total_runs': total_runs_decoded,
        'baseline_ll_per_sample': baseline_ll,
        'baseline_note': ('Heuristic reference point: log(1/n_active_states) is '
                          'not on the same scale as Gaussian-emission HMM LL.'),
        'll_gap_friends_minus_movie': float(friends_test_ll - overall_movie_ll),
        'movie_above_baseline': bool(overall_movie_ll > baseline_ll),
        'per_type': per_type_ll,
        'per_run': {rid: {'ll_per_sample': per_run_ll[rid], 'n_trs': per_run_n_trs[rid]}
                    for rid in per_run_ll},
    }

    # =========================================================================
    # Save outputs
    # =========================================================================

    # Canonicalize keys to 08c-compatible short form (e.g. 'bourne01') so
    # downstream transformer / findings scripts can join decoded_states
    # with 08c feature files directly. run_id_map.json records the
    # long<->short mapping and is a required input for
    # m10_05_cross_stimulus_validation (which joins long-keyed run-id
    # JSONs against the short-keyed pickles).
    long_to_short = {
        long_id: normalize_cross_stim_run_id(long_id, "movie10")
        for long_id in decoded_states.keys()
    }
    if len(set(long_to_short.values())) != len(long_to_short):
        dupes = [v for v in long_to_short.values()
                 if list(long_to_short.values()).count(v) > 1]
        raise RuntimeError(
            f"Short run_ids have duplicates after normalization: {set(dupes)}"
        )
    decoded_states_short = {
        long_to_short[rid]: seq for rid, seq in decoded_states.items()
    }
    fo_short = compute_fractional_occupancy(decoded_states_short, n_states)
    run_id_map = {
        "short_to_long": {short: long for long, short in long_to_short.items()},
        "long_to_short": dict(long_to_short),
        "stimulus": "movie10",
    }

    with open(os.path.join(out_dir, 'decoded_states.pkl'), 'wb') as f:
        pickle.dump(decoded_states_short, f, protocol=4)

    with open(os.path.join(out_dir, 'fractional_occupancy.pkl'), 'wb') as f:
        pickle.dump(fo_short, f, protocol=4)

    with open(os.path.join(out_dir, 'run_id_map.json'), 'w') as f:
        json.dump(run_id_map, f, indent=2)

    with open(os.path.join(out_dir, 'movie_ll_summary.json'), 'w') as f:
        json.dump(ll_summary, f, indent=2)

    # =========================================================================
    # Report
    # =========================================================================

    print(f"\n{'='*60}")
    print(f"MOVIE10 SCORE & DECODE SUMMARY")
    print(f"{'='*60}")
    print(f"Subject:                {sub_id}")
    print(f"Parcellation:           {parc}")
    print(f"Movie runs decoded:     {total_runs_decoded}")
    print(f"Total movie TRs:        {total_trs}")
    print(f"Friends test LL/sample: {friends_test_ll:.4f}")
    print(f"Movie overall LL/sample:{overall_movie_ll:.4f}")
    print(f"Baseline LL/sample:     {baseline_ll:.4f}")
    print(f"LL gap (Friends-Movie): {friends_test_ll - overall_movie_ll:.4f}")
    print(f"Movie > baseline:       {overall_movie_ll > baseline_ll}")
    print(f"")
    print(f"Per-type LL/sample:")
    for mtype, info in per_type_ll.items():
        ll_val = info['ll_per_sample']
        ll_str = f"{ll_val:.4f}" if ll_val is not None else "N/A"
        print(f"  {mtype:10s}: {ll_str} ({info['n_runs']} runs, {info['n_trs']} TRs)")
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

        # Color per movie type
        TYPE_COLORS = {
            'bourne': NETWORK_COLORS['SomMot'],      # blue
            'wolf': NETWORK_COLORS['Default'],        # red
            'figures': NETWORK_COLORS['DorsAttn'],    # green
            'life': NETWORK_COLORS['Cont'],           # orange
        }

        # Build ordered run list grouped by type
        ordered_rids = []
        run_colors = []
        run_types = []
        for mtype in ['bourne', 'wolf', 'figures', 'life']:
            for rid in sorted(movie_run_ids.get(mtype, [])):
                if rid in per_run_ll:
                    ordered_rids.append(rid)
                    run_colors.append(TYPE_COLORS.get(mtype, '#888888'))
                    run_types.append(mtype)
        n_runs = len(ordered_rids)

        fig, (ax_ll, ax_st) = plt.subplots(1, 2, figsize=(12, 4.5))

        # Left: per-run LL dot chart, color by type
        ll_vals = [per_run_ll[rid] for rid in ordered_rids]
        for i, (ll, c) in enumerate(zip(ll_vals, run_colors)):
            ax_ll.scatter(i, ll, color=c, s=20, zorder=3, edgecolors='white', linewidths=0.3)
        ax_ll.axhline(friends_test_ll, color='black', ls='-', lw=1.2,
                      label=f'Friends test LL ({friends_test_ll:.2f})')
        ax_ll.axhline(overall_movie_ll, color='gray', ls='--', lw=1.0,
                      label=f'Movie mean LL ({overall_movie_ll:.2f})')
        # Type legend
        for mtype, color in TYPE_COLORS.items():
            ax_ll.scatter([], [], color=color, s=30, label=mtype)
        ax_ll.set_xlabel('Run index (grouped by type)')
        ax_ll.set_ylabel('LL / sample')
        ax_ll.set_title('Per-Run Log-Likelihood')
        ax_ll.legend(fontsize=6, loc='lower left', ncol=2)
        gap = friends_test_ll - overall_movie_ll
        ax_ll.text(0.97, 0.97, f'gap={gap:+.2f}', transform=ax_ll.transAxes,
                   ha='right', va='top', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Right: active states per run
        states_per_run = [len(np.unique(decoded_states[rid])) for rid in ordered_rids]
        ax_st.bar(range(n_runs), states_per_run, color=run_colors,
                  edgecolor='white', linewidth=0.3, alpha=0.8)
        ax_st.axhline(n_active_states, color='black', ls='--', lw=1.0,
                      label=f'Friends active ({n_active_states})')
        ax_st.set_xlabel('Run index (grouped by type)')
        ax_st.set_ylabel('# Active States')
        ax_st.set_title('States Used per Run')
        ax_st.legend(fontsize=7)

        vt_str = f', vt={args.vt}' if args.vt else ''
        fig.suptitle(f'Movie10 Score & Decode ({sub_id}{vt_str})', fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'll_diagnostic.png'), bbox_inches='tight')
        plt.close(fig)
        logger.info("Saved diagnostic figure: ll_diagnostic.png")
    except Exception as e:
        logger.warning(f"Could not generate diagnostic figure: {e}")


if __name__ == '__main__':
    main()
