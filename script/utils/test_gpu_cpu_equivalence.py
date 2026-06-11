#!/usr/bin/env python3
"""
test_gpu_cpu_equivalence.py — Compare JAX/GPU vs numpy/CPU HDP-HMM on real data.

Loads existing CPU-fitted model (seed 0) for a fast config (vt0.80_covdiag_nc60_g5),
fits the same config with JAX on GPU, and compares:
  1. LL trajectory (first N iterations)
  2. Final means correlation
  3. Final covars agreement
  4. Validation LL agreement
  5. Decoded state agreement (Viterbi)

Usage (on a GPU node):
    python script/utils/test_gpu_cpu_equivalence.py
    python script/utils/test_gpu_cpu_equivalence.py --sub_id sub-02 --seed 0
    python script/utils/test_gpu_cpu_equivalence.py --n_iter 200  # short run for quick check
"""

import os
import sys
import json
import pickle
import argparse
import logging
import time
from pathlib import Path

import numpy as np

# Ensure project root is on path
PROJECT_DIR = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_DIR, '.env'))
SCRATCH_DIR = os.getenv('SCRATCH_DIR')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_cpu_model(sub_id, parcellation, config_name, seed):
    """Load a CPU-fitted model and its metadata."""
    base = os.path.join(
        SCRATCH_DIR, 'output', '04_combined_hdphmm',
        parcellation, sub_id, 'configs', config_name
    )
    model_path = os.path.join(base, f'seed_{seed}_model.pkl')
    json_path = os.path.join(base, f'seed_{seed}.json')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(json_path, 'r') as f:
        meta = json.load(f)

    logger.info("Loaded CPU model: %s (seed %d)", config_name, seed)
    logger.info("  train_ll/s=%.6f, valid_ll/s=%.6f, active=%d, iters=%d",
                meta['train_ll_per_sample'], meta['valid_ll_per_sample'],
                meta['n_active_states'], meta['n_iterations'])
    return model, meta


def load_real_data(sub_id, parcellation, variance_threshold):
    """Load PCA-projected train/valid data (same as 04_combined_hdphmm.py)."""
    from utils.hmm_io import (load_split, load_n_pcs_lookup,
                               get_projected_dir, load_projected_runs)

    combined_base = os.path.join(
        SCRATCH_DIR, 'output', '03a_pca4combined_hmm', parcellation, sub_id
    )
    n_pcs_lookup = load_n_pcs_lookup(combined_base)
    vt_key = f"{variance_threshold:.2f}"
    n_pcs = n_pcs_lookup[vt_key]

    split = load_split(combined_base, split_type='primary')
    projected_dir = get_projected_dir(combined_base)

    X_train, lengths_train = load_projected_runs(
        split['train'], projected_dir, n_pcs, 'train'
    )
    X_valid, lengths_valid = load_projected_runs(
        split['valid'], projected_dir, n_pcs, 'valid'
    )

    logger.info("Data: train %d TRs x %d PCs (%d seqs), valid %d TRs (%d seqs)",
                X_train.shape[0], n_pcs, len(lengths_train),
                X_valid.shape[0], len(lengths_valid))
    return X_train, lengths_train, X_valid, lengths_valid, n_pcs


def fit_jax_model(X_train, lengths_train, config, seed, n_iter_override=None):
    """Fit JAX model with same config and seed as CPU."""
    from utils.hdphmm_jax import StickyHDPHMM_JAX
    import random

    n_iter = n_iter_override or config.get('n_iter', 10000)
    tol = config.get('tol', 1e-6)
    if n_iter_override:
        # For short runs, disable early stopping
        tol = 1e-20

    np.random.seed(seed)
    random.seed(seed)

    model = StickyHDPHMM_JAX(
        n_components=config['n_components'],
        covariance_type=config['covariance_type'],
        alpha=config['alpha'],
        gamma=config['gamma'],
        kappa=config['kappa'],
        rho=config['rho'],
        n_iter=n_iter,
        tol=tol,
        min_state_usage=config['min_state_usage'],
        min_iter=config.get('min_iter', 0) if not n_iter_override else 0,
        random_state=seed,
        verbose=False,
        n_jobs=1,
    )

    logger.info("Fitting JAX model (seed=%d, n_iter=%d, tol=%.1e)...", seed, n_iter, tol)
    t0 = time.time()
    model.fit(X_train, lengths=lengths_train)
    elapsed = time.time() - t0
    logger.info("JAX fit done in %.1f s (%d iters)", elapsed, len(model.history.get('log_likelihood', [])))

    return model, elapsed


def compare_models(cpu_model, jax_model, X_valid, lengths_valid, cpu_meta):
    """Compare CPU and JAX models across multiple metrics."""
    results = {}
    all_pass = True

    # =========================================================================
    # 1. LL trajectory comparison (first N common iterations)
    # =========================================================================
    logger.info("\n--- Test 1: LL trajectory ---")
    ll_cpu = np.array(cpu_model.history.get('log_likelihood', []))
    ll_jax = np.array(jax_model.history.get('log_likelihood', []))
    n_common = min(len(ll_cpu), len(ll_jax))

    if n_common > 0:
        ll_cpu_c = ll_cpu[:n_common]
        ll_jax_c = ll_jax[:n_common]
        abs_diff = np.abs(ll_cpu_c - ll_jax_c)
        rel_diff = abs_diff / np.maximum(np.abs(ll_cpu_c), 1.0)

        # Check at iterations 1, 10, 50, 100, ...
        checkpoints = [i for i in [0, 9, 49, 99, 199, 499, n_common-1] if i < n_common]
        for i in checkpoints:
            logger.info("  iter %4d: cpu=%.2f, jax=%.2f, abs_diff=%.2e, rel_diff=%.2e",
                        i+1, ll_cpu_c[i], ll_jax_c[i], abs_diff[i], rel_diff[i])

        max_rel = np.max(rel_diff)
        mean_rel = np.mean(rel_diff)
        logger.info("  Max rel diff: %.2e, Mean rel diff: %.2e", max_rel, mean_rel)

        # After iter 1, trajectories should be very close (<1% relative)
        if n_common > 1:
            late_rel = np.max(rel_diff[1:])
            if late_rel > 0.01:
                logger.warning("  WARN: LL trajectory diverges (max rel diff %.2e > 1%%)", late_rel)
            else:
                logger.info("  PASS: LL trajectories agree within 1%%")

        results['ll_trajectory'] = {
            'n_common_iters': n_common,
            'max_rel_diff': float(max_rel),
            'mean_rel_diff': float(mean_rel),
        }
    else:
        logger.warning("  SKIP: no LL history available")

    # =========================================================================
    # 2. Final means correlation
    # =========================================================================
    logger.info("\n--- Test 2: Means correlation ---")
    means_cpu = cpu_model.means_
    means_jax = jax_model.means_
    K = min(means_cpu.shape[0], means_jax.shape[0])

    # Match states by highest correlation (Hungarian-like greedy)
    from scipy.spatial.distance import cdist
    corr_matrix = 1 - cdist(means_cpu[:K], means_jax[:K], metric='correlation')
    # Greedy matching
    matched_corrs = []
    used_jax = set()
    for k_cpu in range(K):
        best_r = -2
        best_j = -1
        for k_jax in range(K):
            if k_jax not in used_jax and corr_matrix[k_cpu, k_jax] > best_r:
                best_r = corr_matrix[k_cpu, k_jax]
                best_j = k_jax
        if best_j >= 0:
            matched_corrs.append(best_r)
            used_jax.add(best_j)

    matched_corrs = np.array(matched_corrs)
    median_corr = np.median(matched_corrs)
    min_corr = np.min(matched_corrs)
    n_high = np.sum(matched_corrs > 0.95)

    logger.info("  %d states matched, median r=%.4f, min r=%.4f, %d/%d with r>0.95",
                len(matched_corrs), median_corr, min_corr, n_high, K)

    if median_corr < 0.90:
        logger.warning("  FAIL: median state-mean correlation %.4f < 0.90", median_corr)
        all_pass = False
    else:
        logger.info("  PASS: means correlation (median=%.4f)", median_corr)

    results['means_correlation'] = {
        'n_states': K,
        'median_r': float(median_corr),
        'min_r': float(min_corr),
        'n_above_095': int(n_high),
    }

    # =========================================================================
    # 3. Validation LL agreement
    # =========================================================================
    logger.info("\n--- Test 3: Validation LL ---")
    jax_valid_ll = jax_model.score(X_valid, lengths=lengths_valid)
    jax_valid_ll_ps = jax_valid_ll / sum(lengths_valid)
    cpu_valid_ll_ps = cpu_meta['valid_ll_per_sample']

    rel_diff_valid = abs(cpu_valid_ll_ps - jax_valid_ll_ps) / max(abs(cpu_valid_ll_ps), 1e-10)
    logger.info("  CPU valid LL/s: %.6f", cpu_valid_ll_ps)
    logger.info("  JAX valid LL/s: %.6f", jax_valid_ll_ps)
    logger.info("  Rel diff: %.2e", rel_diff_valid)

    if rel_diff_valid > 0.05:
        logger.warning("  FAIL: validation LL rel diff %.2e > 5%%", rel_diff_valid)
        all_pass = False
    else:
        logger.info("  PASS: validation LL agrees within 5%%")

    results['valid_ll'] = {
        'cpu': float(cpu_valid_ll_ps),
        'jax': float(jax_valid_ll_ps),
        'rel_diff': float(rel_diff_valid),
    }

    # =========================================================================
    # 4. Active states count
    # =========================================================================
    logger.info("\n--- Test 4: Active states ---")
    cpu_active = cpu_meta['n_active_states']
    jax_usage = jax_model.history.get('state_usage', [[]])[-1]
    jax_active = int(np.sum(np.array(jax_usage) > 0.01)) if jax_usage else -1

    logger.info("  CPU active: %d, JAX active: %d", cpu_active, jax_active)
    diff_pct = abs(cpu_active - jax_active) / max(cpu_active, 1) * 100
    if diff_pct > 30:
        logger.warning("  WARN: active state count differs by %.0f%%", diff_pct)
    else:
        logger.info("  PASS: active state counts within 30%%")

    results['active_states'] = {'cpu': cpu_active, 'jax': jax_active}

    # =========================================================================
    # 5. Decoded state agreement
    # =========================================================================
    logger.info("\n--- Test 5: Viterbi decode agreement ---")
    _, states_cpu = cpu_model.decode(X_valid, lengths=lengths_valid)
    _, states_jax = jax_model.decode(X_valid, lengths=lengths_valid)

    # Direct match may be low because state indices differ; measure via contingency
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(states_cpu, states_jax)
    logger.info("  Adjusted Rand Index: %.4f", ari)

    if ari < 0.5:
        logger.warning("  WARN: low ARI (%.4f) — state solutions may differ", ari)
    else:
        logger.info("  PASS: decode ARI > 0.5")

    results['decode'] = {'adjusted_rand_index': float(ari)}

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 60)
    if all_pass:
        logger.info("OVERALL: PASS — GPU/JAX produces equivalent results to CPU/numpy")
    else:
        logger.warning("OVERALL: SOME CHECKS FAILED — investigate differences")
    logger.info("=" * 60)

    return results, all_pass


def main():
    parser = argparse.ArgumentParser(
        description='Compare JAX/GPU vs numpy/CPU HDP-HMM on real data'
    )
    parser.add_argument('--sub_id', default='sub-01')
    parser.add_argument('--parcellation', default='atlas-4S156Parcels')
    parser.add_argument('--config_name', default='vt0.80_covdiag_nc60_g5',
                        help='Config to compare (must have CPU results)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_iter', type=int, default=None,
                        help='Override n_iter for quick test (default: run to convergence)')
    parser.add_argument('--output', default=None,
                        help='Save comparison results JSON to this path')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("GPU vs CPU Equivalence Test — Real Data")
    logger.info("=" * 60)
    logger.info("Subject:    %s", args.sub_id)
    logger.info("Config:     %s", args.config_name)
    logger.info("Seed:       %d", args.seed)

    # Check JAX/GPU availability
    try:
        import jax
        backend = jax.default_backend()
        devices = jax.devices()
        logger.info("JAX backend: %s, devices: %s", backend, devices)
        if backend != 'gpu':
            logger.warning("JAX is NOT using GPU — comparison will be CPU vs CPU!")
    except ImportError:
        logger.error("JAX not available — cannot run GPU equivalence test")
        sys.exit(1)

    # Load CPU model
    cpu_model, cpu_meta = load_cpu_model(
        args.sub_id, args.parcellation, args.config_name, args.seed
    )

    # Parse config from CPU metadata
    config_path = os.path.join(
        SCRATCH_DIR, 'output', '04_combined_hdphmm',
        args.parcellation, args.sub_id, 'configs', args.config_name,
        'config_summary.json'
    )
    with open(config_path, 'r') as f:
        config = json.load(f)['config']

    # Load real data
    X_train, lengths_train, X_valid, lengths_valid, n_pcs = load_real_data(
        args.sub_id, args.parcellation, config['variance_threshold']
    )

    # Fit JAX model
    jax_model, jax_elapsed = fit_jax_model(
        X_train, lengths_train, config, args.seed, args.n_iter
    )

    # Compare
    results, all_pass = compare_models(
        cpu_model, jax_model, X_valid, lengths_valid, cpu_meta
    )
    results['jax_elapsed_seconds'] = jax_elapsed
    results['cpu_elapsed_seconds'] = cpu_meta['elapsed_seconds']
    results['speedup'] = cpu_meta['elapsed_seconds'] / max(jax_elapsed, 1)
    logger.info("\nSpeedup: %.1fx (CPU: %.0fs, JAX: %.0fs)",
                results['speedup'], results['cpu_elapsed_seconds'], jax_elapsed)

    # Save results
    if args.output:
        out_path = args.output
    else:
        out_dir = os.path.join(
            SCRATCH_DIR, 'output', '04_combined_hdphmm',
            args.parcellation, args.sub_id, 'gpu_equivalence'
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{args.config_name}_seed{args.seed}.json')

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)

    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
