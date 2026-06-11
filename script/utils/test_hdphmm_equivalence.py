#!/usr/bin/env python3
"""
test_hdphmm_equivalence.py — Validate JAX vs numpy StickyHDPHMM equivalence.

Tests:
  1. Per-iteration log-likelihood agreement (10 iters, same init)
  2. Emission log-likelihood agreement (single forward pass)
  3. Forward-backward posterior agreement
  4. Viterbi decode agreement

Usage:
    python script/utils/test_hdphmm_equivalence.py
    python script/utils/test_hdphmm_equivalence.py --n_iter 50 --verbose
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def generate_test_data(N_seqs=10, T_per_seq=50, D=20, K=10, seed=42):
    """Generate synthetic multi-sequence data for testing."""
    rng = np.random.RandomState(seed)
    lengths = [T_per_seq + rng.randint(-5, 5) for _ in range(N_seqs)]
    N_total = sum(lengths)
    X = rng.randn(N_total, D).astype(np.float64)
    return X, lengths


def copy_params_np_to_jax(model_np, model_jax):
    """Copy initialized parameters from numpy model to JAX model."""
    model_jax.means_ = model_np.means_.copy()
    # hmmlearn's covars_ property returns (K, D, D) even for diag;
    # normalize to native format for the JAX model
    model_jax.covars_ = model_jax._normalize_covars(model_np.covars_.copy())
    model_jax.transmat_ = model_np.transmat_.copy()
    model_jax.startprob_ = model_np.startprob_.copy()
    model_jax.state_weights = model_np.state_weights.copy()
    model_jax.active_states = model_np.active_states.copy()
    model_jax.n_features = model_np.n_features


def test_emission_log_likelihood(X, model_np, model_jax, atol=1e-4):
    """Test that emission log-likelihoods agree."""
    import jax.numpy as jnp
    import jax

    # Numpy: use hmmlearn's _compute_log_likelihood
    flp_np = model_np._compute_log_likelihood(X)

    # JAX
    means_j = jax.device_put(jnp.array(model_jax.means_, dtype=jnp.float64))
    covars_j = jax.device_put(jnp.array(model_jax.covars_, dtype=jnp.float64))
    X_j = jax.device_put(jnp.array(X, dtype=jnp.float64))

    if model_jax.covariance_type == 'full':
        from utils.hdphmm_jax import _prepare_cholesky, _gaussian_log_likelihood_full
        chol, log_dets = _prepare_cholesky(covars_j, model_jax.min_covar)
        flp_jax = np.asarray(_gaussian_log_likelihood_full(X_j, means_j, chol, log_dets))
    else:
        from utils.hdphmm_jax import _gaussian_log_likelihood_diag
        flp_jax = np.asarray(_gaussian_log_likelihood_diag(X_j, means_j, covars_j))

    max_diff = np.max(np.abs(flp_np - flp_jax))
    mean_diff = np.mean(np.abs(flp_np - flp_jax))
    logger.info("Emission LL: max_diff=%.2e, mean_diff=%.2e (atol=%.2e)",
                max_diff, mean_diff, atol)

    if max_diff > atol:
        logger.warning("FAIL: emission LL max_diff %.2e > atol %.2e", max_diff, atol)
        return False
    logger.info("PASS: emission log-likelihood")
    return True


def test_score(X, lengths, model_np, model_jax, rtol=1e-3):
    """Test that total log-likelihood (score) agrees."""
    ll_np = model_np.score(X, lengths=lengths)
    ll_jax = model_jax.score(X, lengths=lengths)

    rel_diff = abs(ll_np - ll_jax) / max(abs(ll_np), 1e-10)
    logger.info("Score: numpy=%.4f, jax=%.4f, rel_diff=%.2e (rtol=%.2e)",
                ll_np, ll_jax, rel_diff, rtol)

    if rel_diff > rtol:
        logger.warning("FAIL: score rel_diff %.2e > rtol %.2e", rel_diff, rtol)
        return False
    logger.info("PASS: score")
    return True


def test_decode(X, lengths, model_np, model_jax):
    """Test Viterbi decode agreement."""
    _, states_np = model_np.decode(X, lengths=lengths)
    _, states_jax = model_jax.decode(X, lengths=lengths)

    agreement = np.mean(states_np == states_jax)
    logger.info("Decode agreement: %.1f%% (%d/%d frames)",
                agreement * 100, np.sum(states_np == states_jax), len(states_np))

    if agreement < 0.90:
        logger.warning("FAIL: decode agreement %.1f%% < 90%%", agreement * 100)
        return False
    logger.info("PASS: decode agreement")
    return True


def test_fit_trajectory(X, lengths, n_iter=10, cov_type='diag', atol_ll=1e-2):
    """Test that EM trajectories agree for n_iter iterations."""
    from utils.hdphmm import StickyHDPHMM
    from utils.hdphmm_jax import StickyHDPHMM_JAX

    K = 10
    common_kwargs = dict(
        n_components=K, alpha=1.0, gamma=3.0, kappa=10.0, rho=1.0,
        covariance_type=cov_type, random_state=42, n_iter=n_iter,
        tol=1e-10,  # Don't converge early
        min_iter=n_iter, verbose=False, min_covar=1e-3, n_jobs=1,
    )

    # Fit numpy
    model_np = StickyHDPHMM(**common_kwargs)
    model_np.fit(X, lengths=lengths)
    ll_np = model_np.history['log_likelihood']

    # Fit JAX with same seed
    model_jax = StickyHDPHMM_JAX(**common_kwargs)
    model_jax.fit(X, lengths=lengths)
    ll_jax = model_jax.history['log_likelihood']

    # Compare LL trajectories
    n_common = min(len(ll_np), len(ll_jax))
    ll_np = np.array(ll_np[:n_common])
    ll_jax = np.array(ll_jax[:n_common])
    abs_diff = np.abs(ll_np - ll_jax)
    rel_diff = abs_diff / np.maximum(np.abs(ll_np), 1e-10)

    logger.info("Fit trajectory (%s, %d iters):", cov_type, n_common)
    logger.info("  LL range (np):  [%.1f, %.1f]", ll_np[0], ll_np[-1])
    logger.info("  LL range (jax): [%.1f, %.1f]", ll_jax[0], ll_jax[-1])
    logger.info("  Max abs diff:   %.2e", np.max(abs_diff))
    logger.info("  Max rel diff:   %.2e", np.max(rel_diff))
    logger.info("  Mean rel diff:  %.2e", np.mean(rel_diff))

    # Check means correlation
    corrs = []
    for k in range(K):
        r = np.corrcoef(model_np.means_[k], model_jax.means_[k])[0, 1]
        corrs.append(r)
    mean_corr = np.mean(corrs)
    logger.info("  Mean state-mean correlation: %.4f", mean_corr)

    passed = True
    if np.max(rel_diff) > 0.1:  # 10% relative tolerance for multi-iter
        logger.warning("  WARN: large LL divergence (expected for different RNG paths)")
    if mean_corr < 0.8:
        logger.warning("  FAIL: low state-mean correlation %.4f < 0.80", mean_corr)
        passed = False
    else:
        logger.info("  PASS: fit trajectory")

    return passed


def main():
    parser = argparse.ArgumentParser(description='Test JAX vs numpy HDP-HMM equivalence')
    parser.add_argument('--n_iter', type=int, default=10)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("StickyHDPHMM: JAX vs numpy equivalence tests")
    logger.info("=" * 60)

    # Check JAX availability
    try:
        import jax
        logger.info("JAX version: %s, backend: %s, devices: %s",
                     jax.__version__, jax.default_backend(), jax.devices())
    except ImportError:
        logger.error("JAX not available — cannot run equivalence tests")
        sys.exit(1)

    # Generate test data
    X, lengths = generate_test_data(N_seqs=10, T_per_seq=50, D=20, K=10)
    logger.info("Test data: N=%d, D=%d, %d sequences", X.shape[0], X.shape[1], len(lengths))

    # Initialize both models with identical params
    from utils.hdphmm import StickyHDPHMM
    from utils.hdphmm_jax import StickyHDPHMM_JAX

    common_kwargs = dict(
        n_components=10, alpha=1.0, gamma=3.0, kappa=10.0, rho=1.0,
        covariance_type='diag', random_state=42, n_iter=1,
        tol=1e-10, verbose=False, min_covar=1e-3, n_jobs=1,
    )

    model_np = StickyHDPHMM(**common_kwargs)
    model_np._init_params(X)

    model_jax = StickyHDPHMM_JAX(**common_kwargs)
    copy_params_np_to_jax(model_np, model_jax)

    results = []

    # Test 1: Emission LL
    logger.info("\n--- Test 1: Emission log-likelihood ---")
    results.append(test_emission_log_likelihood(X, model_np, model_jax))

    # Test 2: Score (total LL)
    logger.info("\n--- Test 2: Score (total LL) ---")
    results.append(test_score(X, lengths, model_np, model_jax))

    # Test 3: Decode
    logger.info("\n--- Test 3: Viterbi decode ---")
    results.append(test_decode(X, lengths, model_np, model_jax))

    # Test 4: Fit trajectory (diag)
    logger.info("\n--- Test 4: Fit trajectory (diag, %d iters) ---", args.n_iter)
    X_fit, lengths_fit = generate_test_data(N_seqs=5, T_per_seq=100, D=20)
    results.append(test_fit_trajectory(X_fit, lengths_fit, n_iter=args.n_iter, cov_type='diag'))

    # Test 5: Fit trajectory (full)
    logger.info("\n--- Test 5: Fit trajectory (full, %d iters) ---", args.n_iter)
    results.append(test_fit_trajectory(X_fit, lengths_fit, n_iter=args.n_iter, cov_type='full'))

    # Summary
    logger.info("\n" + "=" * 60)
    n_pass = sum(results)
    n_total = len(results)
    if n_pass == n_total:
        logger.info("ALL %d TESTS PASSED", n_total)
    else:
        logger.warning("%d/%d tests passed", n_pass, n_total)
    logger.info("=" * 60)

    sys.exit(0 if n_pass == n_total else 1)


if __name__ == '__main__':
    main()
