"""Unit tests for the R5 phase-randomized null (sm_rel_r5_phase_null.py).

The load-bearing claims are (a) the surrogate preserves exactly what the
Methods says it preserves, (b) the standalone Viterbi is a correct decoder, and
(c) the seed rule makes any draw independently reproducible. Each is tested on
synthetic data; nothing here touches the pipeline outputs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from utils.ica_oos_recurrence import phase_randomize
from sm_rel_r5_phase_null import (
    SEED_BASE,
    hmm_params,
    mean_fractional_occupancy,
    viterbi,
)


# --------------------------------------------------------------------------
# Surrogate properties: what the null preserves and destroys
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_t", [128, 129])  # even and odd length
def test_phase_randomize_preserves_power_spectrum(n_t):
    """Each component's power spectrum survives the shared-phase rotation."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_t, 5)).cumsum(0)  # autocorrelated
    Y = phase_randomize(X, np.random.default_rng(7))

    px = np.abs(np.fft.rfft(X, axis=0)) ** 2
    py = np.abs(np.fft.rfft(Y, axis=0)) ** 2
    np.testing.assert_allclose(px, py, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("n_t", [128, 129])
def test_phase_randomize_preserves_cross_component_covariance(n_t):
    """A phase rotation shared across components leaves the covariance intact.

    This is the property that separates the multivariate Prichard-Theiler
    surrogate from independent per-component phase randomization, and it is
    what licenses the manuscript's claim that the null preserves covariance
    structure. An independent-phase surrogate fails this test.
    """
    rng = np.random.default_rng(1)
    mixing = rng.normal(size=(4, 4))
    X = (rng.normal(size=(n_t, 4)).cumsum(0)) @ mixing
    Y = phase_randomize(X, np.random.default_rng(11))

    cx = np.cov(X, rowvar=False)
    cy = np.cov(Y, rowvar=False)
    np.testing.assert_allclose(cx, cy, rtol=1e-8, atol=1e-8)


def test_independent_phase_surrogate_would_destroy_covariance():
    """Guard the contrast: per-component phases break the cross-covariance.

    If someone 'simplifies' phase_randomize to draw one phase vector per
    component, this test is what fails loudly rather than the Methods quietly
    becoming false.
    """
    rng = np.random.default_rng(2)
    mixing = rng.normal(size=(4, 4))
    X = (rng.normal(size=(256, 4)).cumsum(0)) @ mixing

    F = np.fft.rfft(X, axis=0)
    ph = np.random.default_rng(3).uniform(0, 2 * np.pi, size=F.shape)
    ph[0, :] = 0.0
    ph[-1, :] = 0.0
    Y = np.fft.irfft(F * np.exp(1j * ph), n=X.shape[0], axis=0)

    cx = np.cov(X, rowvar=False)
    cy = np.cov(Y, rowvar=False)
    off_x = cx[np.triu_indices(4, k=1)]
    off_y = cy[np.triu_indices(4, k=1)]
    assert not np.allclose(off_x, off_y, rtol=0.1, atol=0.1)


def test_phase_randomize_output_is_real_and_shaped():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(64, 3))
    Y = phase_randomize(X, np.random.default_rng(5))
    assert Y.shape == X.shape
    assert np.isrealobj(Y)


def test_phase_randomize_is_seed_deterministic():
    """The seed rule must make any single draw reproducible on its own."""
    X = np.random.default_rng(6).normal(size=(100, 4))
    a = phase_randomize(X, np.random.default_rng(SEED_BASE + 42))
    b = phase_randomize(X, np.random.default_rng(SEED_BASE + 42))
    c = phase_randomize(X, np.random.default_rng(SEED_BASE + 43))
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


# --------------------------------------------------------------------------
# Viterbi decoder
# --------------------------------------------------------------------------

def _brute_force_best_path(log_emit, log_start, log_trans):
    """Exhaustive search over all state paths; only tractable for tiny inputs."""
    from itertools import product
    n_t, n_k = log_emit.shape
    best, best_path = -np.inf, None
    for path in product(range(n_k), repeat=n_t):
        score = log_start[path[0]] + log_emit[0, path[0]]
        for t in range(1, n_t):
            score += log_trans[path[t - 1], path[t]] + log_emit[t, path[t]]
        if score > best:
            best, best_path = score, np.array(path)
    return best_path


def test_viterbi_matches_brute_force():
    rng = np.random.default_rng(0)
    n_k, n_d, n_t = 3, 2, 7
    means = rng.normal(size=(n_k, n_d)) * 3
    var = rng.uniform(0.5, 2.0, size=(n_k, n_d))
    log_start = np.log(np.full(n_k, 1 / n_k))
    trans = rng.uniform(size=(n_k, n_k)) + np.eye(n_k) * 4
    log_trans = np.log(trans / trans.sum(1, keepdims=True))
    X = rng.normal(size=(n_t, n_d)) * 2

    got = viterbi(X, means, var, log_start, log_trans)

    inv = 1.0 / var
    const = -0.5 * np.log(2 * np.pi * var).sum(1)
    log_emit = np.stack([
        [-0.5 * (((X[t] - means[k]) ** 2 * inv[k]).sum()) + const[k]
         for k in range(n_k)] for t in range(n_t)])
    expected = _brute_force_best_path(log_emit, log_start, log_trans)
    np.testing.assert_array_equal(got, expected)


def test_viterbi_recovers_well_separated_states():
    """With far-apart means and sticky transitions, the path tracks the source."""
    n_d = 3
    means = np.array([np.full(n_d, -20.0), np.full(n_d, 20.0)])
    var = np.ones((2, n_d))
    log_start = np.log([0.5, 0.5])
    log_trans = np.log([[0.95, 0.05], [0.05, 0.95]])
    truth = np.array([0] * 15 + [1] * 15)
    rng = np.random.default_rng(9)
    X = means[truth] + rng.normal(scale=0.5, size=(30, n_d))
    np.testing.assert_array_equal(viterbi(X, means, var, log_start, log_trans), truth)


# --------------------------------------------------------------------------
# Parameter extraction and occupancy
# --------------------------------------------------------------------------

class _StubModel:
    def __init__(self, means, covars, transmat, startprob):
        self.means_, self.covars_ = means, covars
        self.transmat_, self.startprob_ = transmat, startprob
        self.n_components = len(startprob)


def test_hmm_params_accepts_diagonal_and_full_covars():
    """covars_ is stored as (K, D) or (K, D, D); both must yield the same variances."""
    n_k, n_d = 3, 4
    rng = np.random.default_rng(0)
    means = rng.normal(size=(n_k, n_d))
    diag = rng.uniform(0.5, 2.0, size=(n_k, n_d))
    full = np.stack([np.diag(d) for d in diag])
    trans = np.full((n_k, n_k), 1 / n_k)
    start = np.full(n_k, 1 / n_k)

    _, var_d, _, _ = hmm_params(_StubModel(means, diag, trans, start), n_d)
    _, var_f, _, _ = hmm_params(_StubModel(means, full, trans, start), n_d)
    np.testing.assert_allclose(var_d, var_f)
    np.testing.assert_allclose(var_d, diag)


def test_hmm_params_truncates_to_n_pcs_and_floors_variance():
    n_k, n_d, n_pcs = 2, 5, 3
    means = np.arange(n_k * n_d, dtype=float).reshape(n_k, n_d)
    covars = np.zeros((n_k, n_d))  # degenerate: must be floored, not zero
    trans = np.full((n_k, n_k), 0.5)
    start = np.full(n_k, 0.5)

    m, var, log_start, log_trans = hmm_params(
        _StubModel(means, covars, trans, start), n_pcs)
    assert m.shape == (n_k, n_pcs) and var.shape == (n_k, n_pcs)
    assert np.all(var > 0)
    assert np.all(np.isfinite(log_start)) and np.all(np.isfinite(log_trans))


def test_hmm_params_zero_transition_probability_stays_finite():
    """log(0) would poison the recursion; the floor must keep it finite."""
    trans = np.array([[1.0, 0.0], [0.0, 1.0]])
    _, _, _, log_trans = hmm_params(
        _StubModel(np.zeros((2, 2)), np.ones((2, 2)), trans, np.array([1.0, 0.0])), 2)
    assert np.all(np.isfinite(log_trans))
    assert log_trans[0, 1] < -600


def test_mean_fractional_occupancy_weights_runs_equally():
    """Runs of unequal length contribute equally, matching the R5 definition."""
    long_run = np.zeros(100, dtype=int)      # 100% state 0
    short_run = np.ones(4, dtype=int)        # 100% state 1
    occ = mean_fractional_occupancy([long_run, short_run], 2)
    np.testing.assert_allclose(occ, [0.5, 0.5])


def test_mean_fractional_occupancy_sums_to_one():
    rng = np.random.default_rng(3)
    paths = [rng.integers(0, 5, size=n) for n in (40, 55, 61)]
    occ = mean_fractional_occupancy(paths, 5)
    assert occ.shape == (5,)
    np.testing.assert_allclose(occ.sum(), 1.0)


def test_mean_fractional_occupancy_includes_unvisited_states():
    """States absent from every run must appear as zeros, not be dropped."""
    occ = mean_fractional_occupancy([np.zeros(10, dtype=int)], 4)
    assert occ.shape == (4,)
    np.testing.assert_allclose(occ, [1.0, 0.0, 0.0, 0.0])
