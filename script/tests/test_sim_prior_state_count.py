#!/usr/bin/env python3
"""Tests for sm_sim_prior_state_count.py - the prior-predictive occupied-state count.

Locks in the analytic backbone of the simulation:
  * stick_breaking returns a normalized base measure.
  * sample_prior returns valid (row-stochastic) transition matrices.
  * The prior-mean transition matrix is exactly sticky: its stationary
    distribution equals the base measure beta (so long-run occupancy -> beta).
  * At the production setting (gamma=1) the prior occupies only a few states,
    well below the truncation capacity.

Run standalone::

    python script/tests/test_sim_prior_state_count.py
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import sm_sim_prior_state_count as m  # noqa: E402


def test_stick_breaking_normalized():
    rng = np.random.RandomState(0)
    for gamma in (0.5, 1.0, 5.0):
        beta = m.stick_breaking(gamma, 50, rng)
        assert beta.shape == (50,)
        assert np.all(beta >= 0)
        assert abs(beta.sum() - 1.0) < 1e-12


def test_sample_prior_rows_stochastic():
    rng = np.random.RandomState(1)
    beta, startprob, trans = m.sample_prior(1.0, 10.0, 1.0, 1.0, 50, rng)
    assert abs(startprob.sum() - 1.0) < 1e-12
    row_sums = trans.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-12)
    assert np.all(trans >= 0)


def test_mean_matrix_stationary_equals_beta():
    """Prior-mean sticky matrix has stationary distribution exactly beta."""
    rng = np.random.RandomState(2)
    beta = m.stick_breaking(1.0, 50, rng)
    trans = m.mean_transition(beta, kappa=10.0, alpha=1.0, rho=1.0)
    pi = m.stationary(trans)
    assert np.allclose(pi, beta, atol=1e-8)


def test_production_prior_occupies_few_states():
    """gamma=1 prior should favor only a handful of states (far below K_max=50)."""
    rng = np.random.RandomState(0)
    counts = []
    for _ in range(500):
        beta, _s, trans = m.sample_prior(1.0, 10.0, 1.0, 1.0, 50, rng)
        pi = m.stationary(trans)
        counts.append(int((pi > 0.01).sum()))
    counts = np.array(counts)
    median = np.median(counts)
    # Median should be a small single-digit number; certainly not "a dozen",
    # and nowhere near the truncation capacity.
    assert median <= 4, f"expected small prior repertoire, got median {median}"
    assert counts.max() <= 20
    assert counts.mean() < 5


if __name__ == "__main__":
    test_stick_breaking_normalized()
    test_sample_prior_rows_stochastic()
    test_mean_matrix_stationary_equals_beta()
    test_production_prior_occupies_few_states()
    print("all tests passed")
