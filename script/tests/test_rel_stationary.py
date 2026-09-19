"""Tests for utils.stationary — the stationary distribution used by the
assortativity and recurrence-robustness supplements to quantify how closely
recurrence tracks the model's long-run occupancy."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from utils.stationary import stationary_distribution


def test_two_state_chain_matches_closed_form():
    # P = [[1-a, a],[b, 1-b]] has pi = (b, a)/(a+b)
    a, b = 0.2, 0.6
    P = np.array([[1 - a, a], [b, 1 - b]])
    pi = stationary_distribution(P, [0, 1])
    np.testing.assert_allclose(pi, np.array([b, a]) / (a + b), atol=1e-12)


def test_subset_is_renormalized_and_ordered_ascending():
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(5), size=5)          # 5-state chain
    pi_sub = stationary_distribution(P, [4, 1, 3])   # unsorted subset
    assert pi_sub.shape == (3,) and pi_sub.sum() == pytest.approx(1.0)
    Psub = P[np.ix_([1, 3, 4], [1, 3, 4])]
    Psub = Psub / Psub.sum(1, keepdims=True)
    np.testing.assert_allclose(pi_sub @ Psub, pi_sub, atol=1e-10)


def test_result_is_nonnegative_and_floored():
    P = np.array([[1.0, 0.0], [0.5, 0.5]])           # state 0 absorbing
    pi = stationary_distribution(P, [0, 1])
    assert pi.min() >= 1e-15 and pi[0] == pytest.approx(1.0, abs=1e-12)
