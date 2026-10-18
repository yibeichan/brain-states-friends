"""Comparing two HMM fits' states: matching, label agreement, occupancy, verdict."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.state_compare import (  # noqa: E402
    compare_fits, fo_distance, label_agreement, match_states, moved_verdict,
    pooled_fo,
)


def test_match_recovers_permutation():
    rng = np.random.default_rng(0)
    means_a = rng.normal(size=(4, 30))
    perm = np.array([2, 0, 3, 1])           # b[perm[i]] == a[i]
    means_b = np.empty_like(means_a)
    means_b[perm] = means_a
    m = match_states(means_a, means_b, np.arange(4), np.arange(4))
    assert m["mapping"] == {0: 2, 1: 0, 2: 3, 3: 1}
    np.testing.assert_allclose(m["correlations"], 1.0)


def test_match_only_uses_active_states():
    rng = np.random.default_rng(1)
    means = rng.normal(size=(3, 20))
    m = match_states(means, means, np.array([0, 2]), np.array([0, 1, 2]))
    assert m["mapping"] == {0: 0, 2: 2}
    assert m["n_matched"] == 2


def test_label_agreement_permutation_invariant():
    a = {"r1": np.array([0, 0, 1, 1, 2, 2]), "r2": np.array([2, 2, 0, 0])}
    b = {k: np.array([{0: 5, 1: 3, 2: 4}[s] for s in v]) for k, v in a.items()}
    out = label_agreement(a, b, mapping={0: 5, 1: 3, 2: 4})
    assert out["ari"] == pytest.approx(1.0)
    assert out["mapped_agreement"] == pytest.approx(1.0)
    assert out["n_trs"] == 10


def test_label_agreement_over_matched_trs_only():
    a = {"r": np.array([0, 0, 1, 1])}
    b = {"r": np.array([0, 1, 1, 1])}
    out = label_agreement(a, b, mapping={0: 0})
    assert out["mapped_agreement"] == pytest.approx(0.5)
    assert out["unmatched_fraction"] == pytest.approx(0.5)


def test_label_agreement_needs_same_runs_and_lengths():
    with pytest.raises(ValueError, match="runs"):
        label_agreement({"r1": np.zeros(3)}, {"r2": np.zeros(3)}, {})
    with pytest.raises(ValueError, match="TRs"):
        label_agreement({"r1": np.zeros(3)}, {"r1": np.zeros(4)}, {})


def test_pooled_fo_and_distance():
    a = {"r": np.array([0, 0, 0, 1])}
    b = {"r": np.array([1, 1, 1, 0])}
    fo_a, fo_b = pooled_fo(a, 2), pooled_fo(b, 2)
    np.testing.assert_allclose(fo_a, [0.75, 0.25])
    # Mapping a0->b1 and a1->b0 makes the occupancies identical.
    assert fo_distance(fo_a, fo_b, {0: 1, 1: 0}) == pytest.approx(0.0)
    # Identity mapping: total variation 0.5.
    assert fo_distance(fo_a, fo_b, {0: 0, 1: 1}) == pytest.approx(0.5)
    # Unmatched states are pooled per fit: a1 (0.25) against b0 (0.25).
    assert fo_distance(fo_a, fo_b, {0: 1}) == pytest.approx(0.0)
    # B may have more states than A; an empty unmatched B state adds nothing.
    assert fo_distance(fo_a, np.array([0.0, 0.75, 0.25]), {0: 1, 1: 2}) == pytest.approx(0.0)
    # Matched a0/b1 differ by 0.25, unmatched mass 0.25 vs 0.5: 0.5 * (0.25 + 0.25).
    assert fo_distance(fo_a, np.array([0.5, 0.5]), {0: 1}) == pytest.approx(0.25)


def _fit(means, decoded, k_used):
    return {"means": means, "decoded": decoded, "k_used": k_used}


def test_compare_fits_identical():
    rng = np.random.default_rng(2)
    means = rng.normal(size=(3, 25))
    dec = {"r": rng.integers(0, 3, size=200)}
    out = compare_fits(_fit(means, dec, 3), _fit(means, dec, 3))
    assert out["median_r"] == pytest.approx(1.0)
    assert out["ari"] == pytest.approx(1.0)
    assert out["fo_distance"] == pytest.approx(0.0)


def test_compare_fits_identical_with_inactive_states():
    # State 3 is occupied (FO 0.01 < 0.02, so inactive) and never matched;
    # identical fits must still be at FO distance 0.
    rng = np.random.default_rng(3)
    means = rng.normal(size=(4, 25))
    labels = np.repeat([0, 1, 2], 33)
    dec = {"r": np.concatenate([labels, [3]])}
    out = compare_fits(_fit(means, dec, 4), _fit(means, dec, 4))
    assert out["n_active_a"] == 3
    assert out["fo_distance"] == pytest.approx(0.0)


def test_moved_verdict():
    within = [{"median_r": 0.95, "ari": 0.6, "fo_distance": 0.10},
              {"median_r": 0.90, "ari": 0.5, "fo_distance": 0.20}]
    inside = {"median_r": 0.92, "ari": 0.55, "fo_distance": 0.15}
    v = moved_verdict(inside, within)
    assert v["moved"] is False
    assert v["by_metric"]["median_r"]["within_min"] == pytest.approx(0.90)

    worse = {"median_r": 0.85, "ari": 0.55, "fo_distance": 0.15}
    v = moved_verdict(worse, within)
    assert v["moved"] is True
    assert v["by_metric"]["median_r"]["outside"] is True
    assert v["by_metric"]["ari"]["outside"] is False

    # Larger FO distance is worse.
    v = moved_verdict({"median_r": 0.92, "ari": 0.55, "fo_distance": 0.25}, within)
    assert v["by_metric"]["fo_distance"]["outside"] is True


def test_moved_verdict_needs_within_pairs():
    with pytest.raises(ValueError):
        moved_verdict({"median_r": 1, "ari": 1, "fo_distance": 0}, [])
