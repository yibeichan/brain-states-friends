"""Comparing two HMM fits' states: matching, label agreement, occupancy, verdict."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.state_compare import (  # noqa: E402
    compare_fits, distribution_test, fo_distance, label_agreement, match_states,
    moved_verdict, pooled_fo,
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
    assert out["n_matched"] == 3          # the inactive state is not matched
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


@pytest.mark.parametrize("name, worse, better", [
    ("median_r", 0.80, 0.99), ("ari", 0.40, 0.70), ("fo_distance", 0.30, 0.05)])
def test_moved_verdict_direction_per_metric(name, worse, better):
    within = [{"median_r": 0.95, "ari": 0.6, "fo_distance": 0.10},
              {"median_r": 0.90, "ari": 0.5, "fo_distance": 0.20}]
    inside = {"median_r": 0.92, "ari": 0.55, "fo_distance": 0.15}
    assert moved_verdict(inside | {name: worse}, within)["by_metric"][name]["outside"] is True
    assert moved_verdict(inside | {name: better}, within)["by_metric"][name]["outside"] is False


def test_moved_verdict_needs_strictly_worse_than_every_pair():
    within = [{"median_r": 0.95, "ari": 0.6, "fo_distance": 0.10},
              {"median_r": 0.90, "ari": 0.5, "fo_distance": 0.20}]
    at_edge = {"median_r": 0.90, "ari": 0.5, "fo_distance": 0.20}
    v = moved_verdict(at_edge, within)
    assert v["moved"] is False
    assert not any(e["outside"] for e in v["by_metric"].values())


def test_moved_verdict_rejects_non_finite_values():
    within = [{"median_r": 0.95, "ari": 0.6, "fo_distance": 0.10},
              {"median_r": 0.90, "ari": 0.5, "fo_distance": 0.20}]
    inside = {"median_r": 0.92, "ari": 0.55, "fo_distance": 0.15}
    with pytest.raises(ValueError, match="cross-variant"):
        moved_verdict(inside | {"ari": float("nan")}, within)
    with pytest.raises(ValueError, match="within-variant"):
        moved_verdict(inside, within + [inside | {"median_r": float("nan")}])


def _pair_metrics(n, value):
    """All-pairs metrics for n fits, from value(i, j) -> (median_r, ari, fo)."""
    out = {}
    for i in range(n):
        for j in range(i + 1, n):
            r, ari, fo = value(i, j)
            out[(i, j)] = {"median_r": r, "ari": ari, "fo_distance": fo}
    return out


def test_distribution_test_detects_shift():
    labels = np.array([0] * 4 + [1] * 4)
    same = lambda i, j: labels[i] == labels[j]  # noqa: E731
    pairs = _pair_metrics(8, lambda i, j: (0.95, 0.5, 0.1) if same(i, j)
                          else (0.80, 0.3, 0.3))
    out = distribution_test(pairs, labels)
    assert out["n_splits"] == 70 and out["exact"] is True
    for name in ("median_r", "ari", "fo_distance"):
        m = out["by_metric"][name]
        assert m["statistic"] > 0
        # Only the observed split and its mirror image reach the statistic.
        assert m["p"] == pytest.approx(2 / 70)
    assert out["by_metric"]["median_r"]["within_median"] == pytest.approx(0.95)
    assert out["by_metric"]["median_r"]["cross_median"] == pytest.approx(0.80)


def _reference_p(pairs, labels, name, worse):
    """Brute force: share of label splits whose median-difference statistic is at
    least the observed one."""
    n, n1 = len(labels), int((labels == labels[0]).sum())

    def stat(lab):
        w = [pairs[(i, j)][name] for i, j in itertools.combinations(range(n), 2)
             if lab[i] == lab[j]]
        c = [pairs[(i, j)][name] for i, j in itertools.combinations(range(n), 2)
             if lab[i] != lab[j]]
        d = np.median(w) - np.median(c)
        return d if worse == "lower" else -d

    obs = stat(labels == labels[0])
    splits = []
    for idx in itertools.combinations(range(n), n1):
        lab = np.zeros(n, dtype=bool)
        lab[list(idx)] = True
        splits.append(stat(lab))
    return float(np.mean(np.array(splits) >= obs - 1e-12)), obs


def test_distribution_test_matches_brute_force_on_noise():
    labels = np.array([0] * 4 + [1] * 4)
    rng = np.random.default_rng(4)
    pairs = _pair_metrics(8, lambda i, j: tuple(rng.normal(size=3)))
    out = distribution_test(pairs, labels)
    for name, worse in (("median_r", "lower"), ("ari", "lower"), ("fo_distance", "higher")):
        p, obs = _reference_p(pairs, labels, name, worse)
        assert out["by_metric"][name]["p"] == pytest.approx(p)
        assert out["by_metric"][name]["statistic"] == pytest.approx(obs)


def test_distribution_test_statistic_is_a_median_difference():
    labels = np.array([0, 0, 1, 1])
    ari = {(0, 1): 0.8, (2, 3): 0.6, (0, 2): 0.1, (0, 3): 0.2, (1, 2): 0.3, (1, 3): 0.9}
    pairs = {k: {"median_r": 0.5, "ari": v, "fo_distance": 0.1} for k, v in ari.items()}
    out = distribution_test(pairs, labels)
    # median within 0.7, median cross 0.25 (means would give 0.7 and 0.375)
    assert out["by_metric"]["ari"]["statistic"] == pytest.approx(0.45)


def test_distribution_test_rejects_non_finite_pairs():
    labels = np.array([0] * 3 + [1] * 3)
    pairs = _pair_metrics(6, lambda i, j: (0.9, float("nan") if (i, j) == (0, 1) else 0.5, 0.1))
    with pytest.raises(ValueError, match="ari"):
        distribution_test(pairs, labels)


def test_distribution_test_random_splits_are_shared_across_metrics():
    # fo_distance = -ari and median_r = ari give the same statistic on every split,
    # so their p values agree only if every metric sees the same splits.
    labels = np.array([0] * 4 + [1] * 4)
    rng = np.random.default_rng(5)
    vals = {}
    for i in range(8):
        for j in range(i + 1, 8):
            vals[(i, j)] = float(rng.normal())
    pairs = {k: {"median_r": v, "ari": v, "fo_distance": -v} for k, v in vals.items()}
    out = distribution_test(pairs, labels, max_exact=10, n_random=200, seed=1)
    ps = {name: m["p"] for name, m in out["by_metric"].items()}
    assert ps["median_r"] == ps["ari"] == ps["fo_distance"]


def test_distribution_test_samples_when_too_many_splits():
    labels = np.array([0] * 4 + [1] * 4)
    pairs = _pair_metrics(8, lambda i, j: (0.9, 0.5, 0.1))
    out = distribution_test(pairs, labels, max_exact=10, n_random=50, seed=0)
    assert out["exact"] is False and out["n_splits"] == 51  # 50 random + observed
    # Ties everywhere: every split reaches the observed statistic.
    assert out["by_metric"]["ari"]["p"] == pytest.approx(1.0)
