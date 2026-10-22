"""Unit tests for 05g_excluded_state_check.

Claims tested: (a) segments and neighbours are built within runs, with
first/last segments censored; (b) the excess-return test detects planted
interruptions and stays calibrated when neighbours are independent;
(c) the blend fit separates a blend of two partners from a copy of one;
(d) the parent bootstrap gives a lower reference for shorter visits;
(e) the classification rules. All on synthetic data; nothing touches
pipeline outputs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

import importlib
m = importlib.import_module("05g_excluded_state_check")


def test_run_segments():
    segs = m.run_segments(np.array([0, 0, 1, 1, 1, 0, 2]))
    assert segs == [(0, 0, 2), (1, 2, 3), (0, 5, 1), (2, 6, 1)]
    assert m.run_segments(np.array([], dtype=int)) == []


def test_visit_table_censors_run_edges_and_never_crosses_runs():
    decoded = {"r1": np.array([5, 5, 7, 5]), "r2": np.array([7, 9, 9])}
    v = m.visit_table(decoded).set_index(["run", "start"])
    assert v.loc[("r1", 0), "prev"] == -1          # first segment censored
    assert v.loc[("r1", 3), "next"] == -1          # last segment censored
    assert tuple(v.loc[("r1", 2), ["prev", "next"]]) == (5, 5)
    assert v.loc[("r2", 0), "prev"] == -1          # r1's last label not carried over
    assert len(v) == 5


def test_excess_return_detects_interruptions():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 6, size=200)
    b = a.copy()                                    # every visit returns
    obs, exp, p = m.excess_return_test(a, b, 2000, rng)
    assert obs == 1.0 and exp < 0.3 and p < 0.001


def test_excess_return_calibrated_under_independence():
    rng = np.random.default_rng(1)
    ps = []
    for _ in range(200):
        a = rng.integers(0, 4, size=40)
        b = rng.integers(0, 4, size=40)
        ps.append(m.excess_return_test(a, b, 199, rng)[2])
    assert 0.02 <= np.mean(np.array(ps) < 0.05) <= 0.09


def test_blend_fit_separates_blend_from_copy():
    rng = np.random.default_rng(2)
    a_map, b_map = rng.normal(size=156), rng.normal(size=156)
    blend = 0.6 * a_map + 0.5 * b_map + 0.1 * rng.normal(size=156)
    copy = a_map + 0.1 * rng.normal(size=156)
    deriv = b_map - a_map + 0.1 * rng.normal(size=156)
    assert m.blend_fit(blend, a_map, b_map)["dR2"] > 0.3
    assert m.blend_fit(copy, a_map, b_map)["dR2"] < 0.02
    res = m.blend_fit(deriv, a_map, b_map)               # derivative-like map
    assert res["dR2"] > 0.3 and res["w_a"] < 0 < res["w_b"]


def test_through_partners():
    assert m.through_partners([1, 1, 2, 3], [1, 2, 1, 3]) == (1, 2)
    assert m.through_partners([1, 2], [1, 2]) is None


def test_bootstrap_reference_drops_for_short_visits():
    rng = np.random.default_rng(3)
    comps = np.linalg.qr(rng.normal(size=(156, 10)))[0].T     # (10, 156)
    mu = rng.normal(size=10)
    X = {"r": mu + rng.normal(scale=2.0, size=(4000, 10))}
    windows = [("r", i * 40, 40) for i in range(100)]
    parent_map = X["r"].mean(axis=0) @ comps
    long_ref, _ = m.bootstrap_parent_reference(windows, X, [30] * 20, comps,
                                               parent_map, 300, rng)
    short_ref, _ = m.bootstrap_parent_reference(windows, X, [2] * 5, comps,
                                                parent_map, 300, rng)
    assert short_ref < long_ref <= 1.0


def test_nn_reference_is_percentile_of_nearest_neighbour_r():
    rng = np.random.default_rng(4)
    maps = rng.normal(size=(6, 156))
    maps[1] = maps[0] + 0.01 * rng.normal(size=156)
    ref, nn = m.nn_reference(maps, q=95)
    assert nn[0] > 0.99 and nn[1] > 0.99 and ref > 0.9


def _row(**kw):
    base = dict(run_onset_anchored=False, n_uncensored=50, min_visits=10,
                q_excess_return=0.001, r_parent=0.8, ref_nn=0.6, ref_boot=0.5,
                p_through=0.2, dR2=0.0, dR2_null_p95=0.1)
    base.update(kw)
    return base


def test_classify_rules():
    assert m.classify(_row(), 0.05) == "I"
    assert m.classify(_row(run_onset_anchored=True), 0.05) == "O"
    assert m.classify(_row(n_uncensored=5), 0.05) == "insufficient"
    assert m.classify(_row(r_parent=0.55), 0.05) == "D"        # below ref_nn
    assert m.classify(_row(r_parent=0.65, ref_boot=0.7), 0.05) == "D"  # below ref_boot
    assert m.classify(_row(q_excess_return=0.5, p_through=0.8, dR2=0.3), 0.05) == "B"
    assert m.classify(_row(p_through=0.6, dR2=0.3), 0.05) == "I/B"
    assert m.classify(_row(q_excess_return=0.5, p_through=0.8, dR2=np.nan), 0.05) == "D"


def test_empirical_pc_means_and_active_states():
    decoded = {"r1": np.array([0, 0, 1, 1]), "r2": np.array([1, 1, 1, 1])}
    X = {"r1": np.array([[1.0], [3.0], [5.0], [5.0]]), "r2": np.ones((4, 1))}
    means, counts = m.empirical_pc_means(X, decoded, K=3)
    assert counts.tolist() == [2, 6, 0]
    assert means[0, 0] == pytest.approx(2.0)
    assert means[1, 0] == pytest.approx((5 + 5 + 4) / 6)
    assert np.isnan(means[2, 0])
    assert m.active_states(decoded, 3).tolist() == [0, 1]
