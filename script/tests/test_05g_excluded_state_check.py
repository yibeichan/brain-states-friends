"""Unit tests for 05g_excluded_state_check.

Claims tested: (a) segments and neighbours are built within runs, with
first/last segments censored; (b) the excess-return test detects planted
interruptions, stays calibrated when neighbours are independent, and drops
runs that no permutation changes; (c) the blend fit separates a blend of two
partners from a copy of one; (d) the parent bootstrap gives a lower reference
for shorter visits, places its windows without overlap, keeps the drawn TRs
out of the comparison map, and is undefined only when the state has at least
as many TRs as its parent; (e) the classification rules, with pass, fail and
untested arms; (f) the per-subject driver on a small synthetic subject. All
on synthetic data; nothing touches pipeline outputs.
"""
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
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
    obs, exp, p, n = m.excess_return_test(a, b, 2000, rng)
    assert obs == 1.0 and exp < 0.3 and n == 200
    assert p == pytest.approx(1 / 2001)            # observed counts as one draw


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
    long_ref, _ = m.bootstrap_parent_reference(windows, X, [30] * 20, comps, 300, rng)
    short_ref, _ = m.bootstrap_parent_reference(windows, X, [2] * 5, comps, 300, rng)
    assert short_ref < long_ref <= 1.0


def test_bootstrap_handles_visits_longer_than_any_parent_segment():
    """No zero-variance fallback: windows can span parent segment joins."""
    rng = np.random.default_rng(9)
    comps = np.linalg.qr(rng.normal(size=(156, 10)))[0].T
    X = {"r": rng.normal(size=(400, 10))}
    windows = [("r", i * 4, 3) for i in range(100)]           # 3-TR parent segments
    ref, r = m.bootstrap_parent_reference(windows, X, [8, 8, 8], comps, 200, rng)
    assert np.std(r) > 0 and ref < 1.0


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


I_FAIL = dict(q_excess_return=0.5)
I_UNTESTED = dict(q_excess_return=np.nan)
B_PASS = dict(p_through=0.8, dR2=0.3)
B_UNTESTED = dict(p_through=0.8, dR2_null_p95=np.nan)


@pytest.mark.parametrize("kw, arms, cls", [
    ({}, ("pass", "fail"), "I"),
    (B_PASS, ("pass", "pass"), "I/B"),
    (B_UNTESTED, ("pass", "untested"), "I"),
    ({**I_FAIL, **B_PASS}, ("fail", "pass"), "B"),
    ({**I_UNTESTED, **B_PASS}, ("untested", "pass"), "B"),
    (I_FAIL, ("fail", "fail"), "D"),
    ({**I_FAIL, **B_UNTESTED}, ("fail", "untested"), "untested"),
    (I_UNTESTED, ("untested", "fail"), "untested"),
    ({**I_UNTESTED, **B_UNTESTED}, ("untested", "untested"), "untested"),
    (dict(r_parent=0.55), ("fail", "fail"), "D"),                 # below ref_nn
    (dict(r_parent=0.65, ref_boot=0.7), ("fail", "fail"), "D"),   # below ref_boot
    (dict(ref_boot=np.nan), ("untested", "fail"), "untested"),     # no disjoint split
    (dict(ref_boot=np.nan, r_parent=0.55), ("fail", "fail"), "D"),  # a finite criterion fails
    (dict(r_parent=np.nan, ref_boot=np.nan), ("untested", "fail"), "untested"),
    (dict(q_excess_return=0.5, p_through=0.8, dR2=np.nan), ("fail", "untested"), "untested"),
    (dict(p_through=0.4, dR2=np.nan), ("pass", "fail"), "I"),
])
def test_classify_rules(kw, arms, cls):
    row = _row(**kw)
    assert m.arm_statuses(row, 0.05) == arms
    assert m.classify(row, 0.05) == cls


def test_classify_onset_and_insufficient_come_first():
    assert m.classify(_row(run_onset_anchored=True), 0.05) == "O"
    assert m.classify(_row(n_uncensored=5), 0.05) == "insufficient"
    assert m.classify(_row(n_uncensored=5, **I_UNTESTED), 0.05) == "insufficient"


def test_empirical_pc_means_and_active_states():
    decoded = {"r1": np.array([0, 0, 1, 1]), "r2": np.array([1, 1, 1, 1])}
    X = {"r1": np.array([[1.0], [3.0], [5.0], [5.0]]), "r2": np.ones((4, 1))}
    means, counts = m.empirical_pc_means(X, decoded, K=3)
    assert counts.tolist() == [2, 6, 0]
    assert means[0, 0] == pytest.approx(2.0)
    assert means[1, 0] == pytest.approx((5 + 5 + 4) / 6)
    assert np.isnan(means[2, 0])
    assert m.active_state_indices(decoded, 3).tolist() == [0, 1]


def test_excess_return_stratified_by_run_is_calibrated():
    """Neighbours that differ between runs must not look like interruptions."""
    rng = np.random.default_rng(11)
    ps_strat, ps_pooled = [], []
    for _ in range(150):
        a, b, runs = [], [], []
        for r in range(6):
            pool = rng.choice(20, size=2, replace=False)       # run-specific neighbours
            k = 6
            a += list(rng.choice(pool, size=k)); b += list(rng.choice(pool, size=k))
            runs += [r] * k
        ps_strat.append(m.excess_return_test(a, b, 199, rng, strata=runs)[2])
        ps_pooled.append(m.excess_return_test(a, b, 199, rng)[2])
    assert np.mean(np.array(ps_strat) < 0.05) <= 0.09
    assert np.mean(np.array(ps_pooled) < 0.05) > 0.3           # pooling is anti-conservative


def test_excess_return_expectation_equals_permutation_mean():
    rng = np.random.default_rng(12)
    a = rng.integers(0, 4, 60); b = rng.integers(0, 4, 60); runs = np.repeat(np.arange(6), 10)
    obs, exp, _, _ = m.excess_return_test(a, b, 1, rng, strata=runs)
    sims = []
    for _ in range(4000):
        bp = b.copy()
        for g in range(6):
            idx = np.flatnonzero(runs == g); bp[idx] = b[idx][rng.permutation(idx.size)]
        sims.append(np.mean(a == bp))
    assert exp == pytest.approx(np.mean(sims), abs=0.005)



def test_excess_return_drops_uninformative_runs():
    """Runs with one visit or one next label never change under permutation."""
    rng = np.random.default_rng(14)
    # every run: one visit, or all visits with the same next label
    a = [1, 2, 2, 2, 3, 3, 4]
    b = [1, 2, 2, 2, 5, 5, 4]
    runs = [0, 1, 1, 1, 2, 2, 3]
    obs, exp, p, n = m.excess_return_test(a, b, 199, rng, strata=runs)
    assert n == 0 and np.isnan(p)
    # an informative run is kept and the others are left out of the share
    a2 = a + [6, 7, 6, 7]
    b2 = b + [6, 7, 7, 6]
    obs, exp, p, n = m.excess_return_test(a2, b2, 199, rng, strata=runs + [4] * 4)
    assert n == 4 and obs == 0.5 and exp == 0.5 and 0 < p <= 1
    # min_visits counts informative visits only
    assert np.isnan(m.excess_return_test(a2, b2, 199, rng, strata=runs + [4] * 4,
                                         min_visits=5)[2])


def test_excess_return_nan_when_null_has_no_spread():
    """Two visits, two next labels, neither equal to a prev label: every draw is 0."""
    rng = np.random.default_rng(15)
    obs, exp, p, n = m.excess_return_test([1, 2], [3, 4], 99, rng)
    assert n == 2 and obs == 0 and np.isnan(p)


def _identity_comps():
    return np.eye(156)


def test_place_windows_never_overlap_and_stay_in_range():
    rng = np.random.default_rng(16)
    lengths = np.array([5, 1, 3, 3, 7, 2])
    for N in (21, 22, 40, 400):
        hit_first = hit_last = False
        for _ in range(500):
            starts, L = m.place_windows(N, lengths, rng)
            cover = np.zeros(N, dtype=int)
            for st, ln in zip(starts, L):
                assert st >= 0 and st + ln <= N
                cover[st:st + ln] += 1
            assert cover.max() == 1 and cover.sum() == lengths.sum()
            assert sorted(L.tolist()) == sorted(lengths.tolist())
            hit_first |= cover[0] == 1
            hit_last |= cover[-1] == 1
        assert hit_first and hit_last


def test_bootstrap_short_map_has_exactly_the_state_tr_count():
    """With one parcel direction per TR, r depends only on how many TRs each
    map holds, so it is the same in every draw when windows never overlap."""
    rng = np.random.default_rng(17)
    X = {"r": np.eye(156)[:60]}
    windows = [("r", i * 6, 6) for i in range(10)]
    _, r = m.bootstrap_parent_reference(windows, X, [4, 4, 4, 4, 4, 4, 4, 4],
                                        _identity_comps(), 100, rng)
    assert np.ptp(r) < 1e-12


def test_bootstrap_comparison_map_excludes_drawn_trs():
    """An extreme TR drawn into the short map must not also sit in the rest map."""
    rng = np.random.default_rng(18)
    X = {"r": rng.normal(size=(30, 156))}
    X["r"][20] = 1000 * rng.normal(size=156)
    ref, r = m.bootstrap_parent_reference([("r", 0, 30)], X, [1], _identity_comps(), 400, rng)
    assert r.size == 400 and np.max(np.abs(r)) < 0.6


def test_bootstrap_undefined_only_when_state_has_as_many_trs_as_parent():
    rng = np.random.default_rng(13)
    comps = np.linalg.qr(rng.normal(size=(156, 10)))[0].T
    X = {"r": rng.normal(size=(40, 10))}
    ref, r = m.bootstrap_parent_reference([("r", 0, 40)], X, [20, 20], comps, 50, rng)
    assert np.isnan(ref) and r.size == 0                       # n_s == N
    ref, r = m.bootstrap_parent_reference([("r", 0, 40)], X, [50], comps, 50, rng)
    assert np.isnan(ref) and r.size == 0                       # one visit longer than N
    ref, r = m.bootstrap_parent_reference([("r", 0, 40)], X, [20, 19], comps, 50, rng)
    assert np.isfinite(ref) and r.size == 50                   # n_s = N - 1
    ref, r = m.bootstrap_parent_reference([("r", 0, 40)], X, [10, 10], comps, 50, rng)
    assert np.isfinite(ref)                                    # half the parent


# ---------------------------------------------------------------- driver

E_STATES = (0, 1, 2, 3, 4, 5)
O_STATE, S_STATE, B_STATE, Z_STATE = 6, 7, 8, 9
N_RUNS, N_PCS = 6, 40


def _synthetic_subject(seed=0):
    """Six runs over six eligible states plus four planted excluded states.

    O (6): run-onset-flagged, with planted returns (would be significant).
    S (7): neighbours drawn from a run-specific pair, independent within
        each run; pooled over runs a == b looks in excess, within runs not.
    B (8): always E0 -> B -> E1, map = mean of E0 and E1; E0 and E1 are never
        adjacent, so there are no direct switches for the HRF-blend label.
    Z (9): one E2 -> Z -> E2 visit per run, map = E2 exactly, plus one
        E4 -> Z -> E3 visit in the first run: only that run can be permuted,
        and its 2 visits are fewer than min_visits.
    """
    rng = np.random.default_rng(seed)
    K = 10
    mu = rng.normal(size=(K, N_PCS))
    mu[B_STATE] = 0.5 * (mu[0] + mu[1])
    mu[Z_STATE] = mu[2]
    noise = {k: 0.5 for k in range(K)}
    noise[B_STATE], noise[Z_STATE] = 0.3, 0.0
    pairs = [(2, 3), (4, 5), (3, 5), (2, 4), (2, 5), (3, 4)]

    def ok(x, y):
        return x != y and {x, y} != {0, 1}

    decoded, X_by_run = {}, {}
    for r in range(N_RUNS):
        p_, q_ = pairs[r]
        motifs = [(x, S_STATE, y) for x, y in ((p_, p_), (p_, q_), (q_, p_), (q_, q_))] * 2
        motifs += [(0, B_STATE, 1)] * 2 + [(2, Z_STATE, 2)] + [(4, Z_STATE, 3)] * (r == 0)
        motifs += [(3, O_STATE, 3), (3, O_STATE, 3), (4, O_STATE, 4), (4, O_STATE, 4)]
        order = rng.permutation(len(motifs))
        segs = [(5, 4)]
        for i in order:
            left, mid, right = motifs[i]
            fill = [e for e in E_STATES if ok(segs[-1][0], e) and ok(e, left)]
            segs.append((int(rng.choice(fill)), 4))
            seg_len = {S_STATE: 2, B_STATE: 8, Z_STATE: 6, O_STATE: 3}[mid]
            segs += [(left, 4), (mid, seg_len), (right, 4)]
        fill = [e for e in E_STATES if ok(segs[-1][0], e)]
        segs.append((int(rng.choice(fill)), 4))
        labels = np.concatenate([[k] * n for k, n in segs]).astype(int)
        decoded[f"run-{r}"] = labels
        X_by_run[f"run-{r}"] = mu[labels] + np.array(
            [noise[k] for k in labels])[:, None] * rng.normal(size=(labels.size, N_PCS))
    comps = np.linalg.qr(rng.normal(size=(156, N_PCS)))[0].T
    flags = pd.DataFrame({
        "state": np.arange(K),
        "summary_category": [m.ELIGIBLE] * 6 + ["run_onset"] + ["low_confidence"] * 3,
        "unused": False, "run_onset": [False] * 6 + [True] + [False] * 3,
        "a_anchored": False, "b_anchored": False, "sub_hrf": False,
        "recurrence_score": 1.0}).set_index("state")
    return {"decoded": decoded, "X_by_run": X_by_run, "comps": comps,
            "means_parcel_contrast": mu @ comps,
            "covars": np.exp(rng.normal(size=(K, N_PCS))),
            "flags": flags, "n_pcs": N_PCS, "max_abs_pca_mean": 0.0}


@pytest.fixture(scope="module")
def driver_result():
    inputs = _synthetic_subject()
    mp = pytest.MonkeyPatch()
    mp.setattr(m, "load_inputs", lambda *a, **k: inputs)
    args = Namespace(n_perm=999, n_boot=200, min_visits=5, alpha=0.05, seed=0)
    try:
        df, summary = m.analyse_subject("sub-01", "parc", "0.95", "unused", args)
    finally:
        mp.undo()
    return inputs, df.set_index("state"), summary


def test_driver_onset_state_is_outside_bh(driver_result):
    _, df, _ = driver_result
    assert df.loc[O_STATE, "class"] == "O"
    assert df.loc[O_STATE, "p_excess_return"] < 0.01         # tested, but not corrected
    assert np.isnan(df.loc[O_STATE, "q_excess_return"])
    assert pd.isna(df.loc[O_STATE, "interruption_status"])


def test_driver_stratifies_by_run(driver_result):
    inputs, df, _ = driver_result
    assert df.loc[S_STATE, "p_excess_return"] > 0.05
    v = m.visit_table(inputs["decoded"])
    g = v[(v["state"] == S_STATE) & (v["prev"] >= 0) & (v["next"] >= 0)]
    pooled = m.excess_return_test(g["prev"], g["next"], 999, np.random.default_rng(0))[2]
    assert pooled < 0.05                                       # pooling would reject


def test_driver_blend_without_switches_is_untested_for_hrf(driver_result):
    _, df, summary = driver_result
    assert df.loc[B_STATE, "class"] == "B"
    assert df.loc[B_STATE, "blend_status"] == "pass"
    assert df.loc[B_STATE, "n_switches"] == 0
    assert df.loc[B_STATE, "hrf_blend"] is None
    assert summary["n_hrf_blend_untested"] == 1
    assert (df.loc[B_STATE, "partner_a_category"], df.loc[B_STATE, "partner_b_category"]) \
        == (m.ELIGIBLE, m.ELIGIBLE)


def test_driver_uninformative_state_is_untested_not_d(driver_result):
    _, df, summary = driver_result
    row = df.loc[Z_STATE]
    assert row["n_uncensored"] == N_RUNS + 1 and row["n_informative"] == 2
    assert np.isnan(row["p_excess_return"]) and np.isnan(row["q_excess_return"])
    assert row["parent"] == 2 and row["parent_category"] == m.ELIGIBLE
    assert row["r_parent"] >= row["ref_nn"] and row["r_parent"] >= row["ref_boot"]
    assert (row["interruption_status"], row["blend_status"]) == ("untested", "fail")
    assert row["class"] == "untested"
    assert summary["class_counts"].get("untested", 0) >= 1
    assert summary["n_p_excess_return_untested"] >= 2         # Z and B
