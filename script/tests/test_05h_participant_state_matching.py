"""Unit tests for 05h_participant_state_matching.

Claims tested: (a) the configuration rotation keeps the Gram matrix and the
span of the centred maps and changes which direction is which state; (b) the
PC-block rotation keeps each state's norm within every block; (c) matching
recovers a planted correspondence between two subjects with different
subspaces, and the configuration null does not; (d) the configuration null is
calibrated when the two repertoires share nothing beyond a common gradient;
(e) joint-count occupancy r equals the direct Pearson r of binary occupancy;
(f) network collapse and component removal behave as defined. Synthetic data
only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

import importlib
m = importlib.import_module("05h_participant_state_matching")


def _subject_maps(rng, k=20, P=156, n_pcs=60, shared=None, noise=0.5):
    basis = np.linalg.qr(rng.normal(size=(P, n_pcs)))[0].T        # (n_pcs, P)
    spec = np.linspace(3, 0.2, n_pcs)
    mu = rng.normal(size=(k, n_pcs)) * spec
    maps = mu @ basis
    if shared is not None:
        maps[: shared.shape[0]] = shared + noise * rng.normal(size=shared.shape)
    return maps, mu, basis


def test_configuration_rotation_keeps_gram_and_span():
    rng = np.random.default_rng(0)
    M, _, _ = _subject_maps(rng)
    R = m.rotate_configuration(M, rng)
    C, Cr = M - M.mean(0), R - R.mean(0)
    assert np.allclose(C @ C.T, Cr @ Cr.T, atol=1e-8)
    # same span: projecting Cr onto span(C) changes nothing
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    assert np.allclose(Cr @ Vt.T @ Vt, Cr, atol=1e-8)
    assert not np.allclose(Cr, C)


def test_pc_block_rotation_keeps_block_norms():
    rng = np.random.default_rng(1)
    M, mu, basis = _subject_maps(rng)
    R = m.rotate_pc_blocks(mu, basis, rng, block=5)
    mu_r = R @ basis.T                                            # back to PC coords
    for s in range(0, mu.shape[1], 5):
        assert np.allclose(np.linalg.norm(mu[:, s:s + 5], axis=1),
                           np.linalg.norm(mu_r[:, s:s + 5], axis=1))


def test_planted_correspondence_beats_configuration_null():
    rng = np.random.default_rng(2)
    shared = rng.normal(size=(12, 156))
    A, _, _ = _subject_maps(rng, shared=shared)
    B, _, _ = _subject_maps(rng, shared=shared)
    r = m.match(m.corr_matrix(A, B))[2]
    null = m.pair_null(A, B, 200, rng, "configuration")[0]
    assert m.p_greater(np.median(r), null) < 0.05


def test_configuration_null_calibrated_with_common_gradient():
    rng = np.random.default_rng(3)
    grad = np.linspace(-1, 1, 156)
    ps = []
    for _ in range(150):
        A, _, _ = _subject_maps(rng, k=12)
        B, _, _ = _subject_maps(rng, k=12)
        A += 2 * grad                                             # shared, but not state-specific
        B += 2 * grad
        r = m.match(m.corr_matrix(A, B))[2]
        null = m.pair_null(A, B, 99, rng, "configuration")[0]
        ps.append(m.p_greater(np.median(r), null))
    assert np.mean(np.array(ps) < 0.05) <= 0.10


def test_occupancy_r_matches_direct_pearson():
    rng = np.random.default_rng(4)
    la = {"r1": rng.integers(-1, 3, 50), "r2": rng.integers(-1, 3, 40)}
    lb = {"r1": rng.integers(-1, 4, 55), "r2": rng.integers(-1, 4, 40)}
    runs = ["r1", "r2"]
    counts = m.joint_counts(la, lb, runs, 3, 4)
    T = 50 + 40
    got = m.occupancy_r(counts, T, [(0, 1), (2, 3)])
    xa = np.concatenate([la["r1"][:50], la["r2"]])
    xb = np.concatenate([lb["r1"][:50], lb["r2"]])
    for (a, b), g in zip([(0, 1), (2, 3)], got):
        assert g == pytest.approx(np.corrcoef(xa == a, xb == b)[0, 1])


def test_collapse_and_project_out():
    rng = np.random.default_rng(5)
    M = rng.normal(size=(4, 6))
    groups = np.array(["a", "a", "b", "b", "b", "c"])
    C = m.collapse_networks(M, groups)
    assert np.allclose(C[:, 0], C[:, 1]) and np.allclose(C[:, 2], M[:, 2:5].mean(1))
    V = m.shared_components([rng.normal(size=(10, 6))], 2)
    P = m.project_out(M, V)
    assert np.allclose(P @ V.T, 0, atol=1e-10)


def test_many_to_one_counts_collisions():
    R = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.7]])
    best, coll = m.many_to_one(R)
    assert coll == 1 and best.tolist() == [0.9, 0.8, 0.7]


def test_p_values_handle_nan():
    assert np.isnan(m.p_greater(np.nan, np.arange(10.0)))
    null = np.array([0.1, np.nan, 0.3, np.nan])
    assert m.p_greater(0.2, null) == pytest.approx((1 + 1) / (1 + 2))
    assert m.p_less(0.2, null) == pytest.approx((1 + 1) / (1 + 2))


def test_label_map_does_not_need_sorted_states():
    seq = np.array([4, 2, 9, 4, 7])
    out = m.label_map(seq, [9, 4, 2], 10)
    assert out.tolist() == [1, 2, 0, 1, -1]


def _toy_participants(rng, locked=True, n_runs=8, L=300):
    """Two participants with 3 states each; state 0 of A and 0 of B co-occur."""
    da = {"decoded": {}, "K": 3}
    db = {"decoded": {}, "K": 3}
    for r in range(n_runs):
        base = np.repeat(rng.integers(0, 3, L // 10), 10)
        other = np.repeat(rng.integers(0, 3, L // 10), 10)
        da["decoded"][f"r{r}"] = base
        db["decoded"][f"r{r}"] = base.copy() if locked else other
    return da, db


def test_locking_flags_planted_pair_and_not_shuffled():
    rng = np.random.default_rng(3)
    da, db = _toy_participants(rng, locked=True)
    runs = sorted(da["decoded"])
    obs, null, n_runs, T = m.locking_matrices(da, db, [0, 1, 2], [0, 1, 2], runs, 200, rng)
    r, thr, fwer, pointwise = m.lock_flags(obs, null, np.array([0, 1, 2]), np.array([0, 1, 2]))
    assert all(fwer) and r.min() > 0.9 and n_runs == 8
    da, db = _toy_participants(rng, locked=False)
    obs, null, _, _ = m.locking_matrices(da, db, [0, 1, 2], [0, 1, 2], runs, 200, rng)
    _, _, fwer, _ = m.lock_flags(obs, null, np.array([0, 1, 2]), np.array([0, 1, 2]))
    assert not any(fwer)


def test_locking_trim_drops_run_edges():
    rng = np.random.default_rng(4)
    da, db = _toy_participants(rng, locked=True, n_runs=2, L=100)
    runs = sorted(da["decoded"])
    _, _, _, T = m.locking_matrices(da, db, [0, 1, 2], [0, 1, 2], runs, 5, rng, trim=10)
    assert T == 2 * (100 - 20)


def test_cosine_keeps_amplitude_sign():
    A = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, -1.0]])
    B = np.array([[-2.0, -2.0, -2.0]])
    C = m.cos_matrix(A, B)
    assert C[0, 0] == pytest.approx(-1.0) and C[1, 0] == pytest.approx(0.0)
