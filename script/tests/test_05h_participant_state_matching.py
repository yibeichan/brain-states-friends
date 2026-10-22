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
    null, _, _ = m.pair_null(A, B, 200, rng, "configuration")
    assert m.p_greater(np.median(r), null) < 0.05


def test_configuration_null_calibrated_with_common_gradient():
    rng = np.random.default_rng(3)
    grad = np.linspace(-1, 1, 156)
    ps = []
    for _ in range(60):
        A, _, _ = _subject_maps(rng)
        B, _, _ = _subject_maps(rng)
        A += 2 * grad                                             # shared, but not state-specific
        B += 2 * grad
        r = m.match(m.corr_matrix(A, B))[2]
        null, _, _ = m.pair_null(A, B, 99, rng, "configuration")
        ps.append(m.p_greater(np.median(r), null))
    assert np.mean(np.array(ps) < 0.05) <= 0.15


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
