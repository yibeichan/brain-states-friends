"""Tests for utils.transition_modules (06e).

Claims tested: the rewiring null keeps in/out degree and reciprocal dyads
exactly, creates no self-loops or duplicates, keeps total weight exactly and
strengths approximately; the row shuffle keeps out-strength; the segment
shuffle keeps each state's transition total and destroys only order; the
non-modular chain has rows that sum to one; best_modularity is reproducible
and finds planted modules; MRQAP recovers a planted predictor. Synthetic data
only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from utils import transition_modules as tm


def _random_graph(n=30, p=0.2, recip=0.5, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.random((n, n)) < p
    np.fill_diagonal(A, False)
    # make a share of edges reciprocal
    flip = np.triu(rng.random((n, n)) < recip, 1) & (A | A.T)
    A = A | flip | flip.T
    np.fill_diagonal(A, False)
    W = np.where(A, rng.lognormal(-4, 1, (n, n)), 0.0)
    return W


def _planted(n=40, k=4, p_in=0.5, p_out=0.05, seed=1):
    rng = np.random.default_rng(seed)
    lab = np.repeat(np.arange(k), n // k)
    same = lab[:, None] == lab[None, :]
    A = rng.random((n, n)) < np.where(same, p_in, p_out)
    np.fill_diagonal(A, False)
    return np.where(A, rng.lognormal(-4, 0.5, (n, n)), 0.0), lab


def test_weight_matrix_threshold_and_renorm():
    P = np.array([[0.8, 0.15, 0.05], [0.1, 0.7, 0.2], [0.009, 0.5, 0.491]])
    W = tm.weight_matrix(P, [0, 1, 2], 0.01)
    assert W[2, 0] == 0 and W[0, 2] == 0.05 and np.all(np.diag(W) == 0)
    R = tm.weight_matrix(P, [0, 1, 2], None, renorm_rows=True)
    assert np.allclose(R.sum(axis=1), 1.0)


def test_rewiring_invariants():
    W = _random_graph()
    A = W > 0
    rng = np.random.default_rng(3)
    Wn, info = tm.rewire_null(W, rng)
    An = Wn > 0
    assert info["accepted"] > 5 * A.sum()                      # mixed thoroughly
    assert np.array_equal(An.sum(0), A.sum(0))                 # in-degree
    assert np.array_equal(An.sum(1), A.sum(1))                 # out-degree
    assert (An & An.T).sum() == (A & A.T).sum()                # reciprocal edges
    assert not np.any(np.diag(An))                             # no self-loops
    assert An.sum() == A.sum()                                 # no duplicates lost
    assert np.isclose(Wn.sum(), W.sum())                       # total weight exact
    assert sorted(Wn[An]) == pytest.approx(sorted(W[A]))       # same weights
    assert tm.strength_error(Wn, W) < 0.6                      # strengths roughly kept
    assert not np.array_equal(An, A)                           # topology changed


def test_row_shuffle_keeps_out_strength():
    W = _random_graph(seed=5)
    Wn = tm.row_shuffle_null(W, np.random.default_rng(0))
    assert np.allclose(Wn.sum(1), W.sum(1)) and not np.any(np.diag(Wn))


def test_segment_shuffle_keeps_totals_and_breaks_order():
    rng = np.random.default_rng(0)
    K = 6
    decoded = {f"r{i}": np.repeat(rng.integers(0, K, 40), rng.integers(1, 6, 40))
               for i in range(5)}
    C = sum(tm.run_transition_counts(decoded, K).values())
    segs = tm.segment_tables(decoded)
    Cs = tm.shuffled_counts(segs, K, rng)
    assert Cs.sum() == pytest.approx(C.sum())                  # one transition per TR step
    trs = np.bincount(np.concatenate(list(decoded.values())), minlength=K)
    assert np.all(Cs.sum(axis=1) <= trs) and np.all(Cs.sum(axis=1) >= trs - 5)
    assert not np.allclose(Cs, C)


def test_shuffled_counts_match_direct_count():
    """The fast count equals counting the rebuilt shuffled sequence."""
    K = 4
    decoded = {"r": np.array([0, 0, 1, 2, 2, 2, 1, 3, 3, 0])}
    segs = tm.segment_tables(decoded)
    rng1, rng2 = np.random.default_rng(7), np.random.default_rng(7)
    Cs = tm.shuffled_counts(segs, K, rng1)
    lab, ln = segs["r"]
    perm = rng2.permutation(lab.size)
    seq = np.repeat(lab[perm], ln[perm])
    C = tm.run_transition_counts({"r": seq}, K)["r"]
    assert np.array_equal(Cs, C)


def test_nonmodular_chain_rows():
    rng = np.random.default_rng(0)
    P = rng.random((8, 8)) + np.eye(8) * 5
    P /= P.sum(1, keepdims=True)
    Q, p_stay, q = tm.nonmodular_chain(P, list(range(8)))
    assert np.allclose(Q.sum(1), 1.0) and np.allclose(np.diag(Q), p_stay)
    C = tm.simulate_chain_counts(Q, [200, 300], [0, 3], rng)
    assert C.sum() == 498


def test_best_modularity_reproducible_and_finds_planted_modules():
    W, lab = _planted()
    q1, l1, _ = tm.best_modularity(W, range(10))
    q2, l2, _ = tm.best_modularity(W, range(10))
    assert q1 == q2 and np.array_equal(l1, l2)
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(lab, l1) > 0.9
    Wn, _ = tm.rewire_null(W, np.random.default_rng(1))
    qn, _, _ = tm.best_modularity(Wn, range(10))
    assert q1 > qn + 0.1


def test_module_dwell_and_shuffle():
    decoded = {"r": np.array([0, 0, 1, 1, 2, 2, 3, 3])}
    mod = np.array([0, 0, 1, 1])
    assert tm.module_dwell(decoded, mod) == 4.0
    segs = tm.segment_tables(decoded)
    vals = [tm.module_dwell_shuffled(segs, mod, np.random.default_rng(s)) for s in range(50)]
    assert min(vals) >= 2.0 and np.mean(vals) < 4.0


def test_mrqap_recovers_planted_predictor():
    rng = np.random.default_rng(0)
    n = 30
    X1 = rng.normal(size=(n, n)); X1 = (X1 + X1.T) / 2
    X2 = rng.normal(size=(n, n)); X2 = (X2 + X2.T) / 2
    Y = 2.0 * X1 + rng.normal(scale=0.5, size=(n, n)); Y = (Y + Y.T) / 2
    out = tm.mrqap_dekker(Y, {"x1": X1, "x2": X2}, rng, n_perm=200)
    assert out["x1"]["p"] < 0.01 and out["x2"]["p"] > 0.01


def test_predicted_null_graph_shape_and_threshold():
    W = _random_graph(n=12, p=0.9, seed=2) + 1e-3
    np.fill_diagonal(W, 0)
    preds = {"x": np.random.default_rng(1).normal(size=(12, 12))}
    model = tm.predicted_weight_model(W, preds)
    G = tm.predicted_null_graph(model, 12, 0.01, np.random.default_rng(0))
    assert G.shape == (12, 12) and not np.any(np.diag(G)) and np.all((G == 0) | (G >= 0.01))


def test_consensus_partition_on_clear_blocks():
    lab = np.repeat([0, 1, 2], 6)
    D = (lab[:, None] == lab[None, :]).astype(float) * 0.9 + 0.05
    cons, _, converged = tm.consensus_partition(D, np.random.default_rng(0), n_runs=20)
    from sklearn.metrics import adjusted_rand_score
    assert converged and adjusted_rand_score(lab, cons) == 1.0


def test_bhattacharyya_zero_on_identical():
    mu = np.array([[0.0, 1.0], [0.0, 1.0], [3.0, 1.0]])
    var = np.ones((3, 2))
    D = tm.bhattacharyya_diag(mu, var)
    assert D[0, 1] == pytest.approx(0.0) and D[0, 2] > 1.0
