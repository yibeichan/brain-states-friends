"""Unit tests for sm_rel_r3_assortativity_null.

Load-bearing claims: (a) the coefficient is Newman's numeric assortativity,
i.e. the Pearson correlation of the node attribute across directed edge
endpoints; (b) the strength-stratified permutation never moves a label across
strength bins; (c) rank residualization removes the linear rank dependence on
pi. All on toy graphs; nothing touches pipeline outputs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import networkx as nx
import pytest

import sm_rel_r3_assortativity_null as m


def _toy_graph(attr_by_node, edges):
    G = nx.DiGraph()
    for n, a in attr_by_node.items():
        G.add_node(n, recurrence_score=float(a))
    for u, v, w in edges:
        G.add_edge(u, v, weight=float(w))
    return G


def test_assortativity_equals_pearson_over_edge_endpoints():
    G = _toy_graph({0: 1, 1: 2, 2: 5, 3: 7},
                   [(0, 1, .5), (1, 0, .5), (2, 3, .5), (3, 2, .5), (0, 3, .1)])
    src = [G.nodes[u]["recurrence_score"] for u, _ in G.edges]
    dst = [G.nodes[v]["recurrence_score"] for _, v in G.edges]
    expected = np.corrcoef(src, dst)[0, 1]
    assert m.assortativity(G, "recurrence_score") == pytest.approx(expected, abs=1e-12)


def test_perfectly_assortative_pairs_give_one():
    G = _toy_graph({0: 1, 1: 1, 2: 5, 3: 5}, [(0, 1, 1), (1, 0, 1), (2, 3, 1), (3, 2, 1)])
    assert m.assortativity(G, "recurrence_score") == pytest.approx(1.0)


def test_node_strength_sums_weighted_in_and_out_degree():
    G = _toy_graph({0: 0, 1: 0, 2: 0}, [(0, 1, .2), (1, 0, .3), (2, 0, .4)])
    s = m.node_strength(G)
    assert s[0] == pytest.approx(.2 + .3 + .4) and s[1] == pytest.approx(.2 + .3) and s[2] == pytest.approx(.4)


def test_strength_bins_are_contiguous_quantiles():
    strength = {n: float(n) for n in range(45)}      # 45 nodes like a real repertoire
    bins = m.strength_bins(strength, n_bins=5)
    sizes = np.bincount(list(bins.values()))
    assert sizes.tolist() == [9, 9, 9, 9, 9]
    assert all(bins[n] <= bins[n + 1] for n in range(44))   # monotone in strength


def test_stratified_permutation_never_crosses_bins():
    values = {n: float(n) for n in range(20)}
    bins = {n: n // 5 for n in range(20)}
    rng = np.random.default_rng(0)
    for _ in range(50):
        perm = m.stratified_permutation(values, bins, rng)
        for b in range(4):
            nodes = [n for n in range(20) if bins[n] == b]
            assert sorted(perm[n] for n in nodes) == sorted(values[n] for n in nodes)


def test_rank_residuals_are_orthogonal_to_rank_of_covariate():
    rng = np.random.default_rng(1)
    y = {n: rng.uniform() for n in range(30)}
    x = {n: y[n] + 0.1 * rng.normal() for n in range(30)}
    from scipy.stats import rankdata
    r = m.rank_residuals(x, y)
    ry = rankdata([y[n] for n in sorted(y)])
    res = np.array([r[n] for n in sorted(y)])
    assert abs(np.dot(res, ry - ry.mean())) < 1e-8 and abs(res.sum()) < 1e-8


def test_permutation_null_restores_original_attribute_and_is_seeded():
    G = _toy_graph({0: 1, 1: 2, 2: 5, 3: 7},
                   [(0, 1, .5), (1, 0, .5), (2, 3, .5), (3, 2, .5), (0, 3, .1)])
    values = {n: G.nodes[n]["recurrence_score"] for n in G.nodes}
    n1 = m.permutation_null(G, "recurrence_score", values, 20, np.random.default_rng(5))
    n2 = m.permutation_null(G, "recurrence_score", values, 20, np.random.default_rng(5))
    np.testing.assert_array_equal(n1, n2)
    assert {n: G.nodes[n]["recurrence_score"] for n in G.nodes} == values
    assert n1.shape == (20,) and np.all(np.isfinite(n1))


def test_stratified_null_with_one_node_per_bin_is_degenerate_constant():
    G = _toy_graph({0: 1, 1: 2, 2: 5, 3: 7}, [(0, 1, .5), (1, 2, .5), (2, 3, .5), (3, 0, .5)])
    values = {n: G.nodes[n]["recurrence_score"] for n in G.nodes}
    bins = {n: n for n in G.nodes}                      # nothing can move
    null = m.permutation_null(G, "recurrence_score", values, 5, np.random.default_rng(0), bins=bins)
    assert np.allclose(null, m.assortativity(G, "recurrence_score"))
