"""Tests for the Figure S15 helpers in fig_F3_transition_structure.py.

Claims tested: the FC percentile mapping ranks each off-diagonal pair once,
excludes the diagonal and undefined pairs, averages ties and spans 0-100; the
within- minus between-module difference matches a hand computation; the 06e
cross-check passes on agreement and raises on a small disagreement or an
undefined similarity, naming the state; display_order returns Panel A's
module order; draw_state_matrix draws one ribbon cell per state per margin,
the flagged-state dots and one separator per module boundary. Synthetic data
only.
"""
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "fig_F3_transition_structure",
    Path(__file__).resolve().parent.parent / "fig_F3_transition_structure.py",
)
f3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f3)


def _sym(upper, n):
    """Symmetric n x n matrix from its upper-triangle values (row-major)."""
    S = np.full((n, n), 9.0)
    S[np.triu_indices(n, 1)] = upper
    S.T[np.triu_indices(n, 1)] = upper
    return S


def test_offdiag_percentile_ranks_pairs_once_and_spans_0_to_100():
    S = _sym([0.1, 0.4, 0.2, 0.6, 0.3, 0.5], 4)
    P = f3.offdiag_percentile(S)
    assert np.isnan(np.diag(P)).all()
    assert np.array_equal(P, P.T, equal_nan=True)
    upper = P[np.triu_indices(4, 1)]
    # ranks 1..6 of six distinct values -> (rank - 1) / 5 * 100
    assert np.allclose(upper, [0, 60, 20, 100, 40, 80])


def test_offdiag_percentile_averages_ties_and_skips_undefined_pairs():
    S = _sym([0.2, 0.2, 0.5, np.nan, 0.9, 0.1], 4)
    P = f3.offdiag_percentile(S)
    upper = P[np.triu_indices(4, 1)]
    assert np.isnan(upper[3])
    # five finite values: 0.1 -> 0, the two 0.2s share rank 2.5, 0.5 -> 4, 0.9 -> 5
    assert np.allclose(upper[[0, 1, 2, 4, 5]], [37.5, 37.5, 75, 100, 0])


def test_offdiag_percentile_needs_two_finite_pairs():
    S = _sym([0.3, np.nan, np.nan], 3)
    assert np.isnan(f3.offdiag_percentile(S)).all()


def test_within_minus_between_matches_hand_computation():
    # labels [0, 0, 1, 1]: within pairs (0,1)=0.8 and (2,3)=0.6,
    # between pairs (0,2)=0.1, (0,3)=0.2, (1,2)=0.3, (1,3)=0.4
    S = _sym([0.8, 0.1, 0.2, 0.3, 0.4, 0.6], 4)
    assert f3.within_minus_between(S, [0, 0, 1, 1]) == pytest.approx(0.7 - 0.25)
    # relabelling the modules leaves co-membership, and the value, unchanged
    assert f3.within_minus_between(S, [5, 5, 2, 2]) == pytest.approx(0.45)


def test_check_against_06e_passes_on_agreement_and_returns_the_value():
    S = _sym([0.8, 0.1, 0.2, 0.3, 0.4, 0.6], 4)
    assert f3.check_against_06e("sub-00", "fc_rv", S, [0, 0, 1, 1],
                                [10, 11, 12, 13], 0.45) == pytest.approx(0.45)


def test_check_against_06e_raises_on_a_small_disagreement():
    S = _sym([0.8, 0.1, 0.2, 0.3, 0.4, 0.6], 4)
    mine = f3.within_minus_between(S, [0, 0, 1, 1])
    with pytest.raises(RuntimeError, match=r"sub-00 map_r: .*differs from 06e"):
        f3.check_against_06e("sub-00", "map_r", S, [0, 0, 1, 1],
                             [10, 11, 12, 13], mine + 2e-9)


def test_check_against_06e_names_states_with_undefined_similarity():
    S = _sym([0.8, 0.1, np.nan, 0.3, 0.4, 0.6], 4)
    with pytest.raises(RuntimeError, match=r"undefined similarity for states \[10, 13\]"):
        f3.check_against_06e("sub-00", "fc_rv", S, [0, 0, 1, 1],
                             [10, 11, 12, 13], 0.45)


def test_check_against_06e_raises_when_06e_value_is_undefined():
    S = _sym([0.8, 0.1, 0.2, 0.3, 0.4, 0.6], 4)
    with pytest.raises(RuntimeError, match="differs from 06e"):
        f3.check_against_06e("sub-00", "fc_rv", S, [0, 0, 1, 1],
                             [10, 11, 12, 13], float("nan"))


def test_display_order_is_panel_A_module_order(tmp_path):
    G = nx.DiGraph()
    rec = {3: 0.2, 7: 0.9, 1: 0.5, 4: 0.4}
    nets = {3: "Visual", 7: "Default", 1: "Limbic", 4: "Visual"}
    for s in rec:
        G.add_node(str(s), recurrence_score=rec[s], dominant_network=nets[s])
    G.add_edge("3", "7", weight=0.1)
    graph = tmp_path / "transition_graph.graphml"
    nx.write_graphml(G, graph)
    part = tmp_path / "partitions.json"
    part.write_text(json.dumps({"nodes": [3, 7, 1, 4], "labels": [0, 1, 0, 1]}))

    _G, order, mods, recs, onets = f3.display_order(graph, part)
    assert (order, mods) == f3.module_order([3, 7, 1, 4], [0, 1, 0, 1], rec)
    assert order == [7, 4, 1, 3]
    assert recs == rec
    assert onets == nets


def test_draw_state_matrix_draws_ribbons_dots_and_separators():
    fig, ax = plt.subplots()
    M = np.ma.masked_invalid(np.arange(16, dtype=float).reshape(4, 4))
    f3.draw_state_matrix(ax, M, plt.get_cmap("viridis"),
                         matplotlib.colors.Normalize(0, 15),
                         ["Visual", "Default", "Default", "Limbic"],
                         [0, 0, 1, 1], [True, False, False, True],
                         {"Visual": "#111111", "Default": "#222222"}, "#8C8C8C")
    assert len(ax.images) == 1
    assert len(ax.patches) == 8          # one ribbon cell per state, top and left
    assert len(ax.collections) == 2      # flagged-state dots, top and left
    assert all(len(c.get_offsets()) == 2 for c in ax.collections)
    assert len(ax.lines) == 2            # one boundary, drawn across and down
    plt.close(fig)
