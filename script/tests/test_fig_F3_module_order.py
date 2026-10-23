"""Tests for the Fig 3A ordering helpers in fig_F3_transition_structure.py.

Claims tested: modules are ranked by their most recurrent state, states run
by recurrence within a module, ties break deterministically; module
boundaries fall where the rank changes; the partition loader fails when the
file is missing or its states differ from the graph's; the caption numbers
come from the N2_E and N3_E tests. Synthetic data only.
"""
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "fig_F3_transition_structure",
    Path(__file__).resolve().parent.parent / "fig_F3_transition_structure.py",
)
f3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f3)


def test_module_rank_by_top_recurrence_then_recurrence_within():
    nodes = [0, 1, 2, 3, 4, 5]
    labels = [0, 0, 1, 1, 2, 2]
    # module 1 holds the most recurrent state (3), then module 2 (4), then 0
    rec = {0: 0.30, 1: 0.10, 2: 0.20, 3: 0.90, 4: 0.50, 5: 0.40}
    order, ranks = f3.module_order(nodes, labels, rec)
    assert order == [3, 2, 4, 5, 0, 1]
    assert ranks == [0, 0, 1, 1, 2, 2]
    assert f3.module_boundaries(ranks) == [2, 4]


def test_module_order_ties_are_deterministic():
    nodes = [5, 2, 7, 1]
    labels = [1, 1, 0, 0]
    rec = {5: 0.5, 2: 0.5, 7: 0.5, 1: 0.5}
    order, ranks = f3.module_order(nodes, labels, rec)
    # equal top recurrence -> lower label first; equal recurrence -> lower id
    assert order == [1, 7, 2, 5]
    assert ranks == [0, 0, 1, 1]


def test_module_order_rejects_bad_input():
    with pytest.raises(ValueError):
        f3.module_order([0, 1], [0], {0: 1.0, 1: 1.0})
    with pytest.raises(ValueError):
        f3.module_order([0, 0], [0, 1], {0: 1.0})
    with pytest.raises(ValueError):
        f3.module_order([0, 1], [0, 1], {0: 1.0})


def test_load_partition(tmp_path):
    with pytest.raises(FileNotFoundError):
        f3.load_partition(tmp_path / "partitions.json", [0, 1])
    p = tmp_path / "partitions.json"
    p.write_text(json.dumps({"nodes": [0, 1, 3], "labels": [0, 1, 0],
                             "networks": ["Vis", "Vis", "Cont"],
                             "consensus_labels": [0, 1, 0]}))
    nodes, labels = f3.load_partition(p, ["0", "1", "3"])
    assert nodes == [0, 1, 3] and labels == [0, 1, 0]
    with pytest.raises(ValueError):
        f3.load_partition(p, [0, 1, 2])
    p.write_text(json.dumps({"nodes": [0, 1, 3]}))
    with pytest.raises(KeyError):
        f3.load_partition(p, [0, 1, 3])


def test_rule_null_stats_reads_n2_n3_on_E():
    def t(q, m, p, n):
        return {"q_obs": q, "q_null_mean": m, "effect": q - m, "p": p, "n_draws": n}
    summary = {"modular": False,
               "tests": {"N1_M": t(0.9, 0.1, 0.5, 10),
                         "N2_E": t(0.4, 0.2, 0.001, 1000),
                         "N3_E": t(0.4, 0.1, 0.002, 500)}}
    out = f3.rule_null_stats(summary)
    assert out["modular"] is False
    assert set(out) == {"modular", "N2_E", "N3_E"}
    assert out["N2_E"]["effect"] == pytest.approx(0.2)
    assert out["N3_E"]["n_draws"] == 500
