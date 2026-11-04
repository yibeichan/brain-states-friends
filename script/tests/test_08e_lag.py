"""Tests for the 08e transfer lag (--lag).

The transfer decoder is fitted on Friends features and applied to the test
stimulus at one brain-to-feature lag. These tests pin:
  (a) the best-lag rule matches 08d's (highest balanced accuracy, lag 0
      excluded, first cell on a tie), including when lag 0 holds the overall
      maximum, and Figure 5A-C use the same rule;
  (b) ``friends_best`` reads the participant's 08d D1 output and fails loudly
      when it is missing instead of falling back to lag 0;
  (c) argument parsing and output-folder naming (lag 0 keeps the original
      folder; other lags never overwrite it);
  (d) every feature matrix main() builds is built at the resolved lag.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))
_MODULE_PATH = _SCRIPT_DIR / "08e_transformer_cross_stim_aggregate.py"
_spec = importlib.util.spec_from_file_location("_module_08e_lag", _MODULE_PATH)
m08e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m08e)


def _d1_results(cells):
    """Build a D1 ``results`` dict from {(lag, layer): balanced_accuracy}."""
    out = {}
    for (lag, layer), acc in cells.items():
        out.setdefault(f"lag_{lag}", {})[str(layer)] = {"balanced_accuracy": acc}
    return out


def test_best_lag_skips_lag0_even_when_it_is_the_maximum():
    results = _d1_results({
        (0, 5): 0.30,   # overall maximum, must be ignored
        (3, 21): 0.12,
        (4, 22): 0.11,
    })
    lag_key, layer, acc = m08e.find_best_lag_layer(
        results, exclude_lags=m08e.PEAK_LAG_EXCLUDE,
    )
    assert (lag_key, layer, acc) == ("lag_3", 21, 0.12)


# 08d rounds balanced accuracy to 4 decimals, so exact ties occur in real
# D1 outputs. Here lag_3 layer 7, lag_3 layer 8 and lag_4 layer 2 tie at the
# maximum: the first in dict order (lag_3, layer 7) must win. A None accuracy
# is skipped, as 08d does.
_TIE_RESULTS = {
    "lag_0": {"1": {"balanced_accuracy": 0.5}},
    "lag_1": {"2": {"balanced_accuracy": 0.10}, "3": {"balanced_accuracy": None}},
    "lag_3": {"7": {"balanced_accuracy": 0.14}, "8": {"balanced_accuracy": 0.14}},
    "lag_4": {"2": {"balanced_accuracy": 0.14}},
    "lag_8": {"0": {"balanced_accuracy": 0.139}},
}


def test_best_lag_rule_takes_first_cell_on_a_tie_and_skips_none():
    assert m08e.find_best_lag_layer(_TIE_RESULTS, m08e.PEAK_LAG_EXCLUDE) == \
        ("lag_3", 7, 0.14)


def test_best_lag_rule_matches_08d():
    """The shared rule agrees with 08d's own function, ties included.

    08d is imported for real; an import failure fails this test rather than
    skipping it, so the parity check cannot silently stop running.
    """
    spec = importlib.util.spec_from_file_location(
        "_module_08d_lag", _SCRIPT_DIR / "08d_transformer_depth.py",
    )
    m08d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m08d)
    assert m08e.PEAK_LAG_EXCLUDE == m08d.PEAK_LAG_EXCLUDE
    for results in (
        _TIE_RESULTS,
        _d1_results({(0, 1): 0.5, (1, 2): 0.10, (3, 7): 0.14, (3, 8): 0.13,
                     (8, 0): 0.139}),
    ):
        assert m08e.find_best_lag_layer(results, m08e.PEAK_LAG_EXCLUDE) == \
            m08d._find_best_lag_layer(results, exclude_lags=m08d.PEAK_LAG_EXCLUDE)


def test_fig5_within_panels_use_the_same_rule(monkeypatch, tmp_path):
    """Figure 5A-C pick the same (lag, layer) as the 08e transfer, ties included."""
    monkeypatch.setenv("SCRATCH_DIR", str(tmp_path))
    spec = importlib.util.spec_from_file_location(
        "_fig_f5_rule", _SCRIPT_DIR / "fig_F5_within_friends.py",
    )
    f5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f5)
    profile = {
        int(k.split("_")[1]): {
            int(li): {"balanced_accuracy": v["balanced_accuracy"], "chance_level": 0.05}
            for li, v in layers.items()
        }
        for k, layers in _TIE_RESULTS.items()
    }
    best_lag, peak_layer, *_ = f5.best_lag_layer(profile, 9)
    assert (best_lag, peak_layer) == (3, 7)


def test_resolve_friends_best_lag_reads_d1(tmp_path):
    path = Path(m08e.friends_d1_path(
        str(tmp_path), "atlas-4S156Parcels", "sub-01", "w2v-bert-2.0",
    ))
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "lags": [0, 1, 2, 3, 4],
        "results": _d1_results({(0, 3): 0.9, (3, 11): 0.10, (4, 12): 0.15}),
    }))
    lag, info = m08e.resolve_friends_best_lag(
        str(tmp_path), "atlas-4S156Parcels", "sub-01", "w2v-bert-2.0",
    )
    assert lag == 4
    assert info["friends_best_layer"] == 12
    assert info["friends_d1_path"] == str(path)


def test_resolve_friends_best_lag_missing_d1_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        m08e.resolve_friends_best_lag(
            str(tmp_path), "atlas-4S156Parcels", "sub-01", "dinov2-large",
        )


def test_resolve_friends_best_lag_only_lag0_raises(tmp_path):
    path = Path(m08e.friends_d1_path(
        str(tmp_path), "atlas-4S156Parcels", "sub-02", "dinov2-large",
    ))
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"results": _d1_results({(0, 3): 0.2})}))
    with pytest.raises(ValueError):
        m08e.resolve_friends_best_lag(
            str(tmp_path), "atlas-4S156Parcels", "sub-02", "dinov2-large",
        )


@pytest.mark.parametrize("value, expected", [
    ("friends_best", "friends_best"), ("0", 0), ("3", 3),
])
def test_parse_lag_accepts(value, expected):
    assert m08e.parse_lag(value) == expected


@pytest.mark.parametrize("value", ["-1", "best", "3.5", ""])
def test_parse_lag_rejects(value):
    with pytest.raises(argparse.ArgumentTypeError):
        m08e.parse_lag(value)


def test_output_subdir_keeps_lag0_folder_and_separates_others():
    assert m08e.output_subdir("movie10", "dinov2-large", 0) == "movie10_dinov2-large"
    assert m08e.output_subdir("movie10", "dinov2-large", "friends_best") == \
        "movie10_dinov2-large_lag-best"
    assert m08e.output_subdir("harrypotter", "llama-3.2-3b", 3) == \
        "harrypotter_llama-3.2-3b_lag-3"


def test_main_builds_every_feature_matrix_at_the_resolved_lag():
    """Each build_layer_feature_matrix call in main() passes lag=lag.

    Without the keyword the helper defaults to lag 0, which is the bug this
    option fixes; this fails if either the training or the test call drops it.
    """
    tree = ast.parse(_MODULE_PATH.read_text())
    main_fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    calls = [
        n for n in ast.walk(main_fn)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "build_layer_feature_matrix"
    ]
    assert len(calls) == 2, "expected one training and one test feature matrix"
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "lag" in kw and isinstance(kw["lag"], ast.Name) and kw["lag"].id == "lag"
