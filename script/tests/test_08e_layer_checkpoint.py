"""Tests for 08e per-layer checkpoints.

08e saves each finished layer to its own file so that a task stopped by its
time limit can be resubmitted and continue from the next layer. These tests
pin:
  (a) a saved layer is reused only when its fingerprint and layer index match,
      and a truncated file is recomputed rather than trusted;
  (b) an interrupted run followed by a resumed run gives the same results as
      one uninterrupted run, and the resumed run recomputes only the layers
      that were not saved;
  (c) ``resume=False`` (``--fresh``) recomputes every layer;
  (d) skipped layers are not saved, and the files are removed after the run.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))
_MODULE_PATH = _SCRIPT_DIR / "08e_transformer_cross_stim_aggregate.py"
_spec = importlib.util.spec_from_file_location("_module_08e_ckpt", _MODULE_PATH)
m08e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m08e)

STIM, MODEL = "movie10", "llama-3.2-3b"


def _layer_result(layer_idx):
    """A result dict shaped like _fit_eval_one_layer's, with int keys inside."""
    return {
        "balanced_accuracy": 0.05 + layer_idx / 1000,
        "chance_level": 0.0345,
        "normalized_effect_size": 0.01 * layer_idx,
        "p_perm": 0.001,
        "per_state": {3: {"recall": 0.1}, 8: {"recall": 0.2}},
    }


class _Interrupt(Exception):
    pass


def _compute(calls, stop_at=None, skip=()):
    def compute_layer(layer_idx):
        if stop_at is not None and layer_idx == stop_at:
            raise _Interrupt
        calls.append(layer_idx)
        if layer_idx in skip:
            return None
        return _layer_result(layer_idx)
    return compute_layer


def _run(out_dir, compute_layer, fingerprint="fp-a", resume=True, n=6):
    return m08e.run_layers(
        range(n), compute_layer, str(out_dir), STIM, MODEL, fingerprint,
        resume=resume,
    )


def test_saved_layer_reused_only_with_matching_fingerprint_and_layer(tmp_path):
    path = m08e.partial_layer_path(str(tmp_path), STIM, MODEL, 4)
    m08e.save_partial_layer(path, "fp-a", 4, {"balanced_accuracy": 0.1})
    assert m08e.load_partial_layer(path, "fp-a", 4) == {"balanced_accuracy": 0.1}
    assert m08e.load_partial_layer(path, "fp-b", 4) is None
    assert m08e.load_partial_layer(path, "fp-a", 5) is None
    missing = m08e.partial_layer_path(str(tmp_path), STIM, MODEL, 9)
    assert m08e.load_partial_layer(missing, "fp-a", 9) is None


def test_truncated_layer_file_is_recomputed(tmp_path):
    path = m08e.partial_layer_path(str(tmp_path), STIM, MODEL, 0)
    m08e.save_partial_layer(path, "fp-a", 0, _layer_result(0))
    text = Path(path).read_text()
    Path(path).write_text(text[: len(text) // 2])
    assert m08e.load_partial_layer(path, "fp-a", 0) is None
    calls = []
    results = _run(tmp_path, _compute(calls), n=1)
    assert calls == [0]
    assert results[0]["balanced_accuracy"] == _layer_result(0)["balanced_accuracy"]


def test_resumed_run_matches_uninterrupted_run(tmp_path):
    full_dir, cut_dir = tmp_path / "full", tmp_path / "cut"
    full = _run(full_dir, _compute([]))

    with pytest.raises(_Interrupt):
        _run(cut_dir, _compute([], stop_at=3))
    calls = []
    resumed = _run(cut_dir, _compute(calls))

    assert calls == [3, 4, 5]
    assert resumed == full
    assert json.dumps(resumed, sort_keys=True) == json.dumps(full, sort_keys=True)


def test_changed_fingerprint_recomputes_every_layer(tmp_path):
    _run(tmp_path, _compute([]), fingerprint="fp-a")
    calls = []
    _run(tmp_path, _compute(calls), fingerprint="fp-b")
    assert calls == list(range(6))


def test_fresh_recomputes_every_layer(tmp_path):
    _run(tmp_path, _compute([]))
    calls = []
    _run(tmp_path, _compute(calls), resume=False)
    assert calls == list(range(6))


def test_skipped_layers_are_not_saved_and_retried(tmp_path):
    first = _run(tmp_path, _compute([], skip={2}))
    assert 2 not in first
    assert not Path(m08e.partial_layer_path(str(tmp_path), STIM, MODEL, 2)).exists()
    calls = []
    second = _run(tmp_path, _compute(calls))
    assert calls == [2]
    assert sorted(second) == list(range(6))


def test_clear_partial_layers_removes_the_folder(tmp_path):
    _run(tmp_path, _compute([]))
    part_dir = tmp_path / m08e.PARTIAL_LAYER_DIRNAME
    assert len(list(part_dir.iterdir())) == 6
    m08e.clear_partial_layers(str(tmp_path))
    assert not part_dir.exists()
    m08e.clear_partial_layers(str(tmp_path))  # no folder: no error


def test_fingerprint_depends_on_every_field():
    base = {"lag": 3, "n_permutations": 1000, "test_run_ids": ["a", "b"]}
    fp = m08e.layer_fingerprint(base)
    assert fp == m08e.layer_fingerprint(dict(reversed(list(base.items()))))
    for key, value in [("lag", 4), ("n_permutations", 10), ("test_run_ids", ["a"])]:
        assert m08e.layer_fingerprint({**base, key: value}) != fp


def test_fresh_flag_parses(monkeypatch):
    argv = ["08e", "--sub_id", "sub-01", "--stimulus", STIM, "--model", MODEL]
    monkeypatch.setattr(sys, "argv", argv)
    assert m08e.parse_args().fresh is False
    monkeypatch.setattr(sys, "argv", argv + ["--fresh"])
    assert m08e.parse_args().fresh is True
