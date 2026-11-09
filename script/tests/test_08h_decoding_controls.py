"""Tests for the 08h fold and mask helpers.

Pins (a) leave-one-episode-out folds hold out every run of an episode
together and never put an episode's TRs in both train and test; (b) the
opening mask drops the first 33 + lag TRs of a-runs only; (c) the word-TR
indicator is lag-shifted the way build_layer_feature_matrix shifts features;
(d) with every TR eligible, episode folds of single-run episodes equal LORO.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))
_spec = importlib.util.spec_from_file_location(
    "_module_08h", _SCRIPT_DIR / "08h_decoding_controls.py")
m08h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m08h)

from utils.transformer_analysis import (  # noqa: E402
    build_layer_feature_matrix, build_loro_folds,
)

RUNS = ["s01e01a", "s01e01b", "s01e02a", "s01e02b", "s06e24c", "s06e24d"]
LENS = [50, 40, 45, 30, 20, 25]


def _bounds(lens):
    out, s = [], 0
    for n in lens:
        out.append((s, s + n))
        s += n
    return out


def test_episode_key():
    assert [m08h.episode_key(r) for r in RUNS] == [
        "s01e01", "s01e01", "s01e02", "s01e02", "s06e24", "s06e24"]


def test_episode_folds_hold_out_whole_episodes():
    bounds = _bounds(LENS)
    n = sum(LENS)
    rng = np.random.default_rng(0)
    mask = rng.random(n) > 0.3
    folds, keys = m08h.build_episode_folds(RUNS, bounds, mask)
    assert keys == ["s01e01", "s01e02", "s06e24"]
    masked_global = np.where(mask)[0]
    ep_of_tr = np.empty(n, dtype=object)
    for r, (a, b) in zip(RUNS, bounds):
        ep_of_tr[a:b] = m08h.episode_key(r)
    seen = []
    for (train, test), key in zip(folds, keys):
        g_test, g_train = masked_global[test], masked_global[train]
        assert set(ep_of_tr[g_test]) == {key}
        assert key not in set(ep_of_tr[g_train])
        assert len(train) + len(test) == mask.sum()
        seen.extend(test.tolist())
    assert sorted(seen) == list(range(int(mask.sum())))


def test_episode_folds_equal_loro_for_single_run_episodes():
    runs = ["s01e01a", "s01e02a", "s01e03a"]
    bounds = _bounds([10, 12, 9])
    mask = np.ones(31, dtype=bool)
    ep, _ = m08h.build_episode_folds(runs, bounds, mask)
    loro = build_loro_folds(bounds, eligible_mask=mask)
    assert len(ep) == len(loro)
    for (a, b), (c, d) in zip(ep, loro):
        assert np.array_equal(a, c) and np.array_equal(b, d)


def test_opening_mask_a_runs_only():
    bounds = _bounds(LENS)
    keep = m08h.opening_mask(RUNS, bounds, sum(LENS), lag=3, mask_tr=33)
    for r, (a, b) in zip(RUNS, bounds):
        seg = keep[a:b]
        if r.endswith("a"):
            cut = min(36, b - a)
            assert not seg[:cut].any() and seg[cut:].all()
        else:
            assert seg.all()


def test_word_mask_shift_matches_feature_shift():
    x = np.array([1, 0, 1, 1, 0, 0, 1], dtype=bool)
    feats = {"r": x.astype(float)[:, None]}
    states = {"r": np.zeros(len(x), dtype=int)}
    for lag in range(0, 9):
        shifted = build_layer_feature_matrix(feats, ["r"], states, lag=lag)[:, 0] > 0
        assert np.array_equal(m08h.lag_shift_bool(x, lag), shifted)
