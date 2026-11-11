"""Tests for the 08h fold and mask helpers.

Pins (a) leave-one-episode-out folds hold out every run of an episode
together and never put an episode's TRs in both train and test; (b) the
opening mask drops the first 33 + lag TRs of a-runs only; (c) the word-TR
indicator is lag-shifted the way build_layer_feature_matrix shifts features;
(d) with every TR eligible, episode folds of single-run episodes equal LORO.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

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


def test_select_layers_all_by_default():
    assert m08h.select_layers(None, 4, 2) == [0, 1, 2, 3]


def test_select_layers_best_and_numbers():
    assert m08h.select_layers("best,0", 28, 17) == [0, 17]
    assert m08h.select_layers("best, 17", 28, 17) == [17]      # duplicates collapse


@pytest.mark.parametrize("spec", ["28", "-1", "best,x", ""])
def test_select_layers_refuses_bad_specs(spec):
    with pytest.raises(ValueError):
        m08h.select_layers(spec, 28, 3)


@pytest.mark.parametrize("spec", ["1,,2", "best,", " "])
def test_parse_layers_spec_refuses_empty_items(spec):
    with pytest.raises(ValueError):
        m08h.parse_layers_spec(spec)


def test_folds_without_every_class_are_counted():
    y = np.array([0, 0, 1, 1, 2, 2])
    ok = [(np.array([0, 2, 4]), np.array([1, 3, 5]))]
    bad = [(np.array([0, 1, 2, 3]), np.array([4, 5]))]   # class 2 only in test
    assert m08h.n_folds_without_every_class(y, ok) == 0
    assert m08h.n_folds_without_every_class(y, ok + bad) == 1


# ---------------------------------------------------------------------------
# run_friends end to end on synthetic inputs (loaders stubbed)
# ---------------------------------------------------------------------------

_E2E_RUNS = [f"s01e0{e}{h}" for e in range(1, 5) for h in "ab"]
_T, _K, _NL = 150, 6, 2


def _synthetic():
    rng = np.random.default_rng(0)
    states = {r: np.repeat(rng.integers(0, 3, size=_T // 10), 10) for r in _E2E_RUNS}
    proto = rng.normal(size=(_NL, 3, _K))
    feats = {L: {r: (proto[L][states[r]] * (0.4 + 0.4 * L)
                     + rng.normal(size=(_T, _K))).astype(np.float32)
                 for r in _E2E_RUNS} for L in range(_NL)}
    return states, feats


@pytest.fixture
def e2e(tmp_path, monkeypatch):
    """Stub run_friends' loaders; returns run(cond, n_perm, layers) -> payload."""
    states, feats = _synthetic()
    d1_path = tmp_path / "D1_depth_profile.json"

    def write_d1(best_lag):
        res = {f"lag_{g}": {str(L): {"balanced_accuracy": 0.5 + 0.01 * L
                                     + (0.1 if g == best_lag else 0.0), "p_perm": 0.01}
                            for L in range(_NL)} for g in range(4)}
        d1_path.write_text(json.dumps({"results": res}))

    write_d1(1)
    pca_info = {L: {"K": _K, "n_train_samples": _T * len(_E2E_RUNS)} for L in range(_NL)}
    monkeypatch.setattr(m08h, "SCRATCH_DIR", str(tmp_path))
    monkeypatch.setattr(m08h, "friends_d1_path", lambda *a, **k: str(d1_path))
    monkeypatch.setattr(m08h, "load_content_eligibility",
                        lambda *a, **k: {"content_eligible": [0, 1, 2]})
    monkeypatch.setattr(m08h, "d1_eligibility_problems", lambda *a, **k: [])
    monkeypatch.setattr(m08h.d08, "_load_decoded_states_friends",
                        lambda sub, parc: ({r: states[r].copy() for r in _E2E_RUNS}, None))
    monkeypatch.setattr(m08h, "load_training_split", lambda *a, **k: {"train": _E2E_RUNS})
    monkeypatch.setattr(m08h.d08, "_pca_cache_dir", lambda *a, **k: str(tmp_path / "pub"))
    monkeypatch.setattr(m08h, "mirror_tree", lambda src, dst: None)
    monkeypatch.setattr(m08h.d08, "_pca_split_hash_and_path", lambda *a, **k: ("h", "p"))
    monkeypatch.setattr(m08h, "load_or_fit_pca_cache", lambda *a, **k: (
        feats, pca_info, None, {r: _T for r in _E2E_RUNS}, {}))
    monkeypatch.setattr(m08h.e08, "feature_file_paths", lambda *a, **k: [])

    def run(cond, n_perm, layers=None, out="out"):
        out_dir = tmp_path / out
        out_dir.mkdir(exist_ok=True)
        args = argparse.Namespace(sub_id="sub-99", model="dinov2-large", condition=cond,
                                  n_permutations=n_perm, n_jobs=1, layers=layers,
                                  vt="0.95", out_root=str(tmp_path))
        m08h.run_friends(args, "P", str(out_dir))
        return json.loads((out_dir / f"{cond}.json").read_text())

    run.write_d1 = write_d1
    run.out_dir = tmp_path / "out"
    return run


def test_partial_from_other_permutation_count_is_recomputed(e2e):
    e2e("loeo", 5, layers="0")
    full = e2e("loeo", 12)
    assert {v["n_permutations"] for v in full["per_layer"].values()} == {12}


def test_partial_from_other_lag_is_recomputed(e2e):
    def accs(d):
        return {k: (v["balanced_accuracy"], v.get("p_perm")) for k, v in d["per_layer"].items()}
    first = e2e("opening", 5)
    e2e.write_d1(best_lag=2)
    rerun = e2e("opening", 5)
    fresh = e2e("opening", 5, out="fresh")
    assert rerun["best_lag"] == 2 and rerun["opening_cutoff_trs"] == 35
    assert accs(first) != accs(fresh)          # the lag changes the result...
    assert accs(rerun) == accs(fresh)          # ...and the rerun is not the stale one


def test_partial_without_fingerprint_is_recomputed(e2e):
    e2e("loeo", 5)
    p = e2e.out_dir / "partials" / "loeo_layer00.json"
    legacy = json.loads(p.read_text())
    legacy.pop("fingerprint")
    legacy["balanced_accuracy"] = -1.0
    p.write_text(json.dumps(legacy))
    again = e2e("loeo", 5)
    assert again["per_layer"]["0"]["balanced_accuracy"] != -1.0
    assert "fingerprint" in json.loads(p.read_text())


def test_final_json_includes_layers_another_job_finished(e2e):
    one = e2e("loeo", 5, layers="0")
    assert list(one["per_layer"]) == ["0"]
    assert not one["all_layers_evaluated"]
    assert "p_fdr" not in one["per_layer"]["0"]
    assert one["fdr_family"].startswith("none")
    both = e2e("loeo", 5, layers="1")
    assert list(both["per_layer"]) == ["0", "1"]
    assert both["all_layers_evaluated"]
    assert all("p_fdr" in v for v in both["per_layer"].values())
    assert both["fdr_family"] == "layers at the best lag"


def test_timing_delta_is_against_the_timing_only_decoder(e2e):
    out = e2e("timing", 20)
    t_acc = out["timing_only"]["balanced_accuracy"]
    for v in out["per_layer"].values():
        assert v["delta_vs_timing"] == pytest.approx(v["balanced_accuracy"] - t_acc, abs=2e-4)
        assert 0 <= v["p_delta"] <= 1
        assert v["n_permutations"] == 20


def test_main_refuses_layers_for_the_gate(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(m08h, "run_friends", lambda *a: called.append(a))
    monkeypatch.setattr(sys, "argv", [
        "08h", "--part", "friends", "--sub_id", "sub-01", "--model", "dinov2-large",
        "--condition", "gate", "--layers", "0", "--out_root", str(tmp_path / "side")])
    with pytest.raises(SystemExit, match="gate"):
        m08h.main()
    assert not called and not (tmp_path / "side").exists()


def test_word_tr_mask_reads_rows_from_the_run_start(tmp_path, monkeypatch):
    raw = np.zeros((9, 3), dtype=np.float32)
    raw[[1, 2, 5], 0] = 1.0                       # words at TRs 1, 2, 5
    raw[7:, 1] = 1.0                              # trailing rows beyond the run
    f = tmp_path / "feat.npy"
    np.save(f, raw)
    seen = []
    monkeypatch.setattr(m08h, "transformer_feature_path",
                        lambda scratch, stim, model, layer, stem: seen.append(stem) or str(f))
    decoded = {"r1": np.zeros(7, dtype=int)}
    mask, frac = m08h.word_tr_mask("movie10", "llama-3.2-3b", ["r1"], decoded, lag=2,
                                   feature_key_fn=lambda r: f"key-{r}")
    has = np.array([0, 1, 1, 0, 0, 1, 0], dtype=bool)
    assert np.array_equal(mask, m08h.lag_shift_bool(has, 2))
    assert frac["r1"] == pytest.approx(3 / 7)
    assert seen == ["key-r1"]


def test_run_friends_refuses_a_fold_missing_a_class(e2e, monkeypatch):
    states, _ = _synthetic()
    states["s01e01a"][:20] = 3                     # class 3 only in episode s01e01
    monkeypatch.setattr(m08h.d08, "_load_decoded_states_friends",
                        lambda sub, parc: ({r: states[r].copy() for r in _E2E_RUNS}, None))
    monkeypatch.setattr(m08h, "load_content_eligibility",
                        lambda *a, **k: {"content_eligible": [0, 1, 2, 3]})
    with pytest.raises(SystemExit, match="lack a class"):
        e2e("loeo", 5)


def test_partial_from_other_code_is_recomputed(e2e, monkeypatch):
    e2e("loeo", 5)
    p = e2e.out_dir / "partials" / "loeo_layer00.json"
    doctored = json.loads(p.read_text())
    doctored["balanced_accuracy"] = -1.0
    p.write_text(json.dumps(doctored))
    monkeypatch.setattr(m08h, "CODE_SHA256", "other-code")
    again = e2e("loeo", 5)
    assert again["per_layer"]["0"]["balanced_accuracy"] != -1.0


def test_a_layer_left_uncomputed_is_an_error(e2e, monkeypatch):
    monkeypatch.setattr(m08h, "_friends_layer", lambda li, *a, **k: (li, None))
    with pytest.raises(RuntimeError, match="no partial matching"):
        e2e("loeo", 5)
    assert not (e2e.out_dir / "loeo.json").exists()


def test_main_checks_layers_before_loading_data(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(m08h, "run_friends", lambda *a: called.append(a))
    monkeypatch.setattr(sys, "argv", [
        "08h", "--part", "friends", "--sub_id", "sub-01", "--model", "dinov2-large",
        "--condition", "loeo", "--layers", "best,x", "--out_root", str(tmp_path / "side")])
    with pytest.raises(SystemExit, match="not a layer number"):
        m08h.main()
    assert not called
