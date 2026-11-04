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
import os
import sys
from pathlib import Path

import numpy as np
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


def test_clear_partial_layers_removes_the_folder_and_keeps_the_outputs(tmp_path):
    _run(tmp_path, _compute([]))
    part_dir = tmp_path / m08e.PARTIAL_LAYER_DIRNAME
    assert len(list(part_dir.iterdir())) == 6
    outputs = [tmp_path / f"D3a_transfer_{STIM}_{MODEL}.json", tmp_path / "pca_info.json"]
    for p in outputs:
        p.write_text("{}")
    m08e.clear_partial_layers(str(tmp_path))
    assert not part_dir.exists()
    assert all(p.exists() for p in outputs)
    m08e.clear_partial_layers(str(tmp_path))  # no folder: no error


def test_clear_partial_layers_logs_instead_of_raising(tmp_path, monkeypatch, caplog):
    _run(tmp_path, _compute([]), n=1)

    def failing_rmtree(path, *a, **k):
        raise OSError(39, "Directory not empty", str(path))

    monkeypatch.setattr(m08e.shutil, "rmtree", failing_rmtree)
    with caplog.at_level("WARNING"):
        m08e.clear_partial_layers(str(tmp_path))
    assert "Could not remove" in caplog.text


@pytest.mark.parametrize("make_file, reason", [
    (lambda p: m08e.save_partial_layer(p, "fp-old", 0, {"x": 1}), "fingerprint differs"),
    (lambda p: m08e.save_partial_layer(p, "fp-a", 5, {"x": 1}), "for layer 5"),
    (lambda p: Path(p).write_text('{"fingerprint": "fp-a", "la'), "unreadable"),
])
def test_rejected_saved_layer_is_logged_with_reason(tmp_path, caplog, make_file, reason):
    path = m08e.partial_layer_path(str(tmp_path), STIM, MODEL, 0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    make_file(path)
    with caplog.at_level("WARNING"):
        assert m08e.load_partial_layer(path, "fp-a", 0) is None
    assert reason in caplog.text


def test_missing_saved_layer_is_not_logged(tmp_path, caplog):
    path = m08e.partial_layer_path(str(tmp_path), STIM, MODEL, 0)
    with caplog.at_level("WARNING"):
        assert m08e.load_partial_layer(path, "fp-a", 0) is None
    assert caplog.text == ""


def test_run_layers_reports_reused_layers(tmp_path):
    with pytest.raises(_Interrupt):
        _run(tmp_path, _compute([], stop_at=3))
    reused = []
    m08e.run_layers(range(6), _compute([]), str(tmp_path), STIM, MODEL, "fp-a",
                    reused=reused)
    assert reused == [0, 1, 2]


def test_code_identity_changes_with_the_source(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_text("x = 1\n")
    b.write_text("y = 2\n")
    before = m08e.code_identity((a, b))
    assert before == m08e.code_identity((a, b))
    b.write_text("y = 3\n")
    assert m08e.code_identity((a, b)) != before
    assert {p.name for p in m08e.CODE_IDENTITY_FILES} == {
        "08e_transformer_cross_stim_aggregate.py", "transformer_analysis.py",
        "stats.py", "transformer_io.py",
    }
    assert all(p.is_file() for p in m08e.CODE_IDENTITY_FILES)


def test_code_hash_is_taken_at_import_not_when_main_runs(monkeypatch):
    """An edit made while a job loads data must not be recorded as the
    code that job runs."""
    assert m08e.CODE_SHA256 == m08e.code_identity()
    import ast
    import inspect
    src = inspect.getsource(m08e.main)
    assert "code_identity(" not in src
    assert "CODE_SHA256" in src
    module = ast.parse(Path(m08e.__file__).read_text())
    assigned = [t.id for n in module.body if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name)]
    assert "CODE_SHA256" in assigned


def test_file_stat_identity_tracks_size_mtime_and_presence(tmp_path):
    f = tmp_path / "layer_00" / "run_raw.npy"
    f.parent.mkdir()
    missing = m08e.file_stat_identity([str(f)], str(tmp_path))
    f.write_bytes(b"abc")
    present = m08e.file_stat_identity([str(f)], str(tmp_path))
    assert present != missing
    import os
    st = os.stat(f)
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert m08e.file_stat_identity([str(f)], str(tmp_path)) != present


def test_file_stat_identity_keys_an_annexed_file_on_its_link_target(tmp_path):
    """A git-annex file is a symlink whose target names the content hash.
    Re-fetching the object (new mtime) keeps the identity; a new key or a
    dropped object changes it."""
    import os
    objects = tmp_path / ".git" / "annex" / "objects"
    obj_a = objects / "MD5E-s3--aaa.npy" / "MD5E-s3--aaa.npy"
    obj_a.parent.mkdir(parents=True)
    obj_a.write_bytes(b"abc")
    link = tmp_path / "layer_00" / "run_raw.npy"
    link.parent.mkdir()
    link.symlink_to(os.path.relpath(obj_a, link.parent))
    first = m08e.file_stat_identity([str(link)], str(tmp_path))
    st = os.stat(obj_a)
    os.utime(obj_a, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert m08e.file_stat_identity([str(link)], str(tmp_path)) == first

    obj_b = objects / "MD5E-s3--bbb.npy" / "MD5E-s3--bbb.npy"
    obj_b.parent.mkdir(parents=True)
    obj_b.write_bytes(b"abd")
    link.unlink()
    link.symlink_to(os.path.relpath(obj_b, link.parent))
    assert m08e.file_stat_identity([str(link)], str(tmp_path)) != first

    obj_b.unlink()                                  # annex drop
    dropped = m08e.file_stat_identity([str(link)], str(tmp_path))
    link.unlink()
    assert dropped == m08e.file_stat_identity([str(link)], str(tmp_path))  # = missing
    assert dropped != first


def test_feature_file_paths_use_the_stream_pca_features_layout(tmp_path):
    """The fingerprint identifies the files stream_pca_features reads."""
    from utils.transformer_analysis import transformer_feature_path
    f = Path(transformer_feature_path(str(tmp_path), STIM, MODEL, 0, "wolf01"))
    assert f.relative_to(tmp_path).as_posix() == (
        f"output/08c_transformer_features/{STIM}/{MODEL}/layer_00/wolf01_raw.npy")
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")
    paths = m08e.feature_file_paths(str(tmp_path), STIM, MODEL, ["wolf01"])
    assert str(f) in paths
    assert len(paths) == m08e.MODEL_REGISTRY[MODEL]["n_layers"]
    with pytest.raises(FileNotFoundError):
        m08e.feature_file_paths(str(tmp_path), STIM, MODEL, ["life01"])


def test_stream_pca_features_reads_through_the_shared_path_helper(tmp_path, monkeypatch):
    """stream_pca_features opens exactly the paths transformer_feature_path
    gives, so a layout change in one place moves both."""
    import utils.transformer_analysis as ta
    seen = []
    real = ta.transformer_feature_path

    def recording(*a, **k):
        p = real(*a, **k)
        seen.append(p)
        return p

    monkeypatch.setattr(ta, "transformer_feature_path", recording)
    model = "dinov2-large"
    n_layers = m08e.MODEL_REGISTRY[model]["n_layers"]
    runs = {"wolf01": 6, "bourne01": 5}
    rng = np.random.default_rng(0)
    for run, n in runs.items():
        for layer in range(n_layers):
            f = Path(real(str(tmp_path), STIM, model, layer, run))
            f.parent.mkdir(parents=True, exist_ok=True)
            np.save(f, rng.standard_normal((n, 3)).astype(np.float32))
    feats, _, _, eff, dropped = ta.stream_pca_features(
        STIM, model, sorted(runs), runs, str(tmp_path),
        train_run_ids={"wolf01"},
    )
    assert dropped == {} and eff == runs
    # Pass 1 probes every file and pass 2 loads every file, both through
    # the helper: each path is asked for exactly twice.
    expected = {real(str(tmp_path), STIM, model, layer, run)
                for run in runs for layer in range(n_layers)}
    assert set(seen) == expected
    assert all(seen.count(p) == 2 for p in expected)
    assert sorted(feats) == list(range(n_layers))
    assert all(feats[layer][run].shape[0] == n
               for layer in feats for run, n in runs.items())


def test_feature_path_puts_the_extraction_suffix_in_the_folder_name(tmp_path):
    """08d --features_subdir_suffix runs (e.g. sweep_w3) read their own
    feature folder, never the main 08c features."""
    import utils.transformer_analysis as ta
    main = ta.transformer_feature_path("/s", STIM, MODEL, 3, "wolf01")
    sweep = ta.transformer_feature_path("/s", STIM, MODEL, 3, "wolf01",
                                        extraction_subdir_suffix="_sweep_w3")
    assert main == os.path.join("/s", "output", "08c_transformer_features",
                                STIM, MODEL, "layer_03", "wolf01_raw.npy")
    assert sweep == os.path.join("/s", "output", "08c_transformer_features_sweep_w3",
                                 STIM, MODEL, "layer_03", "wolf01_raw.npy")
    # stream_pca_features with the suffix finds files only in the suffixed folder.
    model = "dinov2-large"
    for layer in range(m08e.MODEL_REGISTRY[model]["n_layers"]):
        f = Path(ta.transformer_feature_path(str(tmp_path), STIM, model, layer,
                                             "wolf01", "_sweep_w3"))
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, np.arange(15, dtype=np.float32).reshape(5, 3) ** 2)
    kw = dict(train_run_ids={"wolf01"})
    _, _, _, eff, _ = ta.stream_pca_features(
        STIM, model, ["wolf01"], {"wolf01": 5}, str(tmp_path),
        extraction_subdir_suffix="_sweep_w3", **kw)
    assert eff == {"wolf01": 5}
    _, _, _, eff, dropped = ta.stream_pca_features(
        STIM, model, ["wolf01"], {"wolf01": 5}, str(tmp_path), **kw)
    assert eff == {} and "wolf01" in dropped


def test_write_json_atomic_leaves_old_file_when_dump_fails(tmp_path):
    path = tmp_path / "D3a.json"
    m08e.write_json_atomic(str(path), {"ok": 1})
    with pytest.raises(TypeError):
        m08e.write_json_atomic(str(path), {"bad": object()})
    assert json.loads(path.read_text()) == {"ok": 1}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["D3a.json"]   # no temp left


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


def test_main_resumes_through_run_layers_and_clears_after_writing():
    """main() routes every layer through run_layers with resume off under
    --fresh or --force, and removes the layer files only after the last
    output is written."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(m08e.main))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    names = [getattr(c.func, "id", None) for c in calls]
    assert names.count("run_layers") == 1
    run_call = calls[names.index("run_layers")]
    resume = {k.arg: k.value for k in run_call.keywords}["resume"]
    assert ast.unparse(resume) == "not (args.fresh or args.force)"
    assert "_fit_eval_one_layer" in ast.unparse(tree)

    src = inspect.getsource(m08e.main)
    clear_at = src.index("clear_partial_layers(out_dir)")
    assert clear_at > src.rindex("write_json_atomic(")
