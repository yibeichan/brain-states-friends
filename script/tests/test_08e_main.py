"""Tests of 08e main() from argument parsing to the written D3a outputs.

The data loaders, PCA and the per-layer fit are replaced by small stubs, so
main() runs in well under a second while every line of its own wiring runs
for real. These tests pin:
  (a) ``--lag friends_best`` writes to ``<stim>_<model>_lag-best``, builds
      every feature matrix at the lag 08d D1 picks, and puts that lag, the
      code identity and the input identities into the layer fingerprint;
  (b) the D3a JSONs (pooled and per film) record the lag, its D1
      provenance, the fingerprint and which layers were reused;
  (c) the per-layer files are removed after a successful cell, with or
      without ``--per_subset``, and the D3a outputs stay;
  (d) an interrupted cell resumes from its saved layers, and ``--force``
      recomputes every layer;
  (e) a completed best-lag cell whose lag no longer matches 08d D1 raises
      instead of being skipped, and a matching one is skipped.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))
_MODULE_PATH = _SCRIPT_DIR / "08e_transformer_cross_stim_aggregate.py"
_spec = importlib.util.spec_from_file_location("_module_08e_main", _MODULE_PATH)
m08e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m08e)

SUB, PARC, STIM, MODEL = "sub-01", "atlas-4S156Parcels", "movie10", "dinov2-large"
D1_BEST_LAG, D1_BEST_LAYER = 4, 12
N_LAYERS = m08e.MODEL_REGISTRY[MODEL]["n_layers"]

FRIENDS_RUNS = ["s01e01a", "s01e01b", "s01e02a"]
TEST_RUNS = ["wolf01", "figures01", "bourne01", "life01"]


class _Interrupt(Exception):
    pass


def _states(n=30):
    return np.array([0, 1, 2] * (n // 3), dtype=int)


def _write_d1(scratch, lag=D1_BEST_LAG, layer=D1_BEST_LAYER):
    path = Path(m08e.friends_d1_path(str(scratch), PARC, SUB, MODEL))
    path.parent.mkdir(parents=True, exist_ok=True)
    results = {f"lag_{l}": {"0": {"balanced_accuracy": 0.05}} for l in range(9)}
    results["lag_0"]["3"] = {"balanced_accuracy": 0.9}   # excluded
    results[f"lag_{lag}"][str(layer)] = {"balanced_accuracy": 0.2}
    path.write_text(json.dumps({"results": results}))
    return path


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Stub every loader so main() runs end to end on tiny data."""
    monkeypatch.setattr(m08e, "SCRATCH_DIR", str(tmp_path))
    _write_d1(tmp_path)
    rec = {"lags": [], "fields": [], "fit_calls": [], "stop_at": None}

    monkeypatch.setattr(m08e, "_load_friends_decoded",
                        lambda sub, parc: {r: _states() for r in FRIENDS_RUNS})
    monkeypatch.setattr(m08e, "_load_cross_stim_decoded",
                        lambda sub, parc, stim: {r: _states() for r in TEST_RUNS})
    monkeypatch.setattr(m08e, "load_training_split",
                        lambda sub, parc, sd: {"train": set(FRIENDS_RUNS[:2]),
                                               "valid": set(), "test": set()})

    def fake_stream(stim, model, run_ids, n_trs, sd, **kw):
        feats = {L: {r: np.zeros((n_trs[r], 2), np.float32) for r in run_ids}
                 for L in range(N_LAYERS)}
        info = {L: {"K": 2, "explained_variance": 0.95, "n_train_samples": 60}
                for L in range(N_LAYERS)}
        return feats, info, {L: None for L in range(N_LAYERS)}, dict(n_trs), {}

    monkeypatch.setattr(m08e, "stream_pca_features", fake_stream)
    monkeypatch.setattr(m08e, "load_content_eligibility", lambda *a, **k: {
        "content_eligible": [0, 1, 2], "eligibility_source": "05e_a4"})
    monkeypatch.setattr(m08e, "precompute_eligible_null_state_sequences",
                        lambda *a, **k: [])
    monkeypatch.setattr(m08e, "_plot_transfer", lambda *a, **k: None)

    def fake_matrix(layer_runs, run_ids, decoded, lag=0):
        rec["lags"].append(lag)
        return np.zeros((sum(len(decoded[r]) for r in run_ids), 2))

    monkeypatch.setattr(m08e, "build_layer_feature_matrix", fake_matrix)

    def fake_fit(layer_idx, *a, subsets=None, **k):
        if rec["stop_at"] is not None and layer_idx == rec["stop_at"]:
            raise _Interrupt
        rec["fit_calls"].append(layer_idx)
        res = {"balanced_accuracy": 0.5, "chance_level": 0.3333,
               "normalized_effect_size": 0.25, "p_perm": 0.01, "per_state": {}}
        if subsets:
            res["per_subset"] = {name: {"balanced_accuracy": 0.4, "p_perm": 0.02}
                                 for name in subsets}
        return res

    monkeypatch.setattr(m08e, "_fit_eval_one_layer", fake_fit)

    real_fp = m08e.layer_fingerprint

    def recording_fp(fields):
        if isinstance(fields, dict) and "sub_id" in fields:
            rec["fields"].append(fields)
        return real_fp(fields)

    monkeypatch.setattr(m08e, "layer_fingerprint", recording_fp)

    real_ckpt = m08e.check_checkpoint

    def recording_ckpt(out_dir, files, label, force=False):
        rec["checkpoint_dir"] = out_dir
        return real_ckpt(out_dir, files, label, force=force)

    monkeypatch.setattr(m08e, "check_checkpoint", recording_ckpt)

    real_run_layers = m08e.run_layers

    def recording_run_layers(*a, **k):
        rec["run_layers"] = {"out_dir": a[2], "resume": k.get("resume")}
        return real_run_layers(*a, **k)

    monkeypatch.setattr(m08e, "run_layers", recording_run_layers)

    def run(*extra):
        monkeypatch.setattr(sys, "argv", [
            "08e", "--sub_id", SUB, "--stimulus", STIM, "--model", MODEL,
            "--n_permutations", "2", *extra,
        ])
        rec["lags"].clear()
        rec["fit_calls"].clear()
        m08e.main()

    rec["run"] = run
    rec["out_dir"] = (tmp_path / "output" / "08e_transformer_cross_stim_aggregate"
                      / PARC / SUB / f"{STIM}_{MODEL}_lag-best")
    return rec


def _pooled(env):
    return json.loads((env["out_dir"] / f"D3a_transfer_{STIM}_{MODEL}.json").read_text())


def _per_subset(env):
    return json.loads((env["out_dir"] / f"D3a_per_subset_{STIM}_{MODEL}.json").read_text())


def test_friends_best_writes_lag_best_folder_at_the_d1_lag(env):
    env["run"]("--per_subset")
    assert Path(env["checkpoint_dir"]) == env["out_dir"]
    assert Path(env["run_layers"]["out_dir"]) == env["out_dir"]
    assert env["lags"] and set(env["lags"]) == {D1_BEST_LAG}
    fields = env["fields"][-1]
    required = {"lag", "code_sha256", "inputs", "n_permutations", "per_subset",
                "intersection_states", "friends_run_ids", "test_run_ids"}
    assert required <= set(fields)
    assert fields["lag"] == D1_BEST_LAG
    assert fields["code_sha256"] == m08e.code_identity()
    assert {"states_masks_sha256", "pca_train_runs_sha256", "pca_layers_k_n_train",
            "friends_features_stat_sha256", "test_features_stat_sha256",
            } <= set(fields["inputs"])


def test_outputs_record_lag_provenance_and_fingerprint(env):
    env["run"]("--per_subset")
    for payload in (_pooled(env), _per_subset(env)):
        assert payload["lag"] == D1_BEST_LAG
        assert payload["lag_source"] == "friends_best"
        assert payload["friends_best_layer"] == D1_BEST_LAYER
        assert payload["friends_d1_path"].endswith("D1_depth_profile.json")
        assert payload["layer_fingerprint"] == m08e.layer_fingerprint(env["fields"][-1])
        assert payload["code_sha256"] == m08e.code_identity()
        assert payload["reused_layers"] == []
    assert sorted(int(k) for k in _pooled(env)["per_layer"]) == list(range(N_LAYERS))


@pytest.mark.parametrize("extra", [(), ("--per_subset",)])
def test_partial_layers_cleared_and_outputs_kept_on_every_success_path(env, extra):
    env["run"](*extra)
    out = env["out_dir"]
    assert not (out / m08e.PARTIAL_LAYER_DIRNAME).exists()
    assert (out / f"D3a_transfer_{STIM}_{MODEL}.json").exists()
    assert (out / "pca_info.json").exists()
    assert (out / f"D3a_per_subset_{STIM}_{MODEL}.json").exists() == bool(extra)
    assert not list(out.glob("*.tmp.*"))


def test_interrupted_cell_resumes_and_records_reused_layers(env):
    env["stop_at"] = 5
    with pytest.raises(_Interrupt):
        env["run"]()
    assert (env["out_dir"] / m08e.PARTIAL_LAYER_DIRNAME).is_dir()
    env["stop_at"] = None
    env["run"]()
    assert env["run_layers"]["resume"] is True
    assert env["fit_calls"] == list(range(5, N_LAYERS))
    assert _pooled(env)["reused_layers"] == [0, 1, 2, 3, 4]


def _patch_pca_info(monkeypatch, change):
    """Wrap the stubbed PCA so the Friends fit returns ``change(info)``."""
    real = m08e.stream_pca_features

    def patched(*a, **k):
        feats, info, models, eff, dropped = real(*a, **k)
        if k.get("pca_models") is None:
            info = change(info)
        return feats, info, models, eff, dropped

    monkeypatch.setattr(m08e, "stream_pca_features", patched)


def test_resume_on_another_node_reuses_layers_despite_pca_float_jitter(env, monkeypatch):
    """explained_variance differs in the 7th digit between CPU types; the
    saved layers must still be reused."""
    env["stop_at"] = 5
    with pytest.raises(_Interrupt):
        env["run"]()
    _patch_pca_info(monkeypatch, lambda info: {
        L: dict(v, explained_variance=v["explained_variance"] + 1e-7)
        for L, v in info.items()})
    env["stop_at"] = None
    env["run"]()
    assert env["fit_calls"] == list(range(5, N_LAYERS))
    assert _pooled(env)["reused_layers"] == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("field", ["K", "n_train_samples"])
def test_resume_with_a_changed_pca_size_recomputes_every_layer(env, monkeypatch, field):
    env["stop_at"] = 5
    with pytest.raises(_Interrupt):
        env["run"]()
    _patch_pca_info(monkeypatch, lambda info: {
        L: dict(v, **{field: v[field] + (1 if L == 7 else 0)})
        for L, v in info.items()})
    env["stop_at"] = None
    env["run"]()
    assert env["fit_calls"] == list(range(N_LAYERS))
    assert _pooled(env)["reused_layers"] == []


def test_force_recomputes_every_layer(env):
    env["stop_at"] = 5
    with pytest.raises(_Interrupt):
        env["run"]()
    env["stop_at"] = None
    env["run"]("--force")
    assert env["run_layers"]["resume"] is False
    assert env["fit_calls"] == list(range(N_LAYERS))
    assert _pooled(env)["reused_layers"] == []


def test_completed_cell_at_the_d1_lag_is_skipped(env):
    env["run"]()
    env["run"]()
    assert env["fit_calls"] == []          # skipped by the checkpoint


def test_completed_cell_at_an_old_lag_raises_instead_of_skipping(env, tmp_path):
    env["run"]()
    _write_d1(tmp_path, lag=3)            # a D1 rerun moved the peak
    with pytest.raises(RuntimeError) as exc:
        env["run"]()
    msg = str(exc.value)
    assert "lag 4 on disk vs 3 from 08d D1" in msg
    assert "Either side may be stale" in msg and "without FORCE" in msg
    assert "--array=0" in msg and msg.isascii()
    assert env["fit_calls"] == []
    env["run"]("--force")                 # recomputes at the new lag
    assert set(env["lags"]) == {3}
    assert _pooled(env)["lag"] == 3


def test_completed_cell_with_moved_best_layer_warns_and_is_skipped(env, tmp_path, caplog):
    """Only the best layer moved, at the same lag: the D3a result does not
    depend on it, so the cell stays complete."""
    env["run"]()
    _write_d1(tmp_path, layer=D1_BEST_LAYER + 1)
    with caplog.at_level("WARNING"):
        env["run"]()
    assert env["fit_calls"] == []
    assert "friends_best_layer" in caplog.text


@pytest.mark.parametrize("eligible, match", [
    ([0, 2], r"intersection states \[1\] are not content-eligible"),
    ([0, 1, 2, 5], r"content_eligible differs from 05e now \(added \[5\]"),
])
def test_completed_cell_with_changed_eligibility_raises(env, monkeypatch, eligible, match):
    env["run"]()
    monkeypatch.setattr(m08e, "load_content_eligibility", lambda *a, **k: {
        "content_eligible": eligible, "eligibility_source": "05e_a4"})
    with pytest.raises(RuntimeError, match=match):
        env["run"]()
    assert env["fit_calls"] == []


def test_skip_guard_checks_every_output_file(env):
    """A stale per-film file is caught even when the pooled file is current."""
    env["run"]("--per_subset")
    path = env["out_dir"] / f"D3a_per_subset_{STIM}_{MODEL}.json"
    payload = json.loads(path.read_text())
    payload["lag"] = 7
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="D3a_per_subset.*lag 7 on disk"):
        env["run"]("--per_subset")


def test_unreadable_completed_output_raises(env):
    env["run"]()
    path = env["out_dir"] / f"D3a_transfer_{STIM}_{MODEL}.json"
    path.write_text(path.read_text()[:20])
    with pytest.raises(RuntimeError, match="cannot be read"):
        env["run"]()


def test_force_flag_parses_and_help_says_it_implies_fresh(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["08e", "--help"])
    with pytest.raises(SystemExit):
        m08e.parse_args()
    assert "Implies --fresh" in " ".join(capsys.readouterr().out.split())
