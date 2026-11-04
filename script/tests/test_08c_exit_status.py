"""08c exits nonzero when every attempted run fails, and then writes no metadata."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))


def _load_08c(tmp_path, monkeypatch):
    # The script requires these at import.
    for var in ("SCRATCH_DIR", "DATA_DIR", "ALGONAUTS_DIR"):
        monkeypatch.setenv(var, str(tmp_path))
    spec = importlib.util.spec_from_file_location(
        "m08c", SCRIPT_DIR / "08c_transformer_features.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _setup(mod, tmp_path, monkeypatch, process_run_audio):
    runs = [
        {"run_id": "lppFR_run-01", "audio_path": "a.wav", "n_trs": 10, "onset_s": 4.0},
        {"run_id": "lppFR_run-02", "audio_path": "b.wav", "n_trs": 12, "onset_s": 4.0},
    ]
    monkeypatch.setattr(mod, "SCRATCH_DIR", str(tmp_path))
    monkeypatch.setattr(mod, "discover_runs", lambda stimulus, config: list(runs))
    monkeypatch.setattr(mod, "load_model", lambda key, device="cuda": (None, None))
    monkeypatch.setattr(mod, "process_run_audio", process_run_audio)
    monkeypatch.setattr(
        sys, "argv",
        ["08c", "--stimulus", "petitprince_fr", "--model", "w2v-bert-2.0",
         "--device", "cpu", "--audio_cache_dir", str(tmp_path / "cache")],
    )
    return (tmp_path / "output" / "08c_transformer_features" / "petitprince_fr"
            / "w2v-bert-2.0" / "metadata.json")


def test_exits_nonzero_when_every_run_fails(tmp_path, monkeypatch):
    mod = _load_08c(tmp_path, monkeypatch)

    def failing(run_info, *a, **kw):
        raise RuntimeError("Could not load libtorchcodec")

    metadata = _setup(mod, tmp_path, monkeypatch, failing)
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code not in (0, None)
    # No metadata claiming an extraction of zero runs.
    assert not metadata.exists()


def test_exits_nonzero_when_every_run_is_skipped(tmp_path, monkeypatch):
    mod = _load_08c(tmp_path, monkeypatch)
    metadata = _setup(mod, tmp_path, monkeypatch, lambda run_info, *a, **kw: False)
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code not in (0, None)
    assert not metadata.exists()


def test_partial_success_returns_and_writes_metadata(tmp_path, monkeypatch):
    mod = _load_08c(tmp_path, monkeypatch)

    def first_only(run_info, *a, **kw):
        if run_info["run_id"].endswith("01"):
            return True
        raise RuntimeError("boom")

    metadata = _setup(mod, tmp_path, monkeypatch, first_only)
    mod.main()  # returns normally
    meta = json.loads(metadata.read_text())
    assert meta["n_runs"] == 1
    assert meta["run_ids"] == ["lppFR_run-01"]
