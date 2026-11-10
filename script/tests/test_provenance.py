"""Provenance records: which code and which input files produced an output."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
from utils import provenance  # noqa: E402
from utils.common import pick_run_session  # noqa: E402
from utils.state_fc import load_matched_data  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_logs():
    provenance.reset()
    yield
    provenance.reset()


def _ts_name(sub, ses, task):
    return f"{sub}_{ses}_task-{task}_space-fsLR_den-91k_parcel_avg.npy"


def _write_ts(d, sub, ses, task, n, value):
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / _ts_name(sub, ses, task), np.full((n, 3), value))


def test_pick_records_session_for_pinned_and_unpinned_runs(tmp_path):
    a = tmp_path / _ts_name("sub-06", "ses-006", "s02e04b")
    b = tmp_path / _ts_name("sub-06", "ses-041", "s02e04b")
    c = tmp_path / _ts_name("sub-06", "ses-007", "s02e05a")
    for f in (a, b, c):
        np.save(f, np.zeros(2))
    pick_run_session("sub-06", "s02e04b", [str(b), str(a)])
    pick_run_session("sub-06", "s02e05a", [str(c)])
    rec = provenance.build_record("05f", "sub-06", "atlas-X")
    assert rec["sessions_read"] == {"s02e04b": "ses-006", "s02e05a": "ses-007"}
    assert rec["duplicate_run_sessions_pinned"] == {"s02e04b": "ses-006", "s05e03a": "ses-028"}
    assert rec["inputs"]["run_file:s02e04b"]["name"] == a.name
    assert "sha256" in rec["inputs"]["run_file:s02e04b"]


def test_record_is_per_subject(tmp_path):
    for sub in ("sub-01", "sub-06"):
        f = tmp_path / _ts_name(sub, "ses-001", "s01e01a")
        np.save(f, np.zeros(2))
        pick_run_session(sub, "s01e01a", [str(f)])
    rec = provenance.build_record("05f", "sub-01", "atlas-X")
    assert list(rec["sessions_read"]) == ["s01e01a"]
    assert all(k.startswith("run_file:") for k in rec["inputs"])
    assert len(rec["inputs"]) == 1


def test_annex_key_is_taken_from_the_symlink_target(tmp_path):
    key = "MD5E-s20128--86e7cc3ce5b95676377fc680016ce4c7.npy"
    obj = tmp_path / "objects" / key
    obj.parent.mkdir()
    obj.write_bytes(b"x")
    link = tmp_path / "f.npy"
    os.symlink(obj, link)
    assert provenance.file_identity(link) == {"name": "f.npy", "annex_key": key}


def test_loader_logs_sessions_lengths_and_skips(tmp_path):
    ts_dir = tmp_path / "output" / "02_parcel_ts_avg" / "atlas-X" / "sub-06"
    _write_ts(ts_dir, "sub-06", "ses-006", "s02e04b", 5, 1.0)
    _write_ts(ts_dir, "sub-06", "ses-041", "s02e04b", 5, 2.0)
    _write_ts(ts_dir, "sub-06", "ses-007", "s02e05a", 6, 1.0)
    decoded = {"s02e04b": np.zeros(5, int), "s02e05a": np.zeros(4, int), "s02e06a": np.zeros(3, int)}
    load_matched_data("sub-06", "atlas-X", decoded, 3, scratch_dir=str(tmp_path))
    rec = provenance.build_record("05f", "sub-06", "atlas-X")
    assert rec["sessions_read"]["s02e04b"] == "ses-006"
    assert rec["runs"]["s02e05a"] == {"n_ts": 6, "n_vit": 4, "n_used": 4}
    assert rec["runs"]["s02e06a"] == {"skipped": "no parcel time series"}


def test_write_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRATCH_DIR", str(tmp_path))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    p = provenance.write_provenance(tmp_path / "out", "04rc", "sub-01", "atlas-X",
                                    covers=["b.npy", "a.json"])
    assert p.name == "provenance_04rc.json"
    rec = json.loads(p.read_text())
    assert rec["schema"] == 1 and rec["stage"] == "04rc" and rec["slurm_job_id"] == "123"
    assert rec["covers"] == ["a.json", "b.npy"]
    assert rec["scratch_dir"] == str(tmp_path)
    assert len(rec["code"]["commit"]) == 40
    assert rec["code"]["head_changed"] is False


def test_code_state_flags_untracked_python(tmp_path):
    repo = tmp_path / "r"
    (repo / "script").mkdir(parents=True)
    os.system(f"git -C {repo} init -q && git -C {repo} -c user.email=a@b -c user.name=a "
              f"commit -q --allow-empty -m init")
    assert provenance.code_state(repo)["dirty"] is False
    (repo / "script" / "new_stage.py").write_text("x = 1\n")
    st = provenance.code_state(repo)
    assert st["dirty"] is True and st["untracked_py"] == ["script/new_stage.py"]
