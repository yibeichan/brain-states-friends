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
    assert len(rec["inputs"]["run_file:s02e04b"]["md5"]) == 32


def test_record_is_per_subject(tmp_path):
    for sub in ("sub-01", "sub-06"):
        f = tmp_path / _ts_name(sub, "ses-001", "s01e01a")
        np.save(f, np.zeros(2))
        pick_run_session(sub, "s01e01a", [str(f)])
    rec = provenance.build_record("05f", "sub-01", "atlas-X")
    assert list(rec["sessions_read"]) == ["s01e01a"]
    assert all(k.startswith("run_file:") for k in rec["inputs"])
    assert len(rec["inputs"]) == 1


def test_same_bytes_get_the_same_identity_annexed_or_plain(tmp_path):
    import hashlib
    content = b"\0" * 128
    key = f"MD5E-s128--{hashlib.md5(content).hexdigest()}.npy"
    obj = tmp_path / "objects" / "Xx" / "Yy" / key / key
    obj.parent.mkdir(parents=True)
    obj.write_bytes(content)
    link = tmp_path / "linked.npy"
    os.symlink(obj, link)
    plain = tmp_path / "plain.npy"
    plain.write_bytes(content)
    li, pi = provenance.file_identity(link), provenance.file_identity(plain)
    assert (li["size"], li["md5"]) == (pi["size"], pi["md5"]) == (128, hashlib.md5(content).hexdigest())
    os.remove(obj)                                   # annexed content not present: the key still identifies it
    assert provenance.file_identity(link)["md5"] == li["md5"]


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
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.json").write_text("{}")
    (out / "b.npy").write_bytes(b"b")
    (out / "stage04.pkl").write_bytes(b"not 04rc's")
    p = provenance.write_provenance(out, "04rc", "sub-01", "atlas-X", covers=["b.npy", "a.json"])
    assert p.name == "provenance_04rc.json"
    rec = json.loads(p.read_text())
    assert rec["schema"] == 2 and rec["stage"] == "04rc" and rec["slurm_job_id"] == "123"
    assert sorted(rec["outputs"]) == ["a.json", "b.npy"]
    assert rec["outputs"]["b.npy"] == {"size": 1, "md5": "92eb5ffee6ae2fec3ad71c777531578f"}
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


def test_default_record_lists_only_files_written_by_this_run(tmp_path, monkeypatch):
    out = tmp_path / "out"
    (out / "figures").mkdir(parents=True)
    old = out / "left_from_an_earlier_run.npy"
    old.write_bytes(b"old")
    os.utime(old, (provenance._T0 - 3600, provenance._T0 - 3600))
    (out / "new.npy").write_bytes(b"new")
    (out / "figures" / "f.png").write_bytes(b"png")
    (out / "provenance_05e_a1.json").write_text("{}")           # another stage's record
    rec = json.loads(provenance.write_provenance(out, "05f", "sub-06", "atlas-X").read_text())
    assert sorted(rec["outputs"]) == ["figures/f.png", "new.npy"]


def test_dirty_at_import_stays_dirty_after_the_edit_is_reverted(monkeypatch):
    monkeypatch.setitem(provenance._CODE_AT_IMPORT, "dirty", True)
    monkeypatch.setitem(provenance._CODE_AT_IMPORT, "modified", [" M script/05f_state_fc.py"])
    monkeypatch.setattr(provenance, "code_state", lambda repo_dir=None: {
        "commit": provenance._CODE_AT_IMPORT["commit"], "branch": "main", "dirty": False,
        "modified": [], "untracked_py": []})
    rec = provenance.build_record("05f", "sub-06", "atlas-X")
    assert rec["code"]["dirty"] is True and rec["code"]["dirty_at_import"] is True
    assert rec["code"]["dirty_at_write"] is False
    assert rec["code"]["modified_at_import"] == [" M script/05f_state_fc.py"]


def test_unreadable_output_is_recorded_not_fatal(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    (out / "good.npy").write_bytes(b"g")
    (out / "bad.npy").write_bytes(b"b")
    real = provenance.content_id

    def flaky(path):
        if Path(path).name == "bad.npy":
            raise PermissionError("denied")
        return real(path)
    monkeypatch.setattr(provenance, "content_id", flaky)
    rec = json.loads(provenance.write_provenance(out, "05f", "sub-06", "atlas-X").read_text())
    assert rec["outputs"]["bad.npy"] == {"unreadable": "denied"}
    assert rec["outputs"]["good.npy"]["md5"]


def test_file_vanishing_during_the_walk_is_skipped(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.npy").write_bytes(b"k")
    (out / "gone.npy").write_bytes(b"g")
    real_stat = os.stat

    def stat(path, *a, **kw):
        if str(path).endswith("gone.npy"):
            raise FileNotFoundError(path)
        return real_stat(path, *a, **kw)
    monkeypatch.setattr(provenance.os, "stat", stat)
    rec = json.loads(provenance.write_provenance(out, "05f", "sub-06", "atlas-X").read_text())
    assert list(rec["outputs"]) == ["keep.npy"]
