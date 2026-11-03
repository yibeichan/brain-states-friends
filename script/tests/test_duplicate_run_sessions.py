"""Runs acquired twice resolve to the same copy in every loader."""
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
from utils.common import (  # noqa: E402
    DUPLICATE_RUN_SESSIONS, pick_run_session, session_from_path,
)
from utils.state_fc import load_matched_data  # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), SCRIPT_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ts_name(sub, ses, task):
    return f"{sub}_{ses}_task-{task}_space-fsLR_den-91k_parcel_avg.npy"


def test_pinned_sessions():
    assert DUPLICATE_RUN_SESSIONS == {
        ("sub-02", "s05e09a"): "ses-037",
        ("sub-06", "s02e04b"): "ses-006",
        ("sub-04", "s02e02a"): "ses-013",
        ("sub-04", "s02e06a"): "ses-014",
        ("sub-06", "s05e03a"): "ses-028",
    }


def test_session_from_path():
    assert session_from_path("/x/sub-02_ses-036_task-s05e09a_bold.nii.gz") == "ses-036"
    assert session_from_path("sub-02/ses-037/func/a.tsv") == "ses-037"
    assert session_from_path("sub-02_task-s05e09a.npy") is None


def test_single_copy_is_returned():
    assert pick_run_session("sub-01", "s01e01a", ["a/sub-01_ses-001_task-s01e01a.npy"]) \
        == "a/sub-01_ses-001_task-s01e01a.npy"


@pytest.mark.parametrize("order", [0, 1])
def test_pinned_copy_wins_in_any_order(order):
    files = [_ts_name("sub-06", "ses-006", "s02e04b"), _ts_name("sub-06", "ses-041", "s02e04b")]
    if order:
        files.reverse()
    assert pick_run_session("sub-06", "s02e04b", files) == files[order]
    files = [_ts_name("sub-02", "ses-036", "s05e09a"), _ts_name("sub-02", "ses-037", "s05e09a")]
    if order:
        files.reverse()
    assert session_from_path(pick_run_session("sub-02", "s05e09a", files)) == "ses-037"


def test_unlisted_duplicate_raises():
    files = [_ts_name("sub-01", "ses-001", "s01e01a"), _ts_name("sub-01", "ses-002", "s01e01a")]
    with pytest.raises(ValueError, match="none is chosen"):
        pick_run_session("sub-01", "s01e01a", files)


def test_pinned_session_missing_raises():
    with pytest.raises(ValueError, match="expected exactly one copy from ses-006"):
        pick_run_session("sub-06", "s02e04b", [_ts_name("sub-06", "ses-041", "s02e04b")])


def test_state_fc_reads_pinned_copy_and_ignores_other_stimuli(tmp_path):
    ts_dir = tmp_path / "output" / "02_parcel_ts_avg" / "atlas-X" / "sub-06"
    ts_dir.mkdir(parents=True)
    # The copy that must be read holds 1.0, the other copy 2.0. ses-041 sorts
    # last, which is what the loader used to pick.
    np.save(ts_dir / _ts_name("sub-06", "ses-006", "s02e04b"), np.full((5, 3), 1.0))
    np.save(ts_dir / _ts_name("sub-06", "ses-041", "s02e04b"), np.full((5, 3), 2.0))
    # Repeated tasks of other stimuli share the folder and must not raise.
    np.save(ts_dir / _ts_name("sub-06", "ses-050", "life01"), np.zeros((5, 3)))
    np.save(ts_dir / _ts_name("sub-06", "ses-051", "life01"), np.zeros((5, 3)))
    decoded = {"s02e04b": np.zeros(5, dtype=int)}
    parcel_ts, _, n_runs = load_matched_data(
        "sub-06", "atlas-X", decoded, 3, scratch_dir=str(tmp_path))
    assert n_runs == 1
    assert np.all(parcel_ts == 1.0)


@pytest.mark.parametrize("wrong_first", [True, False])
def test_03a_loads_pinned_copy(tmp_path, monkeypatch, wrong_first):
    mod = _load_script("03a_pca4combined_hmm")
    d = tmp_path / "atlas-X" / "sub-06"
    d.mkdir(parents=True)
    np.save(d / _ts_name("sub-06", "ses-041", "s02e04b"), np.full((4, 157), 2.0))
    np.save(d / _ts_name("sub-06", "ses-006", "s02e04b"), np.full((4, 157), 1.0))
    # Fix the listing order so the test does not depend on the filesystem's
    # directory order: either copy may come first.
    real_glob = mod.glob.glob
    monkeypatch.setattr(mod.glob, "glob", lambda pat: sorted(real_glob(pat), key=lambda f: (
        "ses-006" in f) == wrong_first))
    data, path = mod.load_single_run("sub-06", "s02e04b", str(tmp_path), "atlas-X")
    assert session_from_path(path) == "ses-006"
    assert data.shape == (4, 156) and np.all(data == 1.0)


@pytest.mark.parametrize("script", ["05e_temporal_trend_a1", "05e_temporal_trend_a3"])
def test_05e_acquisition_time_uses_pinned_copy(tmp_path, monkeypatch, script):
    monkeypatch.setenv("SCRATCH_DIR", str(tmp_path))  # the scripts require it at import
    mod = _load_script(script)
    out = tmp_path / "output" / "00_get_scan" / "sub-06"
    out.mkdir(parents=True)
    rows = [
        ("s02e04b", "ses-006", "10.1", "func/sub-06_ses-006_task-s02e04b_bold.nii.gz"),
        ("s02e05a", "ses-006", "10.2", "func/sub-06_ses-006_task-s02e05a_bold.nii.gz"),
        ("s02e04b", "ses-041", "687.1", "func/sub-06_ses-041_task-s02e04b_bold.nii.gz"),
        # Acquired twice, but only the ses-028 copy was processed.
        ("s05e03a", "ses-028", "300.0", "func/sub-06_ses-028_task-s05e03a_bold.nii.gz"),
        ("s05e03a", "ses-029", "300.5", "func/sub-06_ses-029_task-s05e03a_bold.nii.gz"),
    ]
    with open(out / "sub-06_run_acquisition_times.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "session_id", "rel_acq_time", "filename"])
        w.writerows(rows)
    monkeypatch.setattr(mod, "SCRATCH_DIR", str(tmp_path))
    acq, sessions = mod.load_acquisition_times("sub-06")
    assert acq == {"s02e04b": 10.1, "s02e05a": 10.2, "s05e03a": 300.0}
    assert sessions == {"s02e04b": "ses-006", "s02e05a": "ses-006", "s05e03a": "ses-028"}


def _confounds(root, sub, ses, task, fd):
    d = root / "cneuromod.processed" / "fmriprep" / "friends" / sub / ses / "func"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{sub}_{ses}_task-{task}_desc-confounds_timeseries.tsv").write_text(
        "framewise_displacement\n" + "\n".join(str(fd) for _ in range(3)) + "\n")


@pytest.mark.parametrize("wrong_first", [True, False])
def test_05e_median_fd_uses_pinned_copy(tmp_path, monkeypatch, wrong_first):
    monkeypatch.setenv("SCRATCH_DIR", str(tmp_path))
    mod = _load_script("05e_temporal_trend_a1")
    _confounds(tmp_path, "sub-06", "ses-006", "s02e04b", 0.1)
    _confounds(tmp_path, "sub-06", "ses-041", "s02e04b", 0.9)
    real_glob = mod.glob.glob
    monkeypatch.setattr(mod.glob, "glob", lambda pat: sorted(real_glob(pat), key=lambda f: (
        "ses-006" in f) == wrong_first))
    assert mod.load_median_fd("sub-06", ["s02e04b"], str(tmp_path)) == \
        {"s02e04b": pytest.approx(0.1)}


def test_05e_median_fd_raises_when_pinned_copy_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRATCH_DIR", str(tmp_path))
    mod = _load_script("05e_temporal_trend_a1")
    _confounds(tmp_path, "sub-06", "ses-041", "s02e04b", 0.9)
    with pytest.raises(ValueError, match="expected exactly one copy from ses-006"):
        mod.load_median_fd("sub-06", ["s02e04b"], str(tmp_path))
