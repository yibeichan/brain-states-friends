"""promote.py: only declared, checked changes enter the output dataset."""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from utils import promote  # noqa: E402
from utils.promote import compare_files, glob_to_regex  # noqa: E402

MAIN = subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "origin/main"],
                      capture_output=True, text=True).stdout.strip()


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.email=a@b", "-c", "user.name=a", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


# ----------------------------------------------------------------- globs
@pytest.mark.parametrize("pat,rel,ok", [
    ("05f/*/sub-06/**", "05f/atlas/sub-06/vt0.95/a.npy", True),
    ("05f/*/sub-06/**", "05f/atlas/sub-01/vt0.95/a.npy", False),
    ("05f/*/sub-06/*", "05f/atlas/sub-06/vt0.95/a.npy", False),   # * stays in one segment
    ("**/metadata.json", "metadata.json", True),
    ("**/metadata.json", "a/b/metadata.json", True),
    ("a/**/b.npy", "a/b.npy", True),
    ("a/**/b.npy", "a/x/y/b.npy", True),
    ("a/**/b.npy", "a/xb.npy", False),                         # ** spans whole segments only
    ("a/?.npy", "a/x.npy", True),
    ("a/?.npy", "a/xy.npy", False),
    ("a/*.npy", "a/b/c.npy", False),
])
def test_glob(pat, rel, ok):
    assert bool(glob_to_regex(pat).match(rel)) is ok


# ------------------------------------------------------------ comparison
def _npy(p, a):
    np.save(p, a)
    return p


def test_arrays_within_tolerance_nan_masks_and_dtypes(tmp_path):
    a = _npy(tmp_path / "a.npy", np.array([1.0, np.nan, 3.0]))
    assert compare_files(a, _npy(tmp_path / "b.npy", np.array([1.0 + 1e-14, np.nan, 3.0]))).equal
    assert not compare_files(a, _npy(tmp_path / "c.npy", np.array([1.001, np.nan, 3.0]))).equal
    assert not compare_files(a, _npy(tmp_path / "d.npy", np.array([1.0, 2.0, 3.0]))).equal
    i = _npy(tmp_path / "i.npy", np.array([1, 2]))
    assert not compare_files(i, _npy(tmp_path / "j.npy", np.array([1, 3]))).equal
    assert not compare_files(i, _npy(tmp_path / "k.npy", np.array([True, True]))).equal
    f = _npy(tmp_path / "f.npy", np.array([1.0, 1.0]))
    assert not compare_files(f, _npy(tmp_path / "g.npy", np.array([True, True]))).equal
    c = compare_files(a, tmp_path / "c.npy")
    assert c.max_diff == pytest.approx(0.001)


def _json(p, obj):
    p.write_text(json.dumps(obj))
    return p


def test_json_bool_is_not_a_number_and_keys_must_match(tmp_path):
    a = _json(tmp_path / "a.json", {"x": 1, "y": [1.0, 2.0], "flag": True})
    assert compare_files(a, _json(tmp_path / "b.json", {"flag": True, "y": [1.0, 2.0 + 1e-13], "x": 1})).equal
    assert not compare_files(a, _json(tmp_path / "c.json", {"x": 1, "y": [1.0, 2.0], "flag": 1})).equal
    assert not compare_files(a, _json(tmp_path / "d.json", {"x": 1, "y": [2.0, 1.0], "flag": True})).equal
    assert not compare_files(a, _json(tmp_path / "e.json", {"x": 1, "flag": True})).equal
    t = _json(tmp_path / "t.json", {"x": 1, "timestamp": "now"})
    assert compare_files(t, _json(tmp_path / "u.json", {"x": 1, "timestamp": "later"}),
                         ignore_keys=["timestamp"]).equal


def test_tables_compare_by_cell(tmp_path):
    (tmp_path / "a.csv").write_text("state,rho,label\n0,0.5,x\n1,nan,y\n")
    (tmp_path / "b.csv").write_text("state,rho,label\n0,0.5000000000001,x\n1,nan,y\n")
    (tmp_path / "c.csv").write_text("state,rho,label\n0,0.5,z\n1,nan,y\n")
    assert compare_files(tmp_path / "a.csv", tmp_path / "b.csv").equal
    assert not compare_files(tmp_path / "a.csv", tmp_path / "c.csv").equal


# ------------------------------------------------------------ end to end
RV = "05f_state_fc/atlas-X/sub-06/vt0.95/fc_similarity_corr_rv.npy"
OTHER = "05f_state_fc/atlas-X/sub-06/vt0.95/state_delta_fc.npy"
FLAGS = "05e_temporal_trend_a4/atlas-X/sub-06/vt0.95/state_flags.csv"
REC = "05f_state_fc/atlas-X/sub-06/vt0.95/provenance_05f.json"


def _record(sessions=None, commit=MAIN):
    return {"stage": "05f", "sub_id": "sub-06", "parcellation": "atlas-X",
            "code": {"commit": commit, "dirty": False, "head_changed": False},
            "sessions_read": sessions or {"s02e04b": "ses-006"},
            "inputs": {"run_file:s02e04b": {"name": "x", "annex_key": "K1"}}}


@pytest.fixture
def setup(tmp_path):
    """Dataset at commit 'pinned' (rv=1), then 'wrong' (rv=2); side root rv=1."""
    ds = tmp_path / "ds"
    for rel in (RV, OTHER, FLAGS):
        (ds / rel).parent.mkdir(parents=True, exist_ok=True)
    _git(ds, "init", "-q")
    np.save(ds / RV, np.full((3, 3), 1.0))
    np.save(ds / OTHER, np.zeros((3, 3)))
    (ds / FLAGS).write_text("state,category\n0,eligible\n")
    _git(ds, "add", ".")
    _git(ds, "commit", "-q", "-m", "pinned")
    pinned = _git(ds, "rev-parse", "HEAD")
    np.save(ds / RV, np.full((3, 3), 2.0))
    _git(ds, "commit", "-q", "-am", "wrong")

    side = tmp_path / "side"
    so = side / "output"
    for rel in (RV, OTHER):
        (so / rel).parent.mkdir(parents=True, exist_ok=True)
    np.save(so / RV, np.full((3, 3), 1.0))
    np.save(so / OTHER, np.zeros((3, 3)))
    (so / REC).write_text(json.dumps(_record()))
    # an input linked in from the dataset is not a recomputed output
    (so / "05e_temporal_trend_a4").mkdir()
    os.symlink(ds / "05e_temporal_trend_a4" / "atlas-X", so / "05e_temporal_trend_a4" / "atlas-X")
    return ds, side, pinned


def _manifest(tmp_path, pinned, extra=""):
    p = tmp_path / "m.toml"
    p.write_text(textwrap.dedent(f"""
        message = "sub-06 05f on the pinned duplicate run"
        code_commit = "{MAIN}"
        change = ["05f_state_fc/atlas-X/sub-06/vt0.95/fc_similarity_corr_rv.npy"]
        [[control]]
        path = "{RV}"
        reference = "{pinned}"
        why = "05f loader: s02e04b must read ses-006"
        [[control]]
        path = "{RV}"
        reference = "HEAD"
        expect = "differ"
        min_diff = 0.5
        why = "the promoted file replaces the unpinned FC"
    """) + extra)
    return p


FAKE_DATALAD = """#!/bin/bash
# fake 'datalad save -m MSG [--amend] -- paths': git add + commit
shift; msg=""; amend=""
while [ "$1" != "--" ]; do case "$1" in -m) msg="$2"; shift 2;; --amend) amend=--amend; shift;; esac; done
shift
git add -- "$@" && git -c user.email=a@b -c user.name=a commit -q $amend -m "$msg"
"""


def _plan(ds, side, mpath):
    return promote.build_plan(side, ds, promote.load_manifest(mpath), repo=REPO_DIR)


def test_good_promotion_is_ready_and_executes(setup, tmp_path, monkeypatch):
    ds, side, pinned = setup
    m = _manifest(tmp_path, pinned)
    plan = _plan(ds, side, m)
    assert plan.report.fails == [], plan.report.fails
    assert sorted(r for _, r in plan.copies) == [RV, REC]
    fake = tmp_path / "fake_datalad"
    fake.write_text(FAKE_DATALAD)
    fake.chmod(0o755)
    monkeypatch.setenv("PROMOTE_DATALAD", str(fake))
    rc = promote.main(["--side-root", str(side), "--manifest", str(m), "--dataset", str(ds),
                       "--repo", str(REPO_DIR), "--no-fetch", "--execute"])
    assert rc == 0
    assert np.all(np.load(ds / RV) == 1.0)
    msg = _git(ds, "log", "-1", "--format=%B")
    assert f"code_commit: {MAIN}" in msg and "code_floor: 1f391c08e6" in msg


def test_undeclared_change_blocks(setup, tmp_path):
    ds, side, pinned = setup
    np.save(side / "output" / OTHER, np.ones((3, 3)))
    fails = _plan(ds, side, _manifest(tmp_path, pinned)).report.fails
    assert any("unexpected change outside the change set" in f and OTHER in f for f in fails)


def test_control_outside_the_change_set_or_not_recomputed_blocks(setup, tmp_path):
    ds, side, pinned = setup
    m = tmp_path / "m2.toml"
    m.write_text(textwrap.dedent(f"""
        message = "x"
        code_commit = "{MAIN}"
        change = ["{RV}"]
        no_positive_control = "test"
        [[control]]
        path = "{OTHER}"
        reference = "HEAD"
        why = "off the changed path"
        [[control]]
        path = "{FLAGS}"
        reference = "HEAD"
        why = "linked input, not recomputed"
    """))
    fails = _plan(ds, side, m).report.fails
    assert any("no control lies inside the change set" in f for f in fails)
    assert any(FLAGS in f and "cannot fail" in f for f in fails)


def test_control_against_the_pinned_version_fails_on_the_wrong_fc(setup, tmp_path):
    ds, side, pinned = setup
    np.save(side / "output" / RV, np.full((3, 3), 2.0))     # the side root reproduced the wrong FC
    fails = _plan(ds, side, _manifest(tmp_path, pinned)).report.fails
    assert any(f"control {RV} vs {pinned}" in f and "FAIL" in f for f in fails)
    assert any("expect differ" in f and "FAIL" in f for f in fails)


def test_missing_positive_control_blocks(setup, tmp_path):
    ds, side, pinned = setup
    m = tmp_path / "m3.toml"
    m.write_text(textwrap.dedent(f"""
        message = "x"
        code_commit = "{MAIN}"
        change = ["{RV}"]
        [[control]]
        path = "{RV}"
        reference = "{pinned}"
        why = "05f loader"
    """))
    assert any("no positive control" in f for f in _plan(ds, side, m).report.fails)


def test_record_with_wrong_session_or_branch_code_blocks(setup, tmp_path):
    ds, side, pinned = setup
    (side / "output" / REC).write_text(json.dumps(_record(sessions={"s02e04b": "ses-041"})))
    fails = _plan(ds, side, _manifest(tmp_path, pinned)).report.fails
    assert any("read from ses-041" in f for f in fails)
    first = _git(REPO_DIR, "rev-list", "--max-parents=0", "origin/main").splitlines()[0]
    (side / "output" / REC).write_text(json.dumps(_record(commit=first)))
    fails = _plan(ds, side, _manifest(tmp_path, pinned)).report.fails
    assert any("lacks floor fix" in f for f in fails)


def test_missing_record_for_a_provenance_stage_blocks(setup, tmp_path):
    ds, side, pinned = setup
    (side / "output" / REC).unlink()
    fails = _plan(ds, side, _manifest(tmp_path, pinned)).report.fails
    assert any("writes provenance, but the side root has no record" in f for f in fails)


def test_input_log_differs_from_dataset_record(setup, tmp_path):
    ds, side, pinned = setup
    old = _record()
    old["inputs"] = {"run_file:s02e04b": {"name": "x", "annex_key": "K0"}}
    (ds / REC).write_text(json.dumps(old))
    _git(ds, "add", REC)
    _git(ds, "commit", "-q", "-m", "record")
    m = _manifest(tmp_path, pinned)
    assert any("inputs differ" in f for f in _plan(ds, side, m).report.fails)
    m2 = _manifest(tmp_path, pinned, 'allow_input_changes = "02 rerun"\n')
    # TOML keys after a table belong to it; put the key at the top instead
    m2.write_text('allow_input_changes = "02 rerun"\n' + _manifest(tmp_path, pinned).read_text())
    plan = _plan(ds, side, m2)
    assert not any("inputs differ" in f for f in plan.report.fails)
    assert any("inputs differ" in w for w in plan.report.warns)


def test_broad_change_glob_is_refused(tmp_path):
    m = tmp_path / "m.toml"
    m.write_text(f'message = "x"\ncode_commit = "{MAIN}"\nchange = ["**"]\n')
    with pytest.raises(ValueError, match="declares everything"):
        promote.load_manifest(m)


def test_reference_missing_annex_content_is_reported(tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    _git(ds, "init", "-q")
    os.symlink(".git/annex/objects/xx/yy/MD5E-s1--abc.npy/MD5E-s1--abc.npy", ds / "f.npy")
    _git(ds, "add", "f.npy")
    _git(ds, "commit", "-q", "-m", "annexed")
    with pytest.raises(FileNotFoundError, match="not present"):
        promote.reference_file(ds, "HEAD", "f.npy", tmp_path)


# ------------------------------------------------- review follow-ups
def test_positive_control_needs_a_numeric_difference(tmp_path):
    a = _npy(tmp_path / "a.npy", np.full(3, 1.0))
    reshaped = compare_files(a, _npy(tmp_path / "b.npy", np.full(4, 1.0)))
    nan = compare_files(a, _npy(tmp_path / "c.npy", np.array([1.0, np.nan, 1.0])))
    moved = compare_files(a, _npy(tmp_path / "d.npy", np.full(3, 1.5)))
    assert reshaped.kind == nan.kind == "structural"
    assert moved.kind == "numeric" and moved.max_diff == pytest.approx(0.5)
    j = compare_files(_json(tmp_path / "j.json", {"sha": "a", "rho": 0.4}),
                      _json(tmp_path / "k.json", {"sha": "b", "rho": 0.5}))
    assert j.kind == "structural"          # a string difference is not a numeric change
    j2 = compare_files(_json(tmp_path / "l.json", {"a": 0.4, "b": 0.1}),
                       _json(tmp_path / "m.json", {"a": 0.4001, "b": 0.3}))
    assert j2.max_diff == pytest.approx(0.2)   # max over every leaf, not the first failing one


def test_differ_control_rules(setup, tmp_path):
    ds, side, pinned = setup
    def manifest(differ):
        m = tmp_path / "md.toml"
        m.write_text(textwrap.dedent(f"""
            message = "x"
            code_commit = "{MAIN}"
            change = ["{RV}"]
            [[control]]
            path = "{RV}"
            reference = "{pinned}"
            why = "05f loader"
            [[control]]
            why = "positive"
            expect = "differ"
        """) + differ)
        return m
    no_min = _plan(ds, side, manifest(f'path = "{RV}"\nreference = "HEAD"\n')).report.fails
    assert any("min_diff is required" in f for f in no_min)
    too_big = _plan(ds, side, manifest(f'path = "{RV}"\nreference = "HEAD"\nmin_diff = 5.0\n')).report.fails
    assert any("expect differ" in f and "FAIL" in f for f in too_big)
    outside = _plan(ds, side, manifest(f'path = "{OTHER}"\nreference = "HEAD"\nmin_diff = 0.1\n')).report.fails
    assert any("no positive control" in f and "inside the change set" in f for f in outside)


def test_csv_equal_infinities_and_decimals_override(tmp_path):
    (tmp_path / "a.csv").write_text("x\ninf\n-inf\n")
    (tmp_path / "b.csv").write_text("x\ninf\n-inf\n")
    assert compare_files(tmp_path / "a.csv", tmp_path / "b.csv").equal
    m = {"override": [{"glob": "**/metadata.json", "decimals": 4}]}
    rtol, atol, _ = promote._tol(m, "05f/x/metadata.json")
    assert compare_files(_json(tmp_path / "r1.json", {"a": 0.1235}), _json(tmp_path / "r2.json", {"a": 0.1234}),
                         rtol, atol).equal
    assert not compare_files(_json(tmp_path / "r3.json", {"a": 0.1237}), _json(tmp_path / "r4.json", {"a": 0.1234}),
                             rtol, atol).equal


def test_dataset_file_in_change_set_not_reproduced_blocks(setup, tmp_path):
    ds, side, pinned = setup
    extra = "05f_state_fc/atlas-X/sub-06/vt0.95/stale.npy"
    np.save(ds / extra, np.zeros(2))
    _git(ds, "add", extra)
    _git(ds, "commit", "-q", "-m", "stale")
    m = _manifest(tmp_path, pinned)
    m.write_text(m.read_text().replace(
        'change = ["05f_state_fc/atlas-X/sub-06/vt0.95/fc_similarity_corr_rv.npy"]',
        'change = ["05f_state_fc/atlas-X/sub-06/vt0.95/*.npy"]'))
    fails = _plan(ds, side, m).report.fails
    assert any("not reproduced by the side root" in f and extra in f for f in fails)


def test_write_through_a_linked_input_directory_blocks(setup, tmp_path):
    ds, side, pinned = setup
    (ds / "05e_temporal_trend_a4" / "atlas-X" / "sub-06" / "vt0.95" / "new.json").write_text("{}")
    fails = _plan(ds, side, _manifest(tmp_path, pinned)).report.fails
    assert any("wrote through the link" in f for f in fails)


def test_record_that_is_not_an_object_is_reported(setup, tmp_path):
    ds, side, pinned = setup
    (side / "output" / REC).write_text("[1, 2]")
    fails = _plan(ds, side, _manifest(tmp_path, pinned)).report.fails
    assert any("not a JSON object" in f for f in fails)


def test_large_promotions_save_in_chunks_into_one_commit(setup, tmp_path, monkeypatch):
    ds, side, pinned = setup
    so = side / "output"
    for i in range(5):
        np.save(so / f"05f_state_fc/atlas-X/sub-06/vt0.95/extra_{i}.npy", np.zeros(2))
    m = _manifest(tmp_path, pinned)
    m.write_text(m.read_text().replace(
        'change = ["05f_state_fc/atlas-X/sub-06/vt0.95/fc_similarity_corr_rv.npy"]',
        'change = ["05f_state_fc/atlas-X/sub-06/vt0.95/fc_similarity_corr_rv.npy", '
        '"05f_state_fc/atlas-X/sub-06/vt0.95/extra_*.npy"]'))
    fake = tmp_path / "fake_datalad"
    fake.write_text(FAKE_DATALAD)
    fake.chmod(0o755)
    monkeypatch.setenv("PROMOTE_DATALAD", str(fake))
    monkeypatch.setattr(promote, "SAVE_CHUNK", 2)
    before = int(_git(ds, "rev-list", "--count", "HEAD"))
    rc = promote.main(["--side-root", str(side), "--manifest", str(m), "--dataset", str(ds),
                       "--repo", str(REPO_DIR), "--no-fetch", "--execute"])
    assert rc == 0
    assert int(_git(ds, "rev-list", "--count", "HEAD")) == before + 1
    assert _git(ds, "status", "--porcelain", "--untracked-files=no") == ""


def test_structural_change_does_not_pass_a_positive_control(setup, tmp_path):
    ds, side, pinned = setup
    np.save(side / "output" / RV, np.full((4, 4), 1.0))   # reshaped: a broken recompute
    fails = _plan(ds, side, _manifest(tmp_path, pinned)).report.fails
    assert any("expect differ" in f and "structural difference" in f for f in fails)


def test_reference_found_in_the_object_store_without_git_annex(tmp_path, monkeypatch):
    ds = tmp_path / "ds"
    ds.mkdir()
    _git(ds, "init", "-q")
    key = "MD5E-s128--0123456789abcdef0123456789abcdef.npy"
    obj = ds / ".git" / "annex" / "objects" / "wF" / "xz" / key / key
    obj.parent.mkdir(parents=True)
    np.save(obj.with_suffix(""), np.arange(3.0))
    obj.with_suffix("").with_suffix(".npy").rename(obj)
    os.symlink(f".git/annex/objects/wF/xz/{key}/{key}", ds / "f.npy")
    _git(ds, "add", "f.npy")
    _git(ds, "commit", "-q", "-m", "annexed")
    monkeypatch.setattr(promote.shutil, "which", lambda name: None)
    assert np.array_equal(np.load(promote.reference_file(ds, "HEAD", "f.npy", tmp_path)), np.arange(3.0))
