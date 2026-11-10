"""check_code: outputs must come from merged code that contains every required fix."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
from utils import check_code  # noqa: E402


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.email=a@b", "-c", "user.name=a", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """main: base -> fix -> after; a branch off base; origin/main = after."""
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    shas = {}
    for name in ("base", "fix", "after"):
        (r / f"{name}.txt").write_text(name)
        _git(r, "add", f"{name}.txt")
        _git(r, "commit", "-q", "-m", name)
        shas[name] = _git(r, "rev-parse", "HEAD")
    _git(r, "checkout", "-q", "-b", "feature", shas["base"])
    (r / "branch.txt").write_text("b")
    _git(r, "add", "branch.txt")
    _git(r, "commit", "-q", "-m", "branch")
    shas["branch"] = _git(r, "rev-parse", "HEAD")
    _git(r, "update-ref", "refs/remotes/origin/main", shas["after"])
    _git(r, "checkout", "-q", "main")
    floor = [(shas["fix"], frozenset({"05f"}), "the fix")]
    return r, shas, floor


def test_merged_commit_with_fix_passes(repo):
    r, s, floor = repo
    assert check_code.check_commit(s["after"], "05f", repo=r, floor=floor) == []
    assert check_code.check_commit(s["fix"], "05f", repo=r, floor=floor) == []


def test_commit_before_the_fix_fails_for_its_stage_only(repo):
    r, s, floor = repo
    problems = check_code.check_commit(s["base"], "05f", repo=r, floor=floor)
    assert len(problems) == 1 and "lacks floor fix" in problems[0]
    # the fix does not touch 06b's code path
    assert check_code.check_commit(s["base"], "06b", repo=r, floor=floor) == []
    # stage None (promotion code_commit) applies every floor entry
    assert check_code.check_commit(s["base"], None, repo=r, floor=floor)


def test_branch_commit_fails_even_with_the_fix_absent_from_floor(repo):
    r, s, _ = repo
    problems = check_code.check_commit(s["branch"], "06b", repo=r, floor=[])
    assert len(problems) == 1 and "not on origin/main" in problems[0]


def test_unknown_commit_fails(repo):
    r, _, floor = repo
    assert "unknown" in check_code.check_commit("0" * 40, "05f", repo=r, floor=floor)[0]


def _record(commit, **kw):
    rec = {"stage": "05f", "sub_id": "sub-06", "code": {"commit": commit, "dirty": False,
                                                      "head_changed": False},
           "sessions_read": {"s02e04b": "ses-006", "s02e05a": "ses-007"}}
    rec.update(kw)
    return rec


PINNED = {("sub-06", "s02e04b"): "ses-006"}


def test_record_reading_the_wrong_copy_fails(repo):
    r, s, floor = repo
    good = _record(s["after"])
    assert check_code.check_record(good, repo=r, floor=floor, pinned=PINNED) == []
    bad = _record(s["after"], sessions_read={"s02e04b": "ses-041"})
    problems = check_code.check_record(bad, repo=r, floor=floor, pinned=PINNED)
    assert problems == ["sub-06 s02e04b was read from ses-041; DUPLICATE_RUN_SESSIONS pins ses-006"]


def test_dirty_or_moving_checkout_fails(repo):
    r, s, floor = repo
    dirty = _record(s["after"], code={"commit": s["after"], "dirty": True, "modified": [" M x.py"]})
    assert "dirty" in check_code.check_record(dirty, repo=r, floor=floor, pinned=PINNED)[0]
    moved = _record(s["after"], code={"commit": s["after"], "head_changed": True,
                                     "commit_at_import": s["fix"]})
    assert "HEAD changed" in check_code.check_record(moved, repo=r, floor=floor, pinned=PINNED)[0]


def test_tree_requires_a_record_for_stages_that_write_one(repo, tmp_path):
    r, s, floor = repo
    out = tmp_path / "out" / "05f_state_fc" / "atlas-X" / "sub-06"
    out.mkdir(parents=True)
    problems, _ = check_code.check_tree(out, "05f", repo=r, floor=floor)
    assert problems and "writes provenance" in problems[0]
    (out / "provenance_05f.json").write_text(json.dumps(_record(s["after"])))
    problems, notes = check_code.check_tree(out, "05f", repo=r, floor=floor)
    assert problems == [] and "1 provenance record" in notes[0]


def test_tree_without_records_checks_the_save_checkout_and_says_it_is_weak(repo, tmp_path):
    r, s, floor = repo
    out = tmp_path / "06b"
    out.mkdir()
    problems, notes = check_code.check_tree(out, "06b", repo=r, floor=floor, fallback_commit=s["branch"])
    assert "WEAK CHECK" in notes[0]
    assert problems and "not on origin/main" in problems[0]


def test_floor_file_parses_and_lists_the_duplicate_run_fix():
    floor = check_code.load_floor()
    sha, stages, _ = floor[0]
    assert sha.startswith("1f391c0") and "05f" in stages
    assert check_code.PROVENANCE_STAGES >= {"03a", "04rc", "05e_a1", "05e_a3", "05f"}


def test_floor_line_needs_a_full_sha(tmp_path):
    p = tmp_path / "floor.txt"
    p.write_text("1f391c0 05f short sha\n")
    with pytest.raises(ValueError):
        check_code.load_floor(p)
