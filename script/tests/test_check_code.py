"""check_code: outputs must come from merged code that contains every required fix."""
import hashlib
import json
import os
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
    rec = {"schema": 2, "stage": "05f", "sub_id": "sub-06",
           "code": {"commit": commit, "dirty": False, "head_changed": False},
           "sessions_read": {"s02e04b": "ses-006", "s02e05a": "ses-007"},
           "outputs": {}}
    rec.update(kw)
    return rec


PINNED = {("sub-06", "s02e04b"): "ses-006"}


def _md5(f):
    return hashlib.md5(Path(f).read_bytes()).hexdigest()


def _write_rec(d, commit, files, stage="05f", **kw):
    """provenance_<stage>.json in d listing ``files`` (paths) with their checksums."""
    outputs = {str(Path(f).relative_to(d)): {"size": Path(f).stat().st_size, "md5": _md5(f)}
               for f in files}
    (Path(d) / f"provenance_{stage}.json").write_text(json.dumps(_record(commit, stage=stage,
                                                                         outputs=outputs, **kw)))


def _tree(repo, path, stage, **kw):
    r, s, floor = repo
    return check_code.check_tree(path, stage, repo=r, floor=floor, pinned=PINNED, **kw)


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
    at_import = _record(s["after"], code={"commit": s["after"], "dirty": False, "dirty_at_import": True,
                                          "modified_at_import": [" M script/05f_state_fc.py"]})
    assert "dirty" in check_code.check_record(at_import, repo=r, floor=floor, pinned=PINNED)[0]
    moved = _record(s["after"], code={"commit": s["after"], "head_changed": True,
                                     "commit_at_import": s["fix"]})
    assert "HEAD changed" in check_code.check_record(moved, repo=r, floor=floor, pinned=PINNED)[0]


def test_record_without_output_checksums_fails(repo):
    r, s, floor = repo
    old = _record(s["after"], schema=1)
    del old["outputs"]
    assert any("schema < 2" in p for p in check_code.check_record(old, repo=r, floor=floor, pinned=PINNED))


def test_every_output_must_be_listed_in_its_nearest_record(repo, tmp_path):
    r, s, _ = repo
    vt = tmp_path / "05f_state_fc" / "atlas-X" / "sub-06" / "vt0.95"
    vt.mkdir(parents=True)
    a, b = vt / "a.npy", vt / "b.npy"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    problems, _ = _tree(repo, vt.parent, "05f")
    assert len(problems) == 2 and all("no provenance_05f.json" in p for p in problems)
    _write_rec(vt, s["after"], [a, b])
    problems, notes = _tree(repo, vt.parent, "05f")
    assert problems == [] and "1 provenance record" in notes[0]
    b.write_bytes(b"changed")                       # a partial or later rerun rewrote it
    problems, _ = _tree(repo, vt.parent, "05f")
    assert len(problems) == 1 and "b.npy" in problems[0] and "checksum differs" in problems[0]
    (vt / "c.npy").write_bytes(b"c")                # written by a run the record does not describe
    problems, _ = _tree(repo, vt.parent, "05f")
    assert any("c.npy" in p and "not listed" in p for p in problems)


def test_a_record_below_does_not_cover_files_above_it(repo, tmp_path):
    r, s, _ = repo
    sub = tmp_path / "05f_state_fc" / "atlas-X" / "sub-06"
    (sub / "vt0.95").mkdir(parents=True)
    top = sub / "state_delta_fc.npy"
    top.write_bytes(b"x")
    low = sub / "vt0.95" / "a.npy"
    low.write_bytes(b"y")
    _write_rec(sub / "vt0.95", s["after"], [low])
    problems, _ = _tree(repo, sub, "05f")
    assert len(problems) == 1 and "state_delta_fc.npy" in problems[0]


def test_04rc_outputs_in_a_stage_04_save_need_a_04rc_record(repo, tmp_path):
    r, s, _ = repo
    sub = tmp_path / "04_combined_hdphmm" / "atlas-X" / "sub-01"
    fold = sub / "loso" / "season_1"
    fold.mkdir(parents=True)
    fc = fold / "state_empirical_corr.npy"
    fc.write_bytes(b"fc")
    (fold / "loso_results.json").write_text("{}")   # a stage-04 file: weak check only
    problems, notes = _tree(repo, sub, "04", fallback_commit=s["after"])
    assert any("state_empirical_corr.npy" in p and "provenance_04rc.json" in p for p in problems)
    _write_rec(fold, s["after"], [fc], stage="04rc", sub_id="sub-01")
    problems, notes = _tree(repo, sub, "04", fallback_commit=s["after"])
    assert problems == [] and any("WEAK CHECK" in n for n in notes)


def test_weak_stage_checks_the_save_checkout(repo, tmp_path):
    r, s, _ = repo
    out = tmp_path / "06b"
    out.mkdir()
    (out / "x.json").write_text("{}")
    problems, notes = _tree(repo, out, "06b", fallback_commit=s["branch"])
    assert any("WEAK CHECK" in n for n in notes)
    assert problems and "not on origin/main" in problems[0]


def test_record_that_is_not_an_object_fails(repo, tmp_path):
    r, s, _ = repo
    d = tmp_path / "05f_state_fc" / "x"
    d.mkdir(parents=True)
    (d / "a.npy").write_bytes(b"a")
    (d / "provenance_05f.json").write_text("[]")
    problems, _ = _tree(repo, d, "05f")
    assert any("not a JSON object" in p for p in problems)


def test_annexed_dataset_file_is_checked_from_its_key(repo, tmp_path):
    r, s, _ = repo
    d = tmp_path / "05f_state_fc" / "atlas-X" / "sub-06"
    d.mkdir(parents=True)
    content = b"0123456789"
    key = f"MD5E-s{len(content)}--{hashlib.md5(content).hexdigest()}.npy"
    os.symlink(f"../../../.git/annex/objects/aa/bb/{key}/{key}", d / "a.npy")   # content not present
    rec = _record(s["after"], outputs={"a.npy": {"size": len(content), "md5": hashlib.md5(content).hexdigest()}})
    (d / "provenance_05f.json").write_text(json.dumps(rec))
    problems, _ = _tree(repo, d, "05f")
    assert problems == []


def test_floor_and_pins_are_read_from_main_not_the_running_checkout(repo):
    """An old checkout must not pass an output that a newer floor line on main rules out."""
    r, s, _ = repo
    _git(r, "checkout", "-q", "main")
    (r / "script" / "utils").mkdir(parents=True)
    (r / "script" / "utils" / "code_floor.txt").write_text(f"{s['fix']} 05f the fix\n")
    (r / "script" / "utils" / "common.py").write_text('DUPLICATE_RUN_SESSIONS = {("sub-06", "s02e04b"): "ses-006"}\n')
    _git(r, "add", "script")
    _git(r, "commit", "-q", "-m", "floor line")
    _git(r, "update-ref", "refs/remotes/origin/main", _git(r, "rev-parse", "HEAD"))
    _git(r, "checkout", "-q", s["base"])            # the old commit runs the check
    problems = check_code.check_commit(s["base"], "05f", repo=r)
    assert any("lacks floor fix" in p for p in problems)
    assert check_code.load_pins(repo=r) == {("sub-06", "s02e04b"): "ses-006"}


def test_missing_floor_on_main_fails_closed(repo):
    r, s, _ = repo
    problems = check_code.check_commit(s["after"], "05f", repo=r)
    assert problems and "cannot read script/utils/code_floor.txt" in problems[0]


def test_floor_file_parses_and_lists_the_duplicate_run_fix():
    floor = check_code.load_floor(path=SCRIPT_DIR / "utils" / "code_floor.txt")
    sha, stages, _ = floor[0]
    assert sha.startswith("1f391c0") and "05f" in stages
    assert check_code.load_floor(repo=SCRIPT_DIR.parent, main_ref="HEAD") == floor
    assert check_code.PROVENANCE_STAGES >= {"03a", "04rc", "05e_a1", "05e_a3", "05f"}


def test_floor_line_needs_a_full_sha(tmp_path):
    p = tmp_path / "floor.txt"
    p.write_text("1f391c0 05f short sha\n")
    with pytest.raises(ValueError):
        check_code.load_floor(p)


def test_failed_fetch_is_a_failure_unless_no_fetch(repo, capsys):
    r, s, _ = repo
    rc = check_code.main(["--repo", str(r), "commit", s["after"], "--stage", "06b"])
    assert rc == 1 and "cannot confirm origin/main is current" in capsys.readouterr().out


def test_datalad_save_accepts_true_and_rejects_unknown_check_code_values():
    script = (SCRIPT_DIR / "utils" / "datalad_save.sh").read_text()
    assert '1|true|yes|on) CHECK_CODE=1' in script and '*) echo "ERROR: CHECK_CODE=' in script


def test_datalad_save_holds_the_promotion_lock_while_saving(tmp_path):
    """Run datalad_save.sh against stub uv/git-annex in a scratch dataset; the stub
    datalad reports whether .git/promote.lock is held while it runs."""
    code = tmp_path / "code"
    code.mkdir()
    scratch = tmp_path / "scratch"
    out = scratch / "output"
    (out / ".datalad").mkdir(parents=True)
    (out / "06b_transition_structure" / "atlas-4S156Parcels" / "sub-01").mkdir(parents=True)
    _git(out, "init", "-q")
    (code / ".env").write_text(f"SCRATCH_DIR='{scratch}'\nDATA_DIR='{tmp_path / 'data'}'\n")
    _git(code, "init", "-q")
    _git(code, "commit", "-q", "--allow-empty", "-m", "c")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    report = tmp_path / "lock_state"
    (bin_ / "git-annex").write_text("#!/bin/bash\nexit 0\n")
    (bin_ / "uv").write_text(f"""#!/bin/bash
# stub for: uv run --project X --extra datalad datalad save|push ...
cmd=""; for a in "$@"; do case "$a" in save|push) cmd="$a";; esac; done
if flock -n "{out}/.git/promote.lock" true; then echo "$cmd:free" >> "{report}"; else echo "$cmd:held" >> "{report}"; fi
""")
    for f in bin_.iterdir():
        f.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", SLURM_SUBMIT_DIR=str(code), HOME=str(tmp_path))
    p = subprocess.run(["bash", str(SCRIPT_DIR / "utils" / "datalad_save.sh"), "--stage", "06b",
                        "--subject", "sub-01"], env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    # the save runs under the lock; the push after it does not hold it
    assert report.read_text().split() == ["save:held", "push:free"]
    script = (SCRIPT_DIR / "utils" / "datalad_save.sh").read_text()
    wait = int(script.split("flock -w ")[1].split()[0])
    hh, mm, ss = (int(x) for x in script.split("#SBATCH --time=")[1].split()[0].split(":"))
    assert wait < hh * 3600 + mm * 60 + ss       # the lock wait ends before the job's time limit
