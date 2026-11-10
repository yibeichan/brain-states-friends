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
    (r / "script" / "utils" / "datalad_save.sh").write_text('    ["05f"]="05f_state_fc"\n')
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


def _run_datalad_save(tmp_path, args=(), env_extra=None):
    """Run datalad_save.sh against stub uv/git-annex in a scratch dataset.

    The stub uv records each call: "check" for the code check (exit code from
    STUB_CHECK_RC), and save/push with whether .git/promote.lock is held."""
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
    (code / "script" / "utils").mkdir(parents=True)
    (code / "script" / "utils" / "code_floor.txt").write_text(
        "1f391c08e617cf608cb35e4c1a5ea7798d967aae 05f stub\n")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    report = tmp_path / "calls"
    tmp = tmp_path
    (bin_ / "git-annex").write_text("#!/bin/bash\nexit 0\n")
    (bin_ / "uv").write_text(f"""#!/bin/bash
# stub for: uv run --project X [--extra datalad] (python check_code.py ... | datalad save|push ...)
lock() {{ if flock -n "{out}/.git/promote.lock" true; then echo free; else echo held; fi; }}
if [[ " $* " == *check_code.py* ]]; then
  echo "check:$(lock)" >> "{report}"; printf '%s\n' "$@" > "{tmp}/check_args"; exit "${{STUB_CHECK_RC:-0}}"
fi
cmd=""; for a in "$@"; do case "$a" in save|push) cmd="$a";; esac; done
if [ "$cmd" = save ]; then
  for ((i=1; i<=$#; i++)); do if [ "${{!i}}" = "-m" ]; then j=$((i+1)); printf '%s' "${{!j}}" > "{tmp}/save_msg"; fi; done
fi
echo "$cmd:$(lock)" >> "{report}"
""")
    for f in bin_.iterdir():
        f.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", SLURM_SUBMIT_DIR=str(code), HOME=str(tmp_path))
    env.pop("CHECK_CODE", None)
    env.update(env_extra or {})
    p = subprocess.run(["bash", str(SCRIPT_DIR / "utils" / "datalad_save.sh"), "--stage", "06b",
                        "--subject", "sub-01", *args], env=env, capture_output=True, text=True)
    calls = report.read_text().split() if report.exists() else []
    _run_datalad_save.out = out
    return p, calls


def test_datalad_save_checks_code_by_default_and_holds_the_lock_while_saving(tmp_path):
    p, calls = _run_datalad_save(tmp_path)
    assert p.returncode == 0, p.stdout + p.stderr
    # the code check runs under the lock, then the save; the push after it does not hold it
    assert calls == ["check:held", "save:held", "push:free"]
    args = (tmp_path / "check_args").read_text().splitlines()
    out = _run_datalad_save.out
    assert args[args.index("tree") + 1] == f"{out}/06b_transition_structure/atlas-4S156Parcels/sub-01/"
    assert args[args.index("--stage") + 1] == "06b" and "--changed-only" in args
    assert "--fallback-commit" in args and "--no-fetch" not in args
    assert (tmp_path / "save_msg").read_text().rstrip().endswith("code_check: passed")
    script = (SCRIPT_DIR / "utils" / "datalad_save.sh").read_text()
    wait = int(script.split("flock -w ")[1].split()[0])
    hh, mm, ss = (int(x) for x in script.split("#SBATCH --time=")[1].split()[0].split(":"))
    assert wait < hh * 3600 + mm * 60 + ss       # the lock wait ends before the job's time limit


@pytest.mark.parametrize("args,env", [(["--no-check-code"], {}), ([], {"CHECK_CODE": "0"}),
                                      ([], {"CHECK_CODE": "no"})])
def test_datalad_save_skips_the_check_only_when_asked(tmp_path, args, env):
    p, calls = _run_datalad_save(tmp_path, args, env)
    assert p.returncode == 0, p.stdout + p.stderr
    assert calls == ["save:held", "push:free"]
    assert "WARNING: code check skipped" in p.stdout
    assert (tmp_path / "save_msg").read_text().rstrip().endswith("code_check: skipped (no reason given)")


def test_skip_reason_and_no_fetch_are_recorded(tmp_path):
    p, calls = _run_datalad_save(tmp_path, ["--no-check-code", "--skip-reason", "frozen worktree of ab88eb0"])
    assert p.returncode == 0, p.stdout + p.stderr
    assert (tmp_path / "save_msg").read_text().rstrip().endswith("code_check: skipped (frozen worktree of ab88eb0)")


def test_no_fetch_is_passed_to_the_check(tmp_path):
    p, calls = _run_datalad_save(tmp_path, ["--no-fetch"])
    assert p.returncode == 0, p.stdout + p.stderr
    assert "--no-fetch" in (tmp_path / "check_args").read_text().splitlines()
    assert "not fetched" in (tmp_path / "save_msg").read_text()


def test_failed_code_check_stops_the_save(tmp_path):
    p, calls = _run_datalad_save(tmp_path, env_extra={"STUB_CHECK_RC": "1"})
    assert p.returncode != 0
    assert calls == ["check:held"]                # nothing saved or pushed


def test_changed_only_checks_just_what_a_save_would_commit(repo, tmp_path):
    r, s, _ = repo
    ds = tmp_path / "ds"
    sub = ds / "05f_state_fc" / "atlas-X" / "sub-01"
    sub.mkdir(parents=True)
    _git(ds, "init", "-q")
    (sub / "legacy.npy").write_bytes(b"old")       # committed long ago, no provenance record
    _git(ds, "add", ".")
    _git(ds, "commit", "-q", "-m", "legacy")
    problems, notes = _tree(repo, sub, "05f", changed_only=True)
    assert problems == [] and any("nothing new or modified" in n for n in notes)
    (sub / "new.npy").write_bytes(b"new")          # this save would commit it: it must be covered
    problems, _ = _tree(repo, sub, "05f", changed_only=True)
    assert len(problems) == 1 and "new.npy" in problems[0]
    _write_rec(sub, s["after"], [sub / "new.npy"])
    problems, _ = _tree(repo, sub, "05f", changed_only=True)
    assert problems == [], problems
    (sub / "legacy.npy").write_bytes(b"modified")  # a modified file is checked too
    problems, _ = _tree(repo, sub, "05f", changed_only=True)
    assert len(problems) == 1 and "legacy.npy" in problems[0]
    # without changed_only the legacy file is checked as well
    problems, _ = _tree(repo, sub, "05f")
    assert any("legacy.npy" in p for p in problems)


def test_changed_only_outside_a_repository_fails(repo, tmp_path):
    d = tmp_path / "plain"
    d.mkdir()
    problems, _ = _tree(repo, d, "05f", changed_only=True)
    assert problems and "not inside a git repository" in problems[0]


def test_fetch_is_retried_once(monkeypatch):
    calls = []

    def fake_git(repo, *args, timeout=None):
        if args[0] == "fetch":
            calls.append(1)
            return (1, "", "cannot lock ref") if len(calls) == 1 else (0, "", "")
        return 0, "2026-11-10", ""
    monkeypatch.setattr(check_code, "_git", fake_git)
    ok, _ = check_code.fetch_main(wait=0)
    assert ok and len(calls) == 2


def test_floor_stage_keys_are_validated(tmp_path):
    p = tmp_path / "floor.txt"
    p.write_text("1f391c08e617cf608cb35e4c1a5ea7798d967aae 5f typo\n")
    with pytest.raises(ValueError, match="unknown stage key"):
        check_code.load_floor(p)
    p.write_text("1f391c08e617cf608cb35e4c1a5ea7798d967aae 05f,04rc,* fine\n")
    assert check_code.load_floor(p)[0][1] == frozenset({"05f", "04rc", "*"})


def test_untracked_python_makes_the_save_checkout_dirty(repo):
    r, s, _ = repo
    assert check_code.code_is_clean(r)
    (r / "script").mkdir(exist_ok=True)
    (r / "script" / "stray.py").write_text("x = 1\n")
    assert not check_code.code_is_clean(r)


# ------------------------------------------------- PR #64 review, round 2
def test_floor_stage_keys_are_validated_against_main_not_the_checkout(repo):
    """A stage key added on main (with its floor line) must not break an older checkout."""
    r, s, _ = repo
    _git(r, "checkout", "-q", "main")
    (r / "script" / "utils").mkdir(parents=True)
    (r / "script" / "utils" / "datalad_save.sh").write_text('    ["07z"]="07z_new_stage"\n')
    (r / "script" / "utils" / "code_floor.txt").write_text(f"{s['fix']} 07z a fix for a new stage\n")
    _git(r, "add", "script")
    _git(r, "commit", "-q", "-m", "new stage")
    _git(r, "update-ref", "refs/remotes/origin/main", _git(r, "rev-parse", "HEAD"))
    floor = check_code.load_floor(repo=r)           # this checkout's datalad_save.sh has no 07z
    assert floor[0][1] == frozenset({"07z"})
    with pytest.raises(ValueError, match="unknown stage key"):
        check_code.parse_floor(f"{s['fix']} 07y typo\n", known=check_code.known_stages(r, "origin/main"))


def test_deleting_an_output_its_record_lists_is_refused(repo, tmp_path):
    r, s, _ = repo
    ds = tmp_path / "ds"
    sub = ds / "05f_state_fc" / "atlas-X" / "sub-06"
    sub.mkdir(parents=True)
    _git(ds, "init", "-q")
    keep, gone = sub / "keep.npy", sub / "metadata.json"
    keep.write_bytes(b"k")
    gone.write_text("{}")
    _write_rec(sub, s["after"], [keep, gone])
    _git(ds, "add", ".")
    _git(ds, "commit", "-q", "-m", "recorded")
    gone.unlink()
    problems, _ = _tree(repo, sub, "05f", changed_only=True)
    assert len(problems) == 1 and "metadata.json" in problems[0] and "still lists it" in problems[0]


def test_skip_reason_with_a_newline_stays_on_one_line(tmp_path):
    p, calls = _run_datalad_save(tmp_path, ["--no-check-code", "--skip-reason", "x\ncode_check: passed"])
    assert p.returncode == 0, p.stdout + p.stderr
    msg = (tmp_path / "save_msg").read_text()
    assert msg.rstrip().endswith("code_check: skipped (x code_check: passed)")
    assert "\ncode_check: passed" not in msg


def test_dirty_checkout_names_the_cause(repo):
    r, s, _ = repo
    (r / "script").mkdir(exist_ok=True)
    (r / "script" / "stray.py").write_text("x = 1\n")
    assert check_code.checkout_dirt(r) == ["untracked .py under script/ (['script/stray.py'])"]
    (r / "base.txt").write_text("edited")
    why = check_code.checkout_dirt(r)
    assert why[0].startswith("modified tracked files (1") and len(why) == 2


def test_missing_stage_map_on_main_fails_closed_and_says_why(repo):
    r, s, _ = repo
    _git(r, "checkout", "-q", "main")
    (r / "script" / "utils").mkdir(parents=True)
    (r / "script" / "utils" / "code_floor.txt").write_text(f"{s['fix']} 06b a fix\n")   # no datalad_save.sh
    _git(r, "add", "script")
    _git(r, "commit", "-q", "-m", "floor without STAGE_MAP")
    _git(r, "update-ref", "refs/remotes/origin/main", _git(r, "rev-parse", "HEAD"))
    with pytest.raises(RuntimeError, match=r"cannot read STAGE_MAP from origin/main:script/utils/datalad_save.sh"):
        check_code.load_floor(repo=r)
    problems = check_code.check_commit(s["after"], "06b", repo=r)
    assert problems and problems[0].startswith("cannot read STAGE_MAP from origin/main")
