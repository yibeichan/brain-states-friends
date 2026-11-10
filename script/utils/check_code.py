#!/usr/bin/env python3
"""Check that outputs come from merged code that contains every required fix.

Used by datalad_save.sh (--check-code) and promote.py before anything enters
the output dataset. Rules (docs/output_provenance.md):

1. The code commit is on main. The repo squash-merges, so a commit on a PR
   branch never becomes an ancestor of main: results computed on a branch are
   saved only after the PR merges and the stage is rerun from main.
2. The commit contains every code-floor fix (code_floor.txt) for its stage.
3. A provenance record is clean: no modified tracked files, no untracked .py
   under script/, HEAD unchanged during the run.
4. The sessions a record read for duplicated runs equal DUPLICATE_RUN_SESSIONS.
5. A stage that writes provenance (PROVENANCE_STAGES) must have a record.

CLI:
    check_code.py commit <sha> [--stage S]
    check_code.py tree <path> --stage S [--fallback-commit SHA]
Exit status 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = SCRIPT_DIR.parent
FLOOR_PATH = Path(__file__).resolve().with_name("code_floor.txt")

# Stages that write provenance_<stage>.json (utils/provenance.py). For these a
# missing record is a failure, not a warning.
PROVENANCE_STAGES = frozenset({"03a", "04rc", "05e_a1", "05e_a3", "05f"})


def _git(repo, *args, timeout=None):
    try:
        p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def load_floor(path=FLOOR_PATH):
    """List of (sha, stages, reason); stages is a frozenset, {'*'} for all."""
    entries = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 2)
        if len(parts) < 3 or len(parts[0]) != 40:
            raise ValueError(f"{path}:{n}: expected '<40-char sha> <stages> <reason>'")
        entries.append((parts[0], frozenset(parts[1].split(",")), parts[2]))
    return entries


def floor_for(stage, floor):
    """Floor entries that apply to ``stage`` (all entries when stage is None)."""
    return [e for e in floor if stage is None or "*" in e[1] or stage in e[1]]


def fetch_main(repo=REPO_DIR, remote="origin", branch="main"):
    """Best-effort fetch so origin/main is current; returns a one-line note."""
    rc, _, err = _git(repo, "fetch", "-q", remote, branch, timeout=60)
    _, date, _ = _git(repo, "log", "-1", "--format=%ci", f"{remote}/{branch}")
    if rc != 0:
        return f"WARNING: git fetch {remote} {branch} failed ({err or rc}); {remote}/{branch} is from {date or 'unknown'}"
    return f"{remote}/{branch} fetched; last commit {date}"


def is_ancestor(repo, a, b):
    rc, _, _ = _git(repo, "merge-base", "--is-ancestor", a, b)
    return rc == 0


def check_commit(commit, stage=None, repo=REPO_DIR, main_ref="origin/main", floor=None):
    """Problems with code ``commit`` for ``stage`` (empty list means it passes)."""
    floor = load_floor() if floor is None else floor
    problems = []
    _, shallow, _ = _git(repo, "rev-parse", "--is-shallow-repository")
    if shallow == "true":
        return [f"{repo} is a shallow clone; ancestry cannot be checked"]
    rc, full, _ = _git(repo, "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}")
    if rc != 0 or not full:
        return [f"code commit {commit} is unknown in {repo}"]
    rc, _, _ = _git(repo, "rev-parse", "--verify", "--quiet", main_ref)
    if rc != 0:
        return [f"{main_ref} is unknown in {repo}"]
    if not is_ancestor(repo, full, main_ref):
        problems.append(
            f"code commit {full[:10]} is not on {main_ref}: results from a branch are "
            "saved only after the PR merges and the stage is rerun from main "
            "(squash merges mean a branch commit never becomes part of main)")
    for sha, _stages, reason in floor_for(stage, floor):
        if not is_ancestor(repo, sha, full):
            problems.append(f"code commit {full[:10]} lacks floor fix {sha[:10]} ({reason})")
    return problems


def check_record(record, repo=REPO_DIR, main_ref="origin/main", floor=None, pinned=None):
    """Problems with one provenance record (see utils/provenance.py)."""
    if pinned is None:
        sys.path.insert(0, str(SCRIPT_DIR))
        from utils.common import DUPLICATE_RUN_SESSIONS as pinned
    stage = record.get("stage")
    code = record.get("code") or {}
    commit = code.get("commit")
    problems = []
    if not commit:
        return [f"record for stage {stage} has no code commit"]
    if code.get("dirty"):
        problems.append(
            f"code checkout was dirty (modified: {code.get('modified') or []}, "
            f"untracked .py: {code.get('untracked_py') or []})")
    if code.get("head_changed"):
        problems.append(f"HEAD changed during the run ({code.get('commit_at_import')} -> {commit})")
    problems += check_commit(commit, stage, repo=repo, main_ref=main_ref, floor=floor)
    sub = record.get("sub_id")
    for run, ses in (record.get("sessions_read") or {}).items():
        want = pinned.get((sub, run))
        if want is not None and ses != want:
            problems.append(f"{sub} {run} was read from {ses}; DUPLICATE_RUN_SESSIONS pins {want}")
    return problems


def find_records(path):
    path = Path(path)
    if path.is_file():
        return [path] if path.name.startswith("provenance_") else []
    return sorted(path.rglob("provenance_*.json"))


def code_is_clean(repo):
    _, tracked, _ = _git(repo, "status", "--porcelain", "--untracked-files=no")
    return not tracked


def check_tree(path, stage, repo=REPO_DIR, main_ref="origin/main", floor=None,
               fallback_commit=None):
    """Check every provenance record under ``path``; returns (problems, notes)."""
    problems, notes = [], []
    records = find_records(path)
    for rp in records:
        try:
            rec = json.loads(rp.read_text())
        except (OSError, ValueError) as e:
            problems.append(f"{rp}: unreadable provenance ({e})")
            continue
        for p in check_record(rec, repo=repo, main_ref=main_ref, floor=floor):
            problems.append(f"{rp}: {p}")
    if records:
        notes.append(f"{len(records)} provenance record(s) checked under {path}")
    elif stage in PROVENANCE_STAGES:
        problems.append(f"stage {stage} writes provenance, but none is under {path}: "
                        "rerun the stage on current code")
    else:
        notes.append(f"WEAK CHECK: stage {stage} writes no provenance; only the save-time "
                     "checkout is checked, not the code that produced the files")
        if fallback_commit:
            problems += [f"save-time checkout: {p}" for p in
                         check_commit(fallback_commit, stage, repo=repo, main_ref=main_ref, floor=floor)]
            if not code_is_clean(repo):
                problems.append(f"save-time checkout {repo} has modified tracked files")
    return problems, notes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(REPO_DIR))
    ap.add_argument("--main-ref", default="origin/main")
    ap.add_argument("--no-fetch", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("commit")
    c.add_argument("sha")
    c.add_argument("--stage")
    t = sub.add_parser("tree")
    t.add_argument("path")
    t.add_argument("--stage", required=True)
    t.add_argument("--fallback-commit")
    a = ap.parse_args(argv)

    if not a.no_fetch:
        print(fetch_main(a.repo))
    if a.cmd == "commit":
        problems, notes = check_commit(a.sha, a.stage, repo=a.repo, main_ref=a.main_ref), []
    else:
        problems, notes = check_tree(a.path, a.stage, repo=a.repo, main_ref=a.main_ref,
                                     fallback_commit=a.fallback_commit)
    for n in notes:
        print(n)
    for p in problems:
        print("FAIL: " + p)
    print("code check: " + ("PASS" if not problems else f"FAIL ({len(problems)})"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
