#!/usr/bin/env python3
"""Check that outputs come from merged code that contains every required fix.

Used by datalad_save.sh (--check-code) and promote.py before anything enters
the output dataset. Rules (script/utils/output_provenance.md):

1. The code commit is on main. The repo squash-merges, so a commit on a PR
   branch never becomes an ancestor of main: results computed on a branch are
   saved only after the PR merges and the stage is rerun from main.
2. The commit contains every code-floor fix for its stage. The floor and the
   duplicate-run table are read from main (``git show <main_ref>:...``), not
   from the checkout running the check, so an old checkout cannot pass outputs
   that a newer fix on main rules out.
3. A provenance record is clean: no modified tracked files or untracked .py
   under script/ when the code was loaded or when the record was written, and
   HEAD unchanged during the run.
4. The sessions a record read for duplicated runs equal DUPLICATE_RUN_SESSIONS.
5. Every output of a stage that writes provenance is listed, with its md5, in
   the nearest record of that stage (the closest ancestor directory holding
   provenance_<stage>.json). A file the record does not list, or lists with
   another checksum, is not covered.

CLI:
    check_code.py commit <sha> [--stage S]
    check_code.py tree <path> --stage S [--fallback-commit SHA]
Exit status 0 when every check passes, 1 otherwise. A failed fetch of main is
a failure unless --no-fetch is given.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = SCRIPT_DIR.parent
FLOOR_REL = "script/utils/code_floor.txt"
PINS_REL = "script/utils/common.py"

# Output top directories of the stages that write provenance_<stage>.json.
STAGE_TOPS = {
    "03a_pca4combined_hmm": "03a",
    "05e_temporal_trend_a1": "05e_a1",
    "05e_temporal_trend_a3": "05e_a3",
    "05f_state_fc": "05f",
}
# 04rc writes these two files into stage 04's LOSO and split-half fold directories.
FILES_04RC = frozenset({"state_empirical_corr.npy", "fc_metadata.json"})
PROVENANCE_STAGES = frozenset(set(STAGE_TOPS.values()) | {"04rc"})
_MD5_KEY_RE = re.compile(r"^MD5E?-s(\d+)(?:-m\d+)?--([0-9a-f]{32})")


def _git(repo, *args, timeout=None):
    try:
        p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except OSError as e:
        return 127, "", str(e)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


# ------------------------------------------------------------ floor and pins
def parse_floor(text, origin="code_floor.txt"):
    """List of (sha, stages, reason); stages is a frozenset, {'*'} for all."""
    entries = []
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 2)
        if len(parts) < 3 or len(parts[0]) != 40:
            raise ValueError(f"{origin}:{n}: expected '<40-char sha> <stages> <reason>'")
        entries.append((parts[0], frozenset(parts[1].split(",")), parts[2]))
    return entries


def load_floor(path=None, repo=REPO_DIR, main_ref="origin/main"):
    """The code floor as of ``main_ref`` (or from ``path`` when given)."""
    if path is not None:
        return parse_floor(Path(path).read_text(), str(path))
    rc, text, err = _git(repo, "show", f"{main_ref}:{FLOOR_REL}")
    if rc != 0:
        raise RuntimeError(f"cannot read {FLOOR_REL} from {main_ref}: {err or rc}")
    return parse_floor(text, f"{main_ref}:{FLOOR_REL}")


def parse_pins(source):
    """DUPLICATE_RUN_SESSIONS from the source text of utils/common.py."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "DUPLICATE_RUN_SESSIONS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("DUPLICATE_RUN_SESSIONS not found")


def load_pins(repo=REPO_DIR, main_ref="origin/main"):
    rc, text, err = _git(repo, "show", f"{main_ref}:{PINS_REL}")
    if rc != 0:
        raise RuntimeError(f"cannot read {PINS_REL} from {main_ref}: {err or rc}")
    return parse_pins(text)


def floor_for(stage, floor):
    """Floor entries that apply to ``stage`` (all entries when stage is None)."""
    return [e for e in floor if stage is None or "*" in e[1] or stage in e[1]]


def fetch_main(repo=REPO_DIR, remote="origin", branch="main"):
    """Fetch main; returns (ok, one-line note)."""
    rc, _, err = _git(repo, "fetch", "-q", remote, branch, timeout=60)
    _, date, _ = _git(repo, "log", "-1", "--format=%ci", f"{remote}/{branch}")
    if rc != 0:
        return False, (f"git fetch {remote} {branch} failed ({err or rc}); "
                       f"{remote}/{branch} is from {date or 'unknown'}")
    return True, f"{remote}/{branch} fetched; last commit {date}"


def is_ancestor(repo, a, b):
    rc, _, _ = _git(repo, "merge-base", "--is-ancestor", a, b)
    return rc == 0


# ----------------------------------------------------------------- commits
def check_commit(commit, stage=None, repo=REPO_DIR, main_ref="origin/main", floor=None):
    """Problems with code ``commit`` for ``stage`` (empty list means it passes)."""
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
    if floor is None:
        try:
            floor = load_floor(repo=repo, main_ref=main_ref)
        except (RuntimeError, ValueError) as e:
            return [str(e)]
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
        try:
            pinned = load_pins(repo=repo, main_ref=main_ref)
        except (RuntimeError, ValueError) as e:
            return [str(e)]
    stage = record.get("stage")
    code = record.get("code") if isinstance(record.get("code"), dict) else {}
    commit = code.get("commit")
    problems = []
    if not commit:
        return [f"record for stage {stage} has no code commit"]
    if record.get("schema", 0) < 2 or not isinstance(record.get("outputs"), dict):
        problems.append("record predates output checksums (schema < 2); rerun the stage on current code")
    if code.get("dirty") or code.get("dirty_at_import"):
        problems.append(
            "code checkout was dirty (at import: modified "
            f"{code.get('modified_at_import') or []}, untracked .py {code.get('untracked_py_at_import') or []}; "
            f"at write: modified {code.get('modified') or []}, untracked .py {code.get('untracked_py') or []})")
    if code.get("head_changed"):
        problems.append(f"HEAD changed during the run ({code.get('commit_at_import')} -> {commit})")
    problems += check_commit(commit, stage, repo=repo, main_ref=main_ref, floor=floor)
    sub = record.get("sub_id")
    for run, ses in (record.get("sessions_read") or {}).items():
        want = pinned.get((sub, run))
        if want is not None and ses != want:
            problems.append(f"{sub} {run} was read from {ses}; DUPLICATE_RUN_SESSIONS pins {want}")
    return problems


# ---------------------------------------------------------------- coverage
def gated_stage(path):
    """Stage whose provenance record must cover the output at ``path``, or None.

    ``path`` may be relative to the output root or absolute; the stage is found
    from the first path component that names a provenance stage's directory.
    """
    parts = Path(path).parts
    for i, part in enumerate(parts):
        if part in STAGE_TOPS:
            return STAGE_TOPS[part]
        if part == "04_combined_hdphmm" and parts[-1] in FILES_04RC:
            return "04rc"
    return None


def _stage_top(path):
    """The stage's top directory containing ``path`` (where record lookup stops)."""
    p = Path(path)
    for anc in [p, *p.parents]:
        if anc.name in STAGE_TOPS or anc.name == "04_combined_hdphmm":
            return anc
    return None


def nearest_record(path, stage):
    """Closest provenance_<stage>.json in path's directory or an ancestor,
    without leaving the stage's top directory."""
    p = Path(path)
    top = _stage_top(p)
    d = p.parent
    while True:
        cand = d / f"provenance_{stage}.json"
        if cand.is_file():
            return cand
        if top is None or d == top or d == d.parent:
            return None
        d = d.parent


def content_id(path):
    """(size, md5) of a file, from its MD5E annex key when it is annexed."""
    real = Path(os.path.realpath(path))       # resolves even when the annexed content is absent
    m = _MD5_KEY_RE.match(real.name)
    if m:
        return int(m.group(1)), m.group(2)
    import hashlib
    h = hashlib.md5()
    try:
        with open(real, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return real.stat().st_size, h.hexdigest()
    except OSError:
        return None


def _load_record(rp):
    try:
        rec = json.loads(Path(rp).read_text())
    except (OSError, ValueError) as e:
        return None, f"unreadable provenance ({e})"
    if not isinstance(rec, dict):
        return None, "provenance is not a JSON object"
    return rec, None


def coverage_problem(path, stage, records_cache=None):
    """Why the output at ``path`` is not covered by its stage's record, or None."""
    rp = nearest_record(path, stage)
    if rp is None:
        return f"no provenance_{stage}.json in its directory or above: rerun the stage on current code"
    cache = {} if records_cache is None else records_cache
    if rp not in cache:
        cache[rp] = _load_record(rp)
    rec, err = cache[rp]
    if err:
        return f"{rp.name}: {err}"
    rel = os.path.relpath(path, rp.parent)
    entry = (rec.get("outputs") or {}).get(rel)
    if not isinstance(entry, dict):
        return f"not listed in {rp} (written by another run, or the record predates it)"
    cid = content_id(path)
    if cid is None:
        return "content not readable"
    if (entry.get("size"), entry.get("md5")) != cid:
        return f"checksum differs from {rp} (the file changed after the recorded run)"
    return None


def find_records(path):
    path = Path(path)
    if path.is_file():
        return [path] if path.name.startswith("provenance_") else []
    return sorted(path.rglob("provenance_*.json"))


def code_is_clean(repo):
    _, tracked, _ = _git(repo, "status", "--porcelain", "--untracked-files=no")
    return not tracked


def _files_under(path):
    path = Path(path)
    if path.is_file() or os.path.islink(path):
        return [path]
    out = []
    for root, dirs, names in os.walk(path):
        dirs[:] = [d for d in dirs if d != ".git"]
        for n in names:
            if not (n.startswith("provenance_") and n.endswith(".json")):
                out.append(Path(root) / n)
    return sorted(out)


def check_tree(path, stage, repo=REPO_DIR, main_ref="origin/main", floor=None,
               fallback_commit=None, pinned=None):
    """Check the outputs under ``path``; returns (problems, notes).

    Every record under ``path``, or nearest to a file under it, is checked.
    Every file of a provenance stage (including 04rc's files inside a stage-04
    save) must be covered by its nearest record. Files of other stages get the
    weak check: the save-time checkout must pass the floor and be clean.
    """
    problems, notes = [], []
    if floor is None:
        try:
            floor = load_floor(repo=repo, main_ref=main_ref)
        except (RuntimeError, ValueError) as e:
            return [str(e)], notes
    if pinned is None:
        try:
            pinned = load_pins(repo=repo, main_ref=main_ref)
        except (RuntimeError, ValueError) as e:
            return [str(e)], notes
    cache = {}
    uncovered, weak = [], 0
    records = set(find_records(path))
    for f in _files_under(path):
        st = gated_stage(f)
        if st is None:
            weak += 1
            continue
        why = coverage_problem(f, st, cache)
        if why:
            uncovered.append(f"{f}: {why}")
        rp = nearest_record(f, st)
        if rp is not None:
            records.add(rp)
    for rp in sorted(records):
        rec, err = cache.get(rp) or _load_record(rp)
        if err:
            problems.append(f"{rp}: {err}")
            continue
        for p in check_record(rec, repo=repo, main_ref=main_ref, floor=floor, pinned=pinned):
            problems.append(f"{rp}: {p}")
    problems += uncovered[:20]
    if len(uncovered) > 20:
        problems.append(f"... and {len(uncovered) - 20} more uncovered outputs")
    if records:
        notes.append(f"{len(records)} provenance record(s) checked for {path}")
    if weak:
        notes.append(f"WEAK CHECK: {weak} file(s) belong to stages that write no provenance; only the "
                     "save-time checkout is checked, not the code that produced them")
        if fallback_commit:
            problems += [f"save-time checkout: {p}" for p in
                         check_commit(fallback_commit, stage, repo=repo, main_ref=main_ref, floor=floor)]
            if not code_is_clean(repo):
                problems.append(f"save-time checkout {repo} has modified tracked files")
    if not records and not weak and not uncovered:
        problems.append(f"nothing to check under {path}")
    return problems, notes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(REPO_DIR))
    ap.add_argument("--main-ref", default="origin/main")
    ap.add_argument("--no-fetch", action="store_true",
                    help="use the local origin/main as is (a failed fetch is otherwise a failure)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("commit")
    c.add_argument("sha")
    c.add_argument("--stage")
    t = sub.add_parser("tree")
    t.add_argument("path")
    t.add_argument("--stage", required=True)
    t.add_argument("--fallback-commit")
    a = ap.parse_args(argv)

    problems, notes = [], []
    if not a.no_fetch:
        ok, note = fetch_main(a.repo)
        print(note)
        if not ok:
            problems.append(f"cannot confirm {a.main_ref} is current: {note}")
    if a.cmd == "commit":
        problems += check_commit(a.sha, a.stage, repo=a.repo, main_ref=a.main_ref)
    else:
        p, notes = check_tree(a.path, a.stage, repo=a.repo, main_ref=a.main_ref,
                              fallback_commit=a.fallback_commit)
        problems += p
    for n in notes:
        print(n)
    for p in problems:
        print("FAIL: " + p)
    print("code check: " + ("PASS" if not problems else f"FAIL ({len(problems)})"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
