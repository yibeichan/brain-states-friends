"""Provenance records for stage outputs.

A stage that reads BOLD-derived inputs writes ``provenance_<stage>.json`` next
to its outputs. The record says which code produced them, from which output
root, and exactly which input files were read. With it, two versions of an
output can be told apart without comparing numbers. It is also what
``check_code.py`` and ``promote.py`` check before anything enters the output
dataset (see script/utils/output_provenance.md).

Use:
    from utils import provenance
    ...                                    # load inputs through the loaders
    provenance.write_provenance(out_dir, "05f", sub_id, parcellation)

Loaders that resolve a run to a file call ``record_input``; ``pick_run_session``
records every resolution. Call ``write_provenance`` only after the outputs
were recomputed, never on a checkpoint skip, or old files get a new stamp.

The record lists every output it describes with its size and md5
(``outputs``): by default every file under ``out_dir`` written since this
process started, or the files named in ``covers``. A file not listed, or
listed with another checksum (a crashed or partial rerun), is not covered,
so the checks refuse it. md5 is used because the output dataset's annex keys
(MD5E) carry it, so dataset files can be checked without reading them.
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCHEMA_VERSION = 2
_REPO_DIR = Path(__file__).resolve().parents[2]
_T0 = time.time()          # process start: outputs written after it belong to this run

# Per-process logs, reset by reset() (tests) and read by write_provenance().
_inputs: dict = {}
_sessions: dict = {}
_runs: dict = {}


def _git(repo, *args):
    try:
        p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    except OSError as e:                      # no git on this node: record, don't crash the stage
        return 127, str(e)
    return p.returncode, p.stdout.strip()


def code_state(repo_dir=None) -> dict:
    """Commit, branch and cleanliness of the checkout the code runs from.

    ``dirty`` is true when a tracked file differs from HEAD or an untracked
    ``.py`` file sits under ``script/``: either way the commit alone does not
    describe the code that ran.
    """
    repo = Path(repo_dir) if repo_dir else _REPO_DIR
    rc, commit = _git(repo, "rev-parse", "HEAD")
    if rc != 0:
        return {"commit": None, "branch": None, "dirty": True, "repo": str(repo),
                "error": "not a git checkout"}
    _, branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    _, tracked = _git(repo, "status", "--porcelain", "--untracked-files=no")
    _, untracked = _git(repo, "ls-files", "--others", "--exclude-standard", "--", "script")
    untracked_py = [f for f in untracked.splitlines() if f.endswith(".py")]
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(tracked) or bool(untracked_py),
        "modified": tracked.splitlines(),
        "untracked_py": untracked_py,
    }


_CODE_AT_IMPORT = code_state()


def reset():
    """Clear the per-process logs (for tests, or a script looping over subjects)."""
    _inputs.clear()
    _sessions.clear()
    _runs.clear()


def content_id(path):
    """(size, md5) of a file's content, or None if it cannot be determined.

    An annexed file (a symlink into the object store, present or not) is
    identified from its MD5E key without reading it; any other file is hashed.
    The same bytes therefore get the same identity whether they are annexed,
    linked in from the dataset, or a plain copy. Shared with check_code.py.
    """
    from utils.check_code import content_id as _cid

    return _cid(path)


def file_identity(path) -> dict:
    """Identity of an input file: its name plus content size and md5."""
    ident = {"name": Path(path).name}
    cid = content_id(path)
    if cid is None:
        ident["missing"] = True
    else:
        ident["size"], ident["md5"] = cid
    return ident


def record_input(kind, sub_id, path, run_id=None):
    """Log one input file read for ``sub_id`` (kind: e.g. 'parcel_ts', 'decoded_states').

    ``sub_id="*"`` marks an input shared by every participant (an atlas table).
    """
    key = (sub_id, kind, run_id or Path(path).name)
    _inputs[key] = file_identity(path)


def record_session(sub_id, run_id, session):
    """Log the acquisition session read for a run (any run, pinned or not)."""
    _sessions[(sub_id, run_id)] = session


def record_run(sub_id, run_id, **fields):
    """Log per-run facts such as n_ts, n_vit, n_used or skipped=reason."""
    _runs.setdefault((sub_id, run_id), {}).update(fields)


def _dataset_head(scratch_dir):
    if not scratch_dir:
        return None
    out = Path(scratch_dir) / "output"
    if not (out / ".git").exists():
        return None
    rc, head = _git(out, "rev-parse", "HEAD")
    return head if rc == 0 else None


def build_record(stage, sub_id, parcellation, outputs=None, extra=None) -> dict:
    """The provenance record for ``sub_id`` as of now (see write_provenance)."""
    from utils.common import DUPLICATE_RUN_SESSIONS

    now = code_state()
    scratch_dir = os.environ.get("SCRATCH_DIR")
    code = dict(now)
    code["commit_at_import"] = _CODE_AT_IMPORT.get("commit")
    code["head_changed"] = _CODE_AT_IMPORT.get("commit") != now.get("commit")
    # The code that ran was loaded at import: a tree dirty then is dirty, even
    # if the edit was reverted before the record was written.
    code["dirty_at_import"] = bool(_CODE_AT_IMPORT.get("dirty"))
    code["dirty_at_write"] = bool(now.get("dirty"))
    code["dirty"] = code["dirty_at_import"] or code["dirty_at_write"]
    code["modified_at_import"] = _CODE_AT_IMPORT.get("modified", [])
    code["untracked_py_at_import"] = _CODE_AT_IMPORT.get("untracked_py", [])
    record = {
        "schema": SCHEMA_VERSION,
        "stage": stage,
        "sub_id": sub_id,
        "parcellation": parcellation,
        "code": code,
        "scratch_dir": scratch_dir,
        "output_dataset_head": _dataset_head(scratch_dir),
        "duplicate_run_sessions_pinned": {
            run: ses for (sub, run), ses in sorted(DUPLICATE_RUN_SESSIONS.items()) if sub == sub_id
        },
        "sessions_read": {run: ses for (sub, run), ses in sorted(_sessions.items()) if sub == sub_id},
        "inputs": {f"{kind}:{name}": ident
                   for (sub, kind, name), ident in sorted(_inputs.items()) if sub in (sub_id, "*")},
        "runs": {run: f for (sub, run), f in sorted(_runs.items()) if sub == sub_id},
        "outputs": outputs if outputs is not None else {},
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "argv": list(sys.argv),
    }
    if extra:
        record["extra"] = extra
    return record


def provenance_name(stage) -> str:
    return f"provenance_{stage}.json"


def _is_record_name(name):
    return name.startswith("provenance_") and name.endswith(".json")


def collect_outputs(out_dir, covers=None, since=None) -> dict:
    """{relative path: {"size", "md5"}} for the files a record describes.

    ``covers`` names them explicitly (relative to out_dir; each must exist).
    Otherwise every regular file under out_dir modified at or after ``since``
    (default: process start) is taken, so files left from an earlier run are
    not covered by this run's record.
    """
    out_dir = Path(out_dir)
    if covers is not None:
        rels = list(covers)
        missing = [r for r in rels if not (out_dir / r).is_file()]
        if missing:
            raise FileNotFoundError(f"covers names files that do not exist: {missing}")
    else:
        t0 = (_T0 if since is None else since) - 1.0     # filesystem mtime granularity
        rels = []
        for root, dirs, names in os.walk(out_dir):
            dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(root, d))]
            for n in names:
                full = os.path.join(root, n)
                if _is_record_name(n) or os.path.islink(full):
                    continue
                try:
                    recent = os.stat(full).st_mtime >= t0
                except OSError:                       # vanished while walking
                    continue
                if recent:
                    rels.append(os.path.relpath(full, out_dir))
    outputs = {}
    for rel in sorted(rels):
        # An unreadable output is recorded as such (no checksum, so it is never
        # covered) instead of failing a stage that has already written everything.
        try:
            cid = content_id(out_dir / rel)
        except OSError as e:
            cid, err = None, str(e)
        else:
            err = "unreadable"
        outputs[rel] = {"size": cid[0], "md5": cid[1]} if cid else {"unreadable": err}
    return outputs


def write_provenance(out_dir, stage, sub_id, parcellation, covers=None, extra=None) -> Path:
    """Write ``out_dir/provenance_<stage>.json`` and return its path.

    ``covers`` lists the files (relative to out_dir) this record describes;
    pass it whenever the directory also holds another stage's files. Without
    it, the files under out_dir written since the process started are listed.
    """
    outputs = collect_outputs(out_dir, covers=covers)
    record = build_record(stage, sub_id, parcellation, outputs=outputs, extra=extra)
    path = Path(out_dir) / provenance_name(stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path
