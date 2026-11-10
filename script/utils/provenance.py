"""Provenance records for stage outputs.

A stage that reads BOLD-derived inputs writes ``provenance_<stage>.json`` next
to its outputs. The record says which code produced them, from which output
root, and exactly which input files were read. With it, two versions of an
output can be told apart without comparing numbers. It is also what
``check_code.py`` and ``promote.py`` check before anything enters the output
dataset (see docs/output_provenance.md).

Use:
    from utils import provenance
    ...                                    # load inputs through the loaders
    provenance.write_provenance(out_dir, "05f", sub_id, parcellation,
                                covers=["fc_similarity_corr_rv.npy", ...])

Loaders that resolve a run to a file call ``record_input``; ``pick_run_session``
records every resolution. Call ``write_provenance`` only after the outputs
were recomputed, never on a checkpoint skip, or old files get a new stamp.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SCHEMA_VERSION = 1
_REPO_DIR = Path(__file__).resolve().parents[2]
_ANNEX_KEY_RE = re.compile(r"^[A-Z0-9]+E?-s\d+(--|-m\d+--)\S+$")
_SHA256_MAX_BYTES = 64 * 1024 * 1024

# Per-process logs, reset by reset() (tests) and read by write_provenance().
_inputs: dict = {}
_sessions: dict = {}
_runs: dict = {}


def _git(repo, *args):
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
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


def file_identity(path) -> dict:
    """Identity of an input file without reading it when possible.

    An annexed file's real path ends in its annex key, which contains its
    checksum, so the key identifies the content for free. A plain file up to
    64 MB is hashed; a larger one is identified by size and mtime only.
    """
    p = Path(path)
    real = Path(os.path.realpath(p))
    ident = {"name": p.name}
    if _ANNEX_KEY_RE.match(real.name):
        ident["annex_key"] = real.name
        return ident
    try:
        st = real.stat()
    except OSError:
        ident["missing"] = True
        return ident
    ident["size"] = st.st_size
    if st.st_size <= _SHA256_MAX_BYTES:
        h = hashlib.sha256()
        with open(real, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        ident["sha256"] = h.hexdigest()
    else:
        ident["mtime"] = st.st_mtime
    return ident


def record_input(kind, sub_id, path, run_id=None):
    """Log one input file read for ``sub_id`` (kind: e.g. 'parcel_ts', 'decoded_states')."""
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


def build_record(stage, sub_id, parcellation, covers=None, extra=None) -> dict:
    """The provenance record for ``sub_id`` as of now (see write_provenance)."""
    from utils.common import DUPLICATE_RUN_SESSIONS

    now = code_state()
    scratch_dir = os.environ.get("SCRATCH_DIR")
    code = dict(now)
    code["commit_at_import"] = _CODE_AT_IMPORT.get("commit")
    code["head_changed"] = _CODE_AT_IMPORT.get("commit") != now.get("commit")
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
                   for (sub, kind, name), ident in sorted(_inputs.items()) if sub == sub_id},
        "runs": {run: f for (sub, run), f in sorted(_runs.items()) if sub == sub_id},
        "covers": sorted(covers) if covers else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "argv": list(sys.argv),
    }
    if extra:
        record["extra"] = extra
    return record


def provenance_name(stage) -> str:
    return f"provenance_{stage}.json"


def write_provenance(out_dir, stage, sub_id, parcellation, covers=None, extra=None) -> Path:
    """Write ``out_dir/provenance_<stage>.json`` and return its path.

    ``covers`` lists the files (relative to out_dir) this record describes;
    pass it whenever the directory also holds another stage's files.
    """
    record = build_record(stage, sub_id, parcellation, covers=covers, extra=extra)
    path = Path(out_dir) / provenance_name(stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path
