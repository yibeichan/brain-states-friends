#!/usr/bin/env python3
"""Promote a side root's outputs into the output dataset, with gates.

A rerun writes to a side root (``<side>/output/...``), never in place. This
tool copies the files a manifest declares as changing into the output
dataset and saves them, only after every gate passes:

* Declared change set. Every recomputed side-root file outside the
  ``change`` globs must equal the dataset's copy (numerically, within
  tolerance). An undeclared difference stops the promotion.
* Controls that can fail. At least one control must be a recomputed file
  inside the change set, compared with a reference version. At least one
  must be a positive control (``expect = "differ"``), unless the manifest
  says why none applies.
* Code. ``code_commit`` and every provenance record copied in must pass
  check_code.py: on main, contains the code floor, clean, and duplicated
  runs read from the pinned session. A record's input log must equal the
  dataset's record for the same directory unless the manifest allows it.

Dry run by default; ``--execute`` reruns every check, then copies, saves
(``datalad save``, no push) and verifies the saved content.

Manifest (TOML):

    message = "sub-06 05f recomputed on the pinned duplicate run"
    code_commit = "<sha the side root ran>"
    change = ["05f_state_fc/atlas-4S156Parcels/sub-06/**"]
    # optional
    scope = ["**"]                       # side-root files to consider
    exclude = ["logs/**"]
    allow_input_changes = ""             # reason, if input logs may differ
    no_positive_control = ""             # reason, if no control can be expected to differ
    [tolerance]                          # defaults: rtol 1e-9, atol 1e-12
    rtol = 1e-9
    atol = 1e-12
    [[override]]                         # per-file tolerance or ignored JSON keys
    glob = "**/metadata.json"
    ignore_keys = ["timestamp"]
    [[control]]
    path = "05f_state_fc/atlas-4S156Parcels/sub-06/vt0.95/fc_similarity_corr_rv.npy"
    reference = "11d867cf6c3"            # dataset commit, or "HEAD"
    expect = "equal"                     # or "differ" (needs min_diff for numeric files)
    why = "05f loader path: sub-06 s02e04b must read ses-006"

See script/utils/output_provenance.md.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import fnmatch
import datetime
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from utils import check_code  # noqa: E402

csv.field_size_limit(sys.maxsize)   # whole-row fields of any length
FIGURE_EXT = {".png", ".pdf", ".svg", ".jpg", ".jpeg"}
DEFAULT_EXCLUDE = ["logs/**", "**/__pycache__/**", "**/.nfs*"]
SAVE_CHUNK = 500   # paths per datalad save call


# ------------------------------------------------------------------ globs
def glob_to_regex(pattern: str) -> re.Pattern:
    """Path glob: ``**`` spans zero or more directories, ``*`` and ``?`` stay
    within one path segment. Matches the whole relative path."""
    segs = pattern.strip("/").split("/")
    out = []
    for i, seg in enumerate(segs):
        last = i == len(segs) - 1
        if seg == "**":
            out.append(".*" if last else "(?:[^/]+/)*")
            continue
        part = ""
        for ch in seg:
            part += "[^/]*" if ch == "*" else "[^/]" if ch == "?" else re.escape(ch)
        out.append(part if last else part + "/")
    return re.compile("".join(out) + r"\Z")


def matches(rel: str, patterns) -> bool:
    return any(glob_to_regex(p).match(rel) for p in patterns)


# ------------------------------------------------------------- comparison
@dataclass
class Cmp:
    """Result of comparing a file with its reference.

    ``kind`` is 'numeric' when the two have the same structure and only
    numeric values were compared (``max_diff`` is then the largest absolute
    difference over every numeric leaf), 'structural' when shape, dtype, keys,
    NaN positions, strings or bytes differ, and 'error' when the files could
    not be read. Only a 'numeric' result can pass a positive control.
    """
    equal: bool
    detail: str = ""
    max_diff: float = 0.0
    kind: str = "numeric"


def _structural(detail):
    return Cmp(False, detail, math.inf, "structural")


def _num_leaf(a, b, rtol, atol):
    """(ok, |diff| or None for a structural mismatch) for two numbers."""
    fa, fb = float(a), float(b)
    if math.isnan(fa) or math.isnan(fb):
        return (math.isnan(fa) and math.isnan(fb)), (0.0 if math.isnan(fa) and math.isnan(fb) else None)
    if math.isinf(fa) or math.isinf(fb):
        return (fa == fb), (0.0 if fa == fb else None)
    d = abs(fa - fb)
    return d <= atol + rtol * abs(fb), d


def _close_arrays(a, b, rtol, atol, label=""):
    if a.shape != b.shape:
        return _structural(f"{label}shape {a.shape} vs {b.shape}")
    ka, kb = a.dtype.kind, b.dtype.kind
    num = "biufc"
    if ka not in num or kb not in num:
        ok = a.dtype == b.dtype and bool(np.array_equal(a, b))
        return Cmp(True, f"{label}equal ({a.dtype})") if ok else _structural(f"{label}{a.dtype} arrays differ")
    # bool, integer, float and complex are different kinds of quantity
    group = lambda k: {"b": "b", "i": "i", "u": "i", "f": "f", "c": "c"}[k]
    if group(ka) != group(kb):
        return _structural(f"{label}dtype {a.dtype} vs {b.dtype}")
    if group(ka) in "bi":
        if np.array_equal(a, b):
            return Cmp(True, f"{label}integer arrays equal")
        md = 1.0 if ka == "b" else float(np.abs(a.astype(np.int64) - b.astype(np.int64)).max())
        return Cmp(False, f"{label}integer arrays differ (max |diff| {md:g})", md)
    na, nb = np.isnan(a), np.isnan(b)
    if not np.array_equal(na, nb):
        return _structural(f"{label}NaN positions differ ({int((na != nb).sum())} entries)")
    ia, ib = np.isinf(a), np.isinf(b)
    if not np.array_equal(ia, ib) or not np.array_equal(a[ia], b[ib]):
        return _structural(f"{label}inf entries differ")
    fin = ~(na | ia)
    if not fin.any():
        return Cmp(True, f"{label}no finite entries")
    d = np.abs(a[fin] - b[fin])
    md = float(d.max())
    ok = bool(np.all(d <= atol + rtol * np.abs(b[fin])))
    return Cmp(ok, f"{label}max |diff| {md:.3g}", md)


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _json_walk(a, b, rtol, atol, ignore, path, acc):
    """Visit every leaf; acc collects (structural detail list, max diff, worst failing leaf)."""
    if isinstance(a, dict) and isinstance(b, dict):
        ka, kb = set(a) - ignore, set(b) - ignore
        if ka != kb:
            acc["structural"].append(f"{path}: keys differ ({sorted(ka ^ kb)[:5]})")
        for k in sorted(ka & kb):
            _json_walk(a[k], b[k], rtol, atol, ignore, f"{path}.{k}", acc)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            acc["structural"].append(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            _json_walk(x, y, rtol, atol, ignore, f"{path}[{i}]", acc)
        return
    if _is_num(a) and _is_num(b):
        if isinstance(a, int) != isinstance(b, int):
            acc["structural"].append(f"{path}: {a!r} vs {b!r} (int vs float)")
            return
        ok, d = _num_leaf(a, b, rtol, atol)
        if d is None:
            acc["structural"].append(f"{path}: {a} vs {b}")
            return
        acc["max"] = max(acc["max"], d)
        if not ok and acc["fail"] is None:
            acc["fail"] = f"{path}: {a} vs {b}"
        return
    if type(a) is not type(b) or a != b:
        acc["structural"].append(f"{path}: {str(a)[:40]!r} vs {str(b)[:40]!r}")


def _from_acc(acc):
    if acc["structural"]:
        return _structural(f"{acc['structural'][0]} ({len(acc['structural'])} structural difference(s))")
    if acc["fail"]:
        return Cmp(False, f"{acc['fail']}; max |diff| {acc['max']:.3g}", acc["max"])
    return Cmp(True, f"max |diff| {acc['max']:.3g}", acc["max"])


def _close_json(a, b, rtol, atol, ignore):
    """JSON: dict key sets exact (order ignored), list order kept, bool and
    str exact, int vs float a structural difference, numbers within tolerance."""
    acc = {"structural": [], "max": 0.0, "fail": None}
    _json_walk(a, b, rtol, atol, set(ignore), "$", acc)
    return _from_acc(acc)


def _float_or_none(s):
    try:
        return float(s)
    except ValueError:
        return None


def _close_table(pa, pb, delim, rtol, atol):
    with open(pa, newline="") as fa, open(pb, newline="") as fb:
        ra, rb = list(csv.reader(fa, delimiter=delim)), list(csv.reader(fb, delimiter=delim))
    if not ra or not rb:
        return Cmp(True, "empty tables") if ra == rb else _structural("one table is empty")
    if ra[0] != rb[0]:
        return _structural(f"header differs: {ra[0][:6]} vs {rb[0][:6]}")
    if len(ra) != len(rb):
        return _structural(f"rows {len(ra) - 1} vs {len(rb) - 1}")
    acc = {"structural": [], "max": 0.0, "fail": None}
    for i, (x, y) in enumerate(zip(ra[1:], rb[1:]), 1):
        if len(x) != len(y):
            acc["structural"].append(f"row {i}: {len(x)} vs {len(y)} cells")
            continue
        for col, u, v in zip(ra[0], x, y):
            fu, fv = _float_or_none(u), _float_or_none(v)
            if fu is None or fv is None:
                if u != v:
                    acc["structural"].append(f"row {i} {col}: {u!r} vs {v!r}")
                continue
            ok, d = _num_leaf(fu, fv, rtol, atol)
            if d is None:
                acc["structural"].append(f"row {i} {col}: {u} vs {v}")
                continue
            acc["max"] = max(acc["max"], d)
            if not ok and acc["fail"] is None:
                acc["fail"] = f"row {i} {col}: {u} vs {v}"
    return _from_acc(acc)


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_files(a, b, rtol=1e-9, atol=1e-12, ignore_keys=()):
    """Compare side-root file ``a`` with reference ``b`` by file type."""
    a, b = Path(a), Path(b)
    ext = a.suffix.lower()
    try:
        if ext == ".npy":
            return _close_arrays(np.load(a, allow_pickle=False), np.load(b, allow_pickle=False), rtol, atol)
        if ext == ".npz":
            with np.load(a, allow_pickle=False) as za, np.load(b, allow_pickle=False) as zb:
                if set(za.files) != set(zb.files):
                    return _structural(f"npz keys differ ({sorted(set(za.files) ^ set(zb.files))})")
                results = [_close_arrays(za[k], zb[k], rtol, atol, label=f"{k}: ") for k in sorted(za.files)]
            bad = [r for r in results if r.kind != "numeric"]
            if bad:
                return bad[0]
            md = max((r.max_diff for r in results), default=0.0)
            fail = next((r for r in results if not r.equal), None)
            return Cmp(fail is None, fail.detail if fail else f"max |diff| {md:.3g}", md)
        if ext == ".json":
            return _close_json(json.loads(a.read_text()), json.loads(b.read_text()), rtol, atol, ignore_keys)
        if ext in (".csv", ".tsv"):
            return _close_table(a, b, "\t" if ext == ".tsv" else ",", rtol, atol)
    except (ValueError, csv.Error) as e:
        # object arrays, malformed tables: fall back to comparing bytes
        try:
            ok = md5(a) == md5(b)
        except OSError as e2:
            return Cmp(False, f"cannot compare: {e2}", math.inf, "error")
        return (Cmp(True, f"bytes equal (not parsed: {e})") if ok
                else _structural(f"bytes differ (not parsed: {e})"))
    except OSError as e:
        return Cmp(False, f"cannot compare: {e}", math.inf, "error")
    ok = md5(a) == md5(b)
    return Cmp(True, "bytes equal") if ok else _structural("bytes differ")


# ------------------------------------------------------- dataset reading
def _run(cmd, cwd=None, check=False, text=True):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=text)
    if check and p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {p.stderr.strip()}")
    return p


def annex_key_location(dataset, key):
    """Local path of annexed content ``key``, or None when it is not present.

    Asks git-annex when it is on PATH; otherwise looks in the object store,
    whose two hash directory levels hold <key>/<key> in every hashing layout.
    """
    if shutil.which("git-annex"):
        p = _run(["git", "annex", "contentlocation", key], cwd=dataset)
        if p.returncode == 0 and p.stdout.strip():
            loc = Path(dataset) / p.stdout.strip()
            return loc if loc.exists() else None
    git_dir = _run(["git", "rev-parse", "--absolute-git-dir"], cwd=dataset).stdout.strip()
    hits = sorted(Path(git_dir, "annex", "objects").glob(f"*/*/{key}/{key}")) if git_dir else []
    return hits[0] if hits else None


def reference_file(dataset, commit, rel, tmpdir):
    """Local path holding ``rel`` as of dataset ``commit``.

    Locked annex files are symlinks in git (mode 120000) whose target ends in
    the annex key; unlocked ones are blobs holding ``/annex/objects/<key>``.
    Plain git files are written out from the blob. Raises when the content is
    not present locally (``datalad get`` it first).
    """
    p = _run(["git", "ls-tree", commit, "--", rel], cwd=dataset)
    line = p.stdout.strip()
    if p.returncode != 0 or not line:
        raise FileNotFoundError(f"{rel} not in dataset commit {commit}")
    meta, _ = line.split("\t", 1)
    mode, _otype, obj = meta.split()
    blob = _run(["git", "cat-file", "blob", obj], cwd=dataset, check=True, text=False).stdout
    key = None
    if mode == "120000":
        key = os.path.basename(blob.decode().strip())
    elif blob.startswith(b"/annex/objects/"):
        key = os.path.basename(blob.decode().strip())
    if key is None:
        out = Path(tmpdir) / f"{obj}{Path(rel).suffix}"
        out.write_bytes(blob)
        return out
    loc = annex_key_location(dataset, key)
    if loc is None:
        raise FileNotFoundError(f"annex content {key} for {rel}@{commit} is not present; datalad get it first")
    return loc


# ------------------------------------------------------------------ plan
@dataclass
class Report:
    fails: list = field(default_factory=list)
    warns: list = field(default_factory=list)
    infos: list = field(default_factory=list)

    def fail(self, m): self.fails.append(m)
    def warn(self, m): self.warns.append(m)
    def info(self, m): self.infos.append(m)


@dataclass
class Plan:
    copies: list = field(default_factory=list)       # (action, rel)
    md5: dict = field(default_factory=dict)          # rel -> md5 of the side file as checked
    report: Report = field(default_factory=Report)


def _has_wildcard(seg):
    return any(ch in seg for ch in "*?[")


def load_manifest(path):
    with open(path, "rb") as fh:
        m = tomllib.load(fh)
    for k in ("message", "code_commit", "change"):
        if not m.get(k):
            raise ValueError(f"manifest {path}: '{k}' is required")
    for g in m["change"]:
        # Breadth is judged by what a glob can match: it must name the stage
        # directory literally, so no glob can declare several stages at once.
        first = g.strip("/").split("/")[0]
        if not first or _has_wildcard(first):
            raise ValueError(f"manifest {path}: change glob {g!r} must start with a literal stage "
                             "directory (e.g. '05f_state_fc/...'); list the paths")
    aic = m.get("allow_input_changes")
    if aic is not None and not (isinstance(aic, dict) and all(isinstance(v, str) and v for v in aic.values())):
        raise ValueError(f"manifest {path}: allow_input_changes must be a table of input-key "
                         "patterns to reasons, e.g. {\"parcel_ts:s02e04b\" = \"02 rerun\"}")
    return m


def _tol(manifest, rel):
    t = manifest.get("tolerance", {})
    rtol, atol, ignore = float(t.get("rtol", 1e-9)), float(t.get("atol", 1e-12)), []
    for o in manifest.get("override", []):
        if matches(rel, [o["glob"]]):
            rtol = float(o.get("rtol", rtol))
            atol = float(o.get("atol", atol))
            if "decimals" in o:
                # values rounded to d decimals may move by one unit in the last place
                atol = max(atol, 1.5 * 10.0 ** -int(o["decimals"]))
            ignore += list(o.get("ignore_keys", []))
    return rtol, atol, ignore


def side_files(side_out, scope, exclude):
    """Recomputed side-root files (symlinks are inputs linked in, not outputs).

    Returns (files, number of symlinks skipped, symlinked directories)."""
    files, linked, linked_dirs = [], 0, []
    for root, dirs, names in os.walk(side_out):
        keep = []
        for d in dirs:
            if os.path.islink(os.path.join(root, d)):
                linked += 1
                linked_dirs.append(os.path.relpath(os.path.join(root, d), side_out))
            else:
                keep.append(d)
        dirs[:] = keep
        for n in names:
            full = os.path.join(root, n)
            rel = os.path.relpath(full, side_out)
            if os.path.islink(full):
                linked += 1
                continue
            if matches(rel, scope) and not matches(rel, exclude):
                files.append(rel)
    return sorted(files), linked, linked_dirs


def _is_record(rel):
    n = os.path.basename(rel)
    return n.startswith("provenance_") and n.endswith(".json")


def _under(rel, d):
    return d == "" or rel == d or rel.startswith(d.rstrip("/") + "/")


gated_stage = check_code.gated_stage


def _norm_ident(ident):
    """Content identity of an input-log entry, comparable across storage forms."""
    if not isinstance(ident, dict):
        return ("?", repr(ident))
    if ident.get("md5"):
        return ("md5", ident.get("size"), ident["md5"])
    key = ident.get("annex_key")                          # schema-1 records
    m = check_code._MD5_KEY_RE.match(key or "")
    if m:
        return ("md5", int(m.group(1)), m.group(2))
    if ident.get("sha256"):
        return ("sha256", ident.get("size"), ident["sha256"])
    return ("?", repr(sorted(ident.items())))


def input_differences(prev, rec):
    """Input-log keys whose content differs between two records (sessions as 'session:<run>')."""
    a_ = {k: _norm_ident(v) for k, v in (prev.get("inputs") or {}).items()}
    b_ = {k: _norm_ident(v) for k, v in (rec.get("inputs") or {}).items()}
    diff = [k for k in set(a_) | set(b_) if a_.get(k) != b_.get(k)]
    sa, sb = prev.get("sessions_read") or {}, rec.get("sessions_read") or {}
    diff += [f"session:{r}" for r in set(sa) | set(sb) if sa.get(r) != sb.get(r)]
    return sorted(diff)


def build_plan(side_root, dataset, manifest, repo=REPO_DIR, main_ref="origin/main",
               strict_figures=False):
    side_out = Path(side_root) / "output"
    dataset = Path(dataset)
    plan = Plan()
    rep = plan.report
    if not side_out.is_dir():
        rep.fail(f"no output/ under side root {side_root}")
        return plan

    # dataset must be clean so its working tree is its HEAD
    st = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=dataset)
    if st.returncode != 0:
        rep.fail(f"{dataset} is not a git dataset")
        return plan
    if st.stdout.strip():
        rep.fail(f"dataset has uncommitted changes ({len(st.stdout.splitlines())} paths); save or revert them first")

    scope = manifest.get("scope", ["**"])
    exclude = DEFAULT_EXCLUDE + list(manifest.get("exclude", []))
    change = manifest["change"]
    files, linked, linked_dirs = side_files(side_out, scope, exclude)
    rep.info(f"{len(files)} recomputed side-root files in scope ({linked} symlinked inputs skipped)")
    # A stage run in the side root can write through a linked input directory
    # straight into the dataset (04rc writes next to its inputs).
    ds_real = os.path.realpath(dataset)
    for d in linked_dirs:
        real = os.path.realpath(side_out / d)
        if real == ds_real or real.startswith(ds_real + os.sep):
            st = _run(["git", "status", "--porcelain", "--", os.path.relpath(real, ds_real)], cwd=dataset)
            if st.stdout.strip():
                rep.fail(f"linked input directory {d} has uncommitted or new files in the dataset: "
                         "a stage wrote through the link; link files, not directories, for stages "
                         "that write next to their inputs")

    for g in change:
        n = sum(1 for r in files if matches(r, [g]))
        rep.info(f"change glob {g!r}: {n} file(s)")
        if n == 0:
            rep.fail(f"change glob {g!r} matches no recomputed file")

    # dataset files in the change set that the side root no longer produces
    side_set = set(files)
    ls = _run(["git", "ls-files", "-z"], cwd=dataset)
    gone = [r for r in ls.stdout.split("\0") if r and matches(r, change)
            and matches(r, scope) and not matches(r, exclude) and r not in side_set]
    for r in gone[:20]:
        msg = f"in the change set but not reproduced by the side root: {r}"
        if manifest.get("keep_unreproduced"):
            rep.warn(msg + f" [kept: {manifest['keep_unreproduced']}]")
        else:
            rep.fail(msg)
    if len(gone) > 20:
        (rep.warn if manifest.get("keep_unreproduced") else rep.fail)(
            f"... and {len(gone) - 20} more dataset files in the change set not reproduced")

    records = []
    for rel in files:
        if _is_record(rel):
            records.append(rel)
            continue
        src, dst = side_out / rel, dataset / rel
        in_change = matches(rel, change)
        if in_change:
            plan.md5[rel] = md5(src)                 # the content the gates below judge
        if not os.path.lexists(dst):
            if in_change:
                plan.copies.append(("NEW", rel))
            else:
                rep.fail(f"undeclared new file: {rel}")
            continue
        if not dst.exists():
            rep.fail(f"dataset content not present for {rel}; datalad get it first")
            continue
        rtol, atol, ignore = _tol(manifest, rel)
        c = compare_files(src, dst, rtol, atol, ignore)
        if in_change:
            if c.equal:
                rep.info(f"declared change but equal to dataset: {rel} ({c.detail})")
            else:
                plan.copies.append(("UPDATE", rel))
            continue
        if c.equal:
            continue
        if Path(rel).suffix.lower() in FIGURE_EXT:
            (rep.fail if strict_figures else rep.warn)(f"figure outside the change set differs: {rel}")
        else:
            rep.fail(f"unexpected change outside the change set: {rel} ({c.detail})")

    # code
    planned = [r for _, r in plan.copies]
    rc = _run(["git", "rev-parse", "--verify", "--quiet", f"{manifest['code_commit']}^{{commit}}"], cwd=repo)
    code_full = rc.stdout.strip() if rc.returncode == 0 else None
    for p in check_code.check_commit(manifest["code_commit"], None, repo=repo, main_ref=main_ref):
        rep.fail(f"code_commit: {p}")

    # coverage: each planned output of a provenance stage must be listed, with
    # the checksum it has now, in its nearest record in the side root
    cache, needed, weak_tops = {}, set(), set()
    for rel in planned:
        stage = gated_stage(rel)
        if stage is None:
            weak_tops.add(Path(rel).parts[0])
            continue
        why = check_code.coverage_problem(side_out / rel, stage, cache)
        if why:
            rep.fail(f"{rel}: {why}")
        rp = check_code.nearest_record(side_out / rel, stage)
        if rp is not None:
            needed.add(os.path.relpath(rp, side_out))
    for rel in records:
        if rel not in needed:
            rep.info(f"provenance record left out (no planned output relies on it): {rel}")
    if weak_tops:
        msg = (f"stages without provenance records, accepted on the manifest's code_commit only: "
               f"{sorted(weak_tops)}")
        if manifest.get("weak_stages"):
            rep.warn(msg + f" [weak_stages: {manifest['weak_stages']}]")
        else:
            rep.fail(msg + "; set weak_stages = \"<reason>\" to accept this")

    allow_inputs = manifest.get("allow_input_changes") or {}
    for rel in sorted(needed):
        rec, err = check_code._load_record(side_out / rel)
        if err:
            rep.fail(f"{rel}: {err}")
            continue
        for p in check_code.check_record(rec, repo=repo, main_ref=main_ref):
            rep.fail(f"{rel}: {p}")
        if code_full and (rec.get("code") or {}).get("commit") != code_full:
            rep.fail(f"{rel}: record names code {str((rec.get('code') or {}).get('commit'))[:10]}, "
                     f"manifest code_commit is {code_full[:10]}")
        old = dataset / rel
        if os.path.lexists(old) and not old.exists():
            rep.fail(f"{rel}: the dataset's record is not present locally; datalad get it first")
        elif old.exists():
            prev, err = check_code._load_record(old)
            if err:
                rep.fail(f"{rel} in the dataset: {err}")
                prev = {}
            for k in input_differences(prev, rec):
                why = next((r for pat, r in allow_inputs.items() if fnmatch.fnmatchcase(k, pat)), None)
                if why:
                    rep.warn(f"{rel}: input {k} differs from the dataset's record [allowed: {why}]")
                else:
                    rep.fail(f"{rel}: input {k} differs from the dataset's record "
                             "(waive it with allow_input_changes = {\"<pattern>\" = \"<reason>\"})")
        plan.md5[rel] = md5(side_out / rel)
        plan.copies.append(("UPDATE" if os.path.lexists(dataset / rel) else "NEW", rel))

    # files the dataset's .gitignore matches would be copied but never saved
    rels_all = [r for _, r in plan.copies]
    if rels_all:
        ci = subprocess.run(["git", "check-ignore", "--stdin", "-z"], cwd=dataset,
                            input="\0".join(rels_all) + "\0", capture_output=True, text=True)
        for r in [x for x in ci.stdout.split("\0") if x][:20]:
            rep.fail(f"{r}: the dataset's .gitignore matches it, so it would be copied but not saved")

    check_controls(manifest, side_out, dataset, files, rep)
    return plan


def check_controls(manifest, side_out, dataset, files, rep):
    controls = manifest.get("control", [])
    if not controls:
        rep.fail("no [[control]]: a promotion needs at least one control that can fail")
        return
    fileset = set(files)
    change = manifest["change"]
    if not any(matches(c.get("path", ""), change) for c in controls):
        rep.fail("no control lies inside the change set; a control outside it cannot detect a wrong change")
    if not any(c.get("expect", "equal") == "differ" and matches(c.get("path", ""), change)
               for c in controls) and not manifest.get("no_positive_control"):
        rep.fail("no positive control (expect = \"differ\") inside the change set; "
                 "add one or set no_positive_control = \"<reason>\"")
    with tempfile.TemporaryDirectory() as tmp:
        for c in controls:
            rel, ref = c.get("path", ""), c.get("reference", "HEAD")
            expect = c.get("expect", "equal")
            tag = f"control {rel} vs {ref} (expect {expect})"
            if not c.get("why"):
                rep.fail(f"{tag}: 'why' is required (name the code path it exercises)")
            if rel not in fileset:
                rep.fail(f"{tag}: not a file this side root recomputed, so it cannot fail")
                continue
            try:
                ref_path = reference_file(dataset, ref, rel, tmp)
            except (FileNotFoundError, RuntimeError, ValueError) as e:
                rep.fail(f"{tag}: {e}")
                continue
            rtol, atol, ignore = _tol(manifest, rel)
            rtol = float(c.get("rtol", rtol))
            atol = float(c.get("atol", c.get("tol", atol)))
            ignore = ignore + list(c.get("ignore_keys", []))
            cmp_ = compare_files(side_out / rel, ref_path, rtol, atol, ignore)
            if expect == "equal":
                (rep.info if cmp_.equal else rep.fail)(f"{tag}: {'PASS' if cmp_.equal else 'FAIL'} ({cmp_.detail})")
            elif expect == "differ":
                # A positive control passes only on a numeric difference of at
                # least min_diff with the structure intact: a broken, reshaped
                # or all-NaN recompute must not count as "the expected change".
                if "min_diff" not in c:
                    rep.fail(f"{tag}: min_diff is required for expect = \"differ\"")
                    continue
                min_diff = float(c["min_diff"])
                if min_diff <= atol:
                    rep.fail(f"{tag}: min_diff {min_diff:g} must exceed atol {atol:g}")
                    continue
                ok = (not cmp_.equal) and cmp_.kind == "numeric" and cmp_.max_diff >= min_diff
                why_not = "" if ok else (f" [{cmp_.kind} difference, not a numeric one]" if cmp_.kind != "numeric" else "")
                (rep.info if ok else rep.fail)(
                    f"{tag}: {'PASS' if ok else 'FAIL'} ({cmp_.detail}; need max |diff| >= {min_diff:g}){why_not}")
            else:
                rep.fail(f"{tag}: expect must be 'equal' or 'differ'")


# --------------------------------------------------------------- execute
def datalad_cmd(repo):
    env = os.environ.get("PROMOTE_DATALAD")
    if env:
        return shlex.split(env)
    return ["uv", "run", "--project", str(repo), "--extra", "datalad", "datalad"]


def _head(dataset):
    return _run(["git", "rev-parse", "HEAD"], cwd=dataset).stdout.strip()


def execute(plan, side_root, dataset, manifest, manifest_path, repo, log, main_ref="origin/main"):
    side_out = Path(side_root) / "output"
    dataset = Path(dataset)
    rels = [r for _, r in plan.copies]
    lock_path = Path(_run(["git", "rev-parse", "--absolute-git-dir"], cwd=dataset).stdout.strip()) / "promote.lock"
    lock = open(lock_path, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log(f"another promotion holds {lock_path}; nothing was changed")
        return 1
    head0 = _head(dataset)
    new_files = [r for a_, r in plan.copies if a_ == "NEW"]
    upd_files = [r for a_, r in plan.copies if a_ == "UPDATE"]

    def recovery(reason):
        log(f"PROMOTION FAILED: {reason}")
        log(f"  dataset HEAD before: {head0}; now: {_head(dataset)}")
        log("  to undo by hand: remove the NEW files and restore the UPDATE files with "
            "'git checkout HEAD -- <path>' (lists below); if HEAD moved, inspect the commits "
            f"after {head0[:10]} with 'git log {head0[:10]}..HEAD' before deciding.")
        for r in new_files:
            log(f"  NEW {r}")
        for r in upd_files:
            log(f"  UPDATE {r}")
        return 1

    try:
        for act, rel in plan.copies:
            src, dst = side_out / rel, dataset / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.parent / f".promote-{os.getpid()}-{dst.name}"
            shutil.copy2(src, tmp)
            if md5(tmp) != plan.md5.get(rel):
                tmp.unlink()
                return recovery(f"{rel} changed in the side root after it was checked")
            os.chmod(tmp, os.stat(tmp).st_mode | 0o200)
            if os.path.lexists(dst):
                os.remove(dst)                      # annex symlink or plain file
            os.replace(tmp, dst)
            log(f"{act} {rel}")
    except OSError as e:
        return recovery(f"copy failed: {e}")
    floor = ",".join(sh[:10] for sh, _, _ in check_code.load_floor(repo=repo, main_ref=main_ref))
    msg = (f"{manifest['message']}\n\n"
           f"code_commit: {manifest['code_commit']}\n"
           f"code_floor: {floor}\n"
           f"side_root: {side_root}\n"
           f"manifest_sha256: {hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()}\n"
           f"promote_check: passed")
    # One commit: save in chunks, amending the promotion's own commit after the
    # first (keeps the command line far below ARG_MAX). Before each amend, HEAD
    # must still be that commit, or another job committed in between.
    expected = head0
    for i in range(0, len(rels), SAVE_CHUNK):
        part = rels[i:i + SAVE_CHUNK]
        if i and _head(dataset) != expected:
            return recovery("HEAD moved during the save (another job committed); not amending its commit")
        cmd = datalad_cmd(repo) + ["save", "-m", msg] + (["--amend"] if i else []) + ["--"] + part
        try:
            p = _run(cmd, cwd=dataset)
        except OSError as e:
            return recovery(f"datalad save failed to start: {e}")
        log(p.stdout + p.stderr)
        if p.returncode != 0:
            return recovery(f"datalad save failed (rc {p.returncode}) on paths {i}..{i + len(part) - 1}")
        expected = _head(dataset)
        if expected == head0:
            return recovery("datalad save made no commit")
        # the promotion's commit must sit directly on head0; anything else means
        # another job committed while this save ran
        if _run(["git", "rev-parse", "HEAD^"], cwd=dataset).stdout.strip() != head0:
            return recovery("HEAD moved during the save (another job committed); not amending its commit")
    # verify: one commit on top of head0, every path saved and clean, content as checked
    if _run(["git", "rev-parse", "HEAD^"], cwd=dataset).stdout.strip() != head0:
        return recovery("the promotion is not a single commit on top of the previous HEAD")
    tops = sorted({Path(r).parts[0] for r in rels})
    st = _run(["git", "status", "--porcelain", "--untracked-files=all", "-z", "--"] + tops, cwd=dataset)
    left = {e[3:] for e in st.stdout.split("\0") if len(e) > 3}
    unsaved = [r for r in rels if r in left]
    ls = _run(["git", "ls-files", "-z", "--"] + tops, cwd=dataset)
    tracked = set(ls.stdout.split("\0"))
    unsaved += [r for r in rels if r not in tracked and r not in unsaved]
    bad = [r for r in rels if (check_code.content_id(dataset / r) or (None, None))[1] != plan.md5.get(r)]
    for r in unsaved:
        log(f"VERIFY FAIL: not saved: {r}")
    for r in bad:
        log(f"VERIFY FAIL: saved content differs from the checked side-root file: {r}")
    head = _run(["git", "log", "-1", "--format=%h %s"], cwd=dataset).stdout.strip()
    log(f"output dataset HEAD: {head}")
    if unsaved or bad:
        return recovery("verification failed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side-root", required=True, help="directory holding the rerun's output/")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dataset", help="output dataset (default: $SCRATCH_DIR/output from .env)")
    ap.add_argument("--repo", default=str(REPO_DIR))
    ap.add_argument("--main-ref", default="origin/main")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--strict-figures", action="store_true",
                    help="fail, not warn, on a differing figure outside the change set")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args(argv)

    dataset = a.dataset or _dataset_from_env(a.repo)
    manifest = load_manifest(a.manifest)
    if not a.no_fetch:
        ok, note = check_code.fetch_main(a.repo)
        print(note)
        if not ok and a.execute:
            print(f"NOT EXECUTING: {a.main_ref} could not be fetched, so the code floor may be stale.")
            return 1
    plan = build_plan(a.side_root, dataset, manifest, repo=a.repo, main_ref=a.main_ref,
                      strict_figures=a.strict_figures)
    rep = plan.report
    for m in rep.infos:
        print("  " + m)
    for m in rep.warns:
        print("WARN: " + m)
    for m in rep.fails:
        print("FAIL: " + m)
    counts = {}
    for act, _ in plan.copies:
        counts[act] = counts.get(act, 0) + 1
    print(f"plan: {counts or 'nothing to copy'}")

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    pdir = Path(a.side_root) / "promote"
    pdir.mkdir(exist_ok=True)
    plan_path = pdir / f"plan_{ts}.tsv"
    with open(plan_path, "w") as fh:
        for act, rel in plan.copies:
            fh.write(f"{act}\t{plan.md5.get(rel, '')}\t{rel}\n")
    print(f"plan written: {plan_path}")

    if rep.fails:
        print(f"NOT READY: {len(rep.fails)} blocking item(s).")
        return 1
    if not a.execute:
        print("DRY RUN. READY: rerun with --execute.")
        return 0
    if not plan.copies:
        print("Nothing to promote.")
        return 0
    with open(pdir / f"execute_{ts}.log", "w") as logf:
        def log(m):
            print(m)
            logf.write(m + "\n")
            logf.flush()
        return execute(plan, a.side_root, dataset, manifest, a.manifest, a.repo, log, main_ref=a.main_ref)


def _dataset_from_env(repo):
    env = Path(repo) / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        if line.startswith("SCRATCH_DIR="):
            return str(Path(line.split("=", 1)[1].strip().strip("'\"")) / "output")
    raise SystemExit("--dataset not given and SCRATCH_DIR not found in .env")


if __name__ == "__main__":
    sys.exit(main())
