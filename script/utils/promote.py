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

See docs/output_provenance.md.
"""
from __future__ import annotations

import argparse
import csv
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

FIGURE_EXT = {".png", ".pdf", ".svg", ".jpg", ".jpeg"}
DEFAULT_EXCLUDE = ["logs/**", "**/__pycache__/**", "**/.nfs*"]
BROAD_GLOBS = {"**", "**/*", "*"}


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
    equal: bool
    detail: str = ""
    max_diff: float = 0.0


def _close_arrays(a, b, rtol, atol, label=""):
    if a.shape != b.shape:
        return Cmp(False, f"{label}shape {a.shape} vs {b.shape}", math.inf)
    ka, kb = a.dtype.kind, b.dtype.kind
    if ka in "fc" or kb in "fc":
        if not (ka in "biufc" and kb in "biufc"):
            return Cmp(False, f"{label}dtype {a.dtype} vs {b.dtype}", math.inf)
        if (ka == "b") != (kb == "b"):
            return Cmp(False, f"{label}dtype {a.dtype} vs {b.dtype}", math.inf)
        af, bf = a.astype(np.complex128 if "c" in ka + kb else np.float64), \
            b.astype(np.complex128 if "c" in ka + kb else np.float64)
        na, nb = np.isnan(af), np.isnan(bf)
        if not np.array_equal(na, nb):
            return Cmp(False, f"{label}NaN positions differ ({int((na != nb).sum())} entries)", math.inf)
        ia, ib = np.isinf(af), np.isinf(bf)
        if not np.array_equal(ia, ib) or not np.array_equal(af[ia], bf[ib]):
            return Cmp(False, f"{label}inf entries differ", math.inf)
        fin = ~(na | ia)
        if not fin.any():
            return Cmp(True)
        d = np.abs(af[fin] - bf[fin])
        md = float(d.max())
        ok = bool(np.all(d <= atol + rtol * np.abs(bf[fin])))
        return Cmp(ok, f"{label}max |diff| {md:.3g}", md)
    if ka != kb and not (ka in "iu" and kb in "iu"):
        return Cmp(False, f"{label}dtype {a.dtype} vs {b.dtype}", math.inf)
    if ka in "biu":
        ok = bool(np.array_equal(a, b))
        md = 0.0 if ok else float(np.abs(a.astype(np.int64) - b.astype(np.int64)).max()) if ka != "b" else 1.0
        return Cmp(ok, f"{label}integer arrays {'equal' if ok else 'differ'}", md)
    ok = bool(np.array_equal(a, b))
    return Cmp(ok, f"{label}{'equal' if ok else 'differ'} ({a.dtype})", 0.0 if ok else math.inf)


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _close_json(a, b, rtol, atol, ignore, path="$"):
    """Recursive JSON comparison: dict key sets exact (order ignored), list
    order kept, bool and str exact, numbers within tolerance."""
    if isinstance(a, dict) and isinstance(b, dict):
        ka = set(a) - ignore
        kb = set(b) - ignore
        if ka != kb:
            return Cmp(False, f"{path}: keys differ ({sorted(ka ^ kb)[:5]})", math.inf)
        worst = Cmp(True)
        for k in sorted(ka):
            c = _close_json(a[k], b[k], rtol, atol, ignore, f"{path}.{k}")
            if not c.equal:
                return c
            if c.max_diff > worst.max_diff:
                worst = c
        return worst
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return Cmp(False, f"{path}: length {len(a)} vs {len(b)}", math.inf)
        worst = Cmp(True)
        for i, (x, y) in enumerate(zip(a, b)):
            c = _close_json(x, y, rtol, atol, ignore, f"{path}[{i}]")
            if not c.equal:
                return c
            if c.max_diff > worst.max_diff:
                worst = c
        return worst
    if _is_num(a) and _is_num(b):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return Cmp(True)
        if any(isinstance(v, float) and (math.isnan(v) or math.isinf(v)) for v in (a, b)):
            return Cmp(a == b, f"{path}: {a} vs {b}", 0.0 if a == b else math.inf)
        d = abs(a - b)
        ok = d <= atol + rtol * abs(b)
        return Cmp(ok, f"{path}: {a} vs {b}" if not ok else f"{path}: |diff| {d:.3g}", float(d))
    if type(a) is not type(b) or a != b:
        return Cmp(False, f"{path}: {str(a)[:40]!r} vs {str(b)[:40]!r}", math.inf)
    return Cmp(True)


def _float_or_none(s):
    try:
        return float(s)
    except ValueError:
        return None


def _close_table(pa, pb, delim, rtol, atol):
    with open(pa, newline="") as fa, open(pb, newline="") as fb:
        ra, rb = list(csv.reader(fa, delimiter=delim)), list(csv.reader(fb, delimiter=delim))
    if not ra or not rb:
        return Cmp(ra == rb, "empty table")
    if ra[0] != rb[0]:
        return Cmp(False, f"header differs: {ra[0][:6]} vs {rb[0][:6]}", math.inf)
    if len(ra) != len(rb):
        return Cmp(False, f"rows {len(ra) - 1} vs {len(rb) - 1}", math.inf)
    worst = 0.0
    for i, (x, y) in enumerate(zip(ra[1:], rb[1:]), 1):
        if len(x) != len(y):
            return Cmp(False, f"row {i}: {len(x)} vs {len(y)} cells", math.inf)
        for col, u, v in zip(ra[0], x, y):
            fu, fv = _float_or_none(u), _float_or_none(v)
            if fu is None or fv is None:
                if u != v:
                    return Cmp(False, f"row {i} {col}: {u!r} vs {v!r}", math.inf)
                continue
            if math.isnan(fu) and math.isnan(fv):
                continue
            d = abs(fu - fv)
            if not d <= atol + rtol * abs(fv):
                return Cmp(False, f"row {i} {col}: {u} vs {v}", d)
            worst = max(worst, d)
    return Cmp(True, f"max |diff| {worst:.3g}", worst)


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
                    return Cmp(False, f"npz keys differ ({sorted(set(za.files) ^ set(zb.files))})", math.inf)
                worst = Cmp(True)
                for k in sorted(za.files):
                    c = _close_arrays(za[k], zb[k], rtol, atol, label=f"{k}: ")
                    if not c.equal:
                        return c
                    if c.max_diff > worst.max_diff:
                        worst = c
                return worst
        if ext == ".json":
            return _close_json(json.loads(a.read_text()), json.loads(b.read_text()), rtol, atol, set(ignore_keys))
        if ext in (".csv", ".tsv"):
            return _close_table(a, b, "\t" if ext == ".tsv" else ",", rtol, atol)
    except (ValueError, OSError) as e:
        return Cmp(False, f"cannot compare: {e}", math.inf)
    ok = md5(a) == md5(b)
    return Cmp(ok, "bytes equal" if ok else "bytes differ", 0.0 if ok else math.inf)


# ------------------------------------------------------- dataset reading
def _run(cmd, cwd=None, check=False, text=True):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=text)
    if check and p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {p.stderr.strip()}")
    return p


def annex_key_location(dataset, key):
    p = _run(["git", "annex", "contentlocation", key], cwd=dataset)
    if p.returncode != 0 or not p.stdout.strip():
        return None
    loc = Path(dataset) / p.stdout.strip()
    return loc if loc.exists() else None


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
    report: Report = field(default_factory=Report)


def load_manifest(path):
    with open(path, "rb") as fh:
        m = tomllib.load(fh)
    for k in ("message", "code_commit", "change"):
        if not m.get(k):
            raise ValueError(f"manifest {path}: '{k}' is required")
    for g in m["change"]:
        if g.strip("/") in BROAD_GLOBS:
            raise ValueError(f"manifest {path}: change glob {g!r} declares everything; list the paths")
    return m


def _tol(manifest, rel):
    t = manifest.get("tolerance", {})
    rtol, atol, ignore = float(t.get("rtol", 1e-9)), float(t.get("atol", 1e-12)), []
    for o in manifest.get("override", []):
        if matches(rel, [o["glob"]]):
            rtol = float(o.get("rtol", rtol))
            atol = float(o.get("atol", atol))
            ignore += list(o.get("ignore_keys", []))
    return rtol, atol, ignore


def side_files(side_out, scope, exclude):
    """Recomputed side-root files (symlinks are inputs linked in, not outputs)."""
    files, linked = [], 0
    for root, dirs, names in os.walk(side_out):
        keep = []
        for d in dirs:
            if os.path.islink(os.path.join(root, d)):
                linked += 1
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
    return sorted(files), linked


def _is_record(rel):
    n = os.path.basename(rel)
    return n.startswith("provenance_") and n.endswith(".json")


def _under(rel, d):
    return d == "" or rel == d or rel.startswith(d.rstrip("/") + "/")


# Output top directories of the stages that write provenance (check_code.PROVENANCE_STAGES).
_STAGE_TOPS = {
    "03a_pca4combined_hmm": "03a",
    "05e_temporal_trend_a1": "05e_a1",
    "05e_temporal_trend_a3": "05e_a3",
    "05f_state_fc": "05f",
}
_04RC_FILES = {"state_empirical_corr.npy", "fc_metadata.json"}


def gated_stage(rel):
    """Stage whose provenance must accompany ``rel``, or None."""
    parts = Path(rel).parts
    if parts and parts[0] in _STAGE_TOPS:
        return _STAGE_TOPS[parts[0]]
    if parts and parts[0] == "04_combined_hdphmm" and parts[-1] in _04RC_FILES:
        return "04rc"
    return None


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
    files, linked = side_files(side_out, scope, exclude)
    rep.info(f"{len(files)} recomputed side-root files in scope ({linked} symlinked inputs skipped)")

    for g in change:
        n = sum(1 for r in files if matches(r, [g]))
        rep.info(f"change glob {g!r}: {n} file(s)")
        if n == 0:
            rep.fail(f"change glob {g!r} matches no recomputed file")

    records = []
    for rel in files:
        if _is_record(rel):
            records.append(rel)
            continue
        src, dst = side_out / rel, dataset / rel
        in_change = matches(rel, change)
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
    allow_inputs = manifest.get("allow_input_changes", "")
    for rel in records:
        rdir = os.path.dirname(rel)
        if not any(_under(r, rdir) for r in planned):
            rep.info(f"provenance record left out (nothing planned under its directory): {rel}")
            continue
        rec = json.loads((side_out / rel).read_text())
        for p in check_code.check_record(rec, repo=repo, main_ref=main_ref):
            rep.fail(f"{rel}: {p}")
        if code_full and (rec.get("code") or {}).get("commit") != code_full:
            rep.fail(f"{rel}: record names code {str((rec.get('code') or {}).get('commit'))[:10]}, "
                     f"manifest code_commit is {code_full[:10]}")
        old = dataset / rel
        if old.exists():
            prev = json.loads(old.read_text())
            for k in ("inputs", "sessions_read"):
                if prev.get(k) != rec.get(k):
                    a_, b_ = prev.get(k) or {}, rec.get(k) or {}
                    diff = sorted(x for x in set(a_) | set(b_) if a_.get(x) != b_.get(x))
                    msg = f"{rel}: {k} differ from the dataset's record ({len(diff)} entries, e.g. {diff[:3]})"
                    if allow_inputs:
                        rep.warn(msg + f" [allowed: {allow_inputs}]")
                    else:
                        rep.fail(msg)
        plan.copies.append(("UPDATE" if os.path.lexists(dataset / rel) else "NEW", rel))
    for rel in planned:
        stage = gated_stage(rel)
        if stage and not any(os.path.basename(r) == f"provenance_{stage}.json"
                             and _under(rel, os.path.dirname(r)) for r in records):
            rep.fail(f"{rel}: stage {stage} writes provenance, but the side root has no record for it")

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
    if not any(c.get("expect", "equal") == "differ" for c in controls) and not manifest.get("no_positive_control"):
        rep.fail("no positive control (expect = \"differ\"); add one or set no_positive_control = \"<reason>\"")
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
            rtol = float(c.get("rtol", 1e-9))
            atol = float(c.get("atol", c.get("tol", 1e-12)))
            cmp_ = compare_files(side_out / rel, ref_path, rtol, atol, c.get("ignore_keys", []))
            if expect == "equal":
                (rep.info if cmp_.equal else rep.fail)(f"{tag}: {'PASS' if cmp_.equal else 'FAIL'} ({cmp_.detail})")
            elif expect == "differ":
                min_diff = float(c.get("min_diff", 0.0))
                ok = (not cmp_.equal) and cmp_.max_diff >= min_diff
                (rep.info if ok else rep.fail)(
                    f"{tag}: {'PASS' if ok else 'FAIL'} ({cmp_.detail}; need max |diff| >= {min_diff:g})")
            else:
                rep.fail(f"{tag}: expect must be 'equal' or 'differ'")


# --------------------------------------------------------------- execute
def datalad_cmd(repo):
    env = os.environ.get("PROMOTE_DATALAD")
    if env:
        return shlex.split(env)
    return ["uv", "run", "--project", str(repo), "--extra", "datalad", "datalad"]


def execute(plan, side_root, dataset, manifest, manifest_path, repo, log):
    side_out = Path(side_root) / "output"
    dataset = Path(dataset)
    for act, rel in plan.copies:
        src, dst = side_out / rel, dataset / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(dst):
            os.remove(dst)                      # annex symlink or plain file
        shutil.copy2(src, dst)
        os.chmod(dst, os.stat(dst).st_mode | 0o200)
        log(f"{act} {rel}")
    floor = ",".join(s[:10] for s, _, _ in check_code.load_floor())
    msg = (f"{manifest['message']}\n\n"
           f"code_commit: {manifest['code_commit']}\n"
           f"code_floor: {floor}\n"
           f"side_root: {side_root}\n"
           f"manifest_sha256: {hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()}\n"
           f"promote_check: passed")
    rels = [r for _, r in plan.copies]
    p = _run(datalad_cmd(repo) + ["save", "-m", msg, "--"] + rels, cwd=dataset)
    log(p.stdout + p.stderr)
    if p.returncode != 0:
        log(f"datalad save failed (rc {p.returncode})")
        return p.returncode
    bad = [r for r in rels if not (dataset / r).exists() or md5(dataset / r) != md5(side_out / r)]
    for r in bad:
        log(f"VERIFY FAIL: saved content differs from side root: {r}")
    head = _run(["git", "log", "-1", "--format=%h %s"], cwd=dataset).stdout.strip()
    log(f"output dataset HEAD: {head}")
    return 1 if bad else 0


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
        print(check_code.fetch_main(a.repo))
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
            fh.write(f"{act}\t{md5(Path(a.side_root) / 'output' / rel)}\t{rel}\n")
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
        return execute(plan, a.side_root, dataset, manifest, a.manifest, a.repo, log)


def _dataset_from_env(repo):
    env = Path(repo) / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        if line.startswith("SCRATCH_DIR="):
            return str(Path(line.split("=", 1)[1].strip().strip("'\"")) / "output")
    raise SystemExit("--dataset not given and SCRATCH_DIR not found in .env")


if __name__ == "__main__":
    sys.exit(main())
