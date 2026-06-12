#!/usr/bin/env python3
"""Programmatic completeness scanner for the 08d D1 (LLaMA) text-depth leg.

Replaces the hand-maintained ``ORPHANS`` assoc-array in
``launch_08d_orphans.sh``. That manifest conflated *main-complete* with
*cell-complete* and stranded cells (e.g. sub-03 lag5 was marked "done" on the
main partial while its neg partial sat at 4/28). This scanner reads every
``(subject, lag)`` cell's partials directly, cross-references ``squeue`` for
in-flight jobs, and reports which cells are **orphaned** - incomplete AND not
covered by any running/pending job - so the D1 text-depth leg can be finished
without hand-guessing.

Decision context (2026-05-31): the run-onset-anchored *negative control* was
dropped from the manuscript (run-onset states are feature-distinctive, not
content-free; see ``findings_08d_transformer_depth.md`` and the schist research
note). Consequently **only MAIN completeness gates resubmission** by default
(``--target main``). Neg-control partial status is still reported for
visibility, and ``--target both`` will also resubmit incomplete neg cells if
you want to keep the (now-exploratory) neg outputs current.

A partial is COMPLETE iff it parses and ``len(results) == n_layers_total``.
A missing file, ``n_layers_total`` of 0, or unparseable JSON counts as
INCOMPLETE (fail-safe). NOTE: a layer whose decoder legitimately returned
``None`` (degenerate fold, <2 classes) is never written to ``results`` (see
``08d_transformer_depth.py`` ~line 540), so such a cell can sit permanently at
``< n_layers_total``. The scanner does NOT auto-resubmit blindly: it prints the
missing-layer indices and the partial mtime so a stuck-short / degenerate cell
is visible to the operator, who decides. Rely on ``--exclude`` (node2803,
node3805) to avoid the known silent-stall nodes; the scanner never auto-kills.

Usage::

    # default: dry-run, report only, submit nothing
    python script/utils/find_08d_orphans.py
    python script/utils/find_08d_orphans.py --submit          # submit orphan arrays
    python script/utils/find_08d_orphans.py --target both     # gate on main AND neg
    python script/utils/find_08d_orphans.py --model w2v-bert-2.0  # scan a different model
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

# ── Constants (verified against script/08d_transformer_depth.{py,sh}) ──────
# Array index → subject: SUBJECTS=(sub-01 .. sub-06); SUB_ID=${SUBJECTS[$idx]}.
SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"]
LAGS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
DEFAULT_MODEL = "llama-3.2-3b"
DEFAULT_PARC = "atlas-4S156Parcels"
DEFAULT_STIMULUS = "friends"

# Submit env - mirrors launch_08d_orphans.sh exactly.
SUBMIT_DEFAULTS = {
    "partition": "mit_preemptable",
    "exclude": "node2803,node3805",
    "time": "1-12:00:00",
    "mem": "160G",
    "cpus": "8",
    "n_jobs": "8",
    "n_perms": "1000",
    "vt": "0.95",
    "script": "script/08d_transformer_depth.sh",
}


# ── Pure helpers (unit-testable; no IO) ────────────────────────────────────
def parse_array_spec(spec: str) -> list[int]:
    """Expand a SLURM array task spec into the list of array indices.

    Handles the forms squeue emits in the ``%i``/``%K`` field after the
    ``jobid_`` prefix is stripped: a bare running task ``"2"``, a range
    ``"[1-4]"``, an explicit list ``"[0,1,2,4]"``, a single ``"[5]"``, and a
    ``%``-throttled range ``"[1-4%2]"`` (the ``%k`` concurrency cap is not an
    index and is dropped). Mixed list+range elements (``"[0,2-4]"``) are
    supported. Unparseable fragments are skipped.

    Do NOT reuse ``chain_08d_successors.sh``'s ``s/[][]//g`` approach - it
    mangles ranges. We strip brackets, drop a trailing ``%k`` throttle, then
    expand element by element.
    """
    spec = spec.strip()
    # Drop surrounding brackets if present.
    if spec.startswith("[") and spec.endswith("]"):
        spec = spec[1:-1]
    # Drop a trailing %k concurrency throttle (e.g. "1-4%2" -> "1-4").
    spec = spec.split("%", 1)[0]
    out: list[int] = []
    for frag in spec.split(","):
        frag = frag.strip()
        if not frag:
            continue
        if "-" in frag:
            lo_hi = frag.split("-", 1)
            try:
                lo, hi = int(lo_hi[0]), int(lo_hi[1])
            except ValueError:
                continue
            out.extend(range(lo, hi + 1))
        else:
            try:
                out.append(int(frag))
            except ValueError:
                continue
    return out


def job_lag(job_name: str, model: str) -> int | None:
    """Return the temporal lag a D1 job operates on, or None if it is not a
    D1 job for ``model``.

    Job names: ``08d_D1_lag{N}_{model}`` (PERLAGS fan), ``08d_D1f_lag{N}_{model}``
    (orphan launcher), and ``08d_D1s_lag{N}_{model}`` (chain resumer) - the
    ``[a-z]?`` between ``D1`` and ``_lag`` matches the optional single-letter
    suffix (``f``/``s``/empty). It is deliberately NOT ``[a-z0-9]*``, which
    would over-match a hypothetical ``08d_D1net_lag…`` / ``08d_D1merge_lag…``
    (those analyses do not run per-lag, but the tight pattern is robust to
    future renames). The model is matched explicitly so w2v/dinov2 D1 jobs are
    not counted as in-flight for a llama scan.
    """
    m = re.fullmatch(rf"08d_D1[a-z]?_lag(\d+)_{re.escape(model)}", job_name)
    return int(m.group(1)) if m else None


def is_complete(n_done: int, n_total: int | None) -> bool:
    """Cell is complete iff n_total is known (>0) and every layer is present."""
    return n_total is not None and n_total > 0 and n_done >= n_total


# ── IO helpers ─────────────────────────────────────────────────────────────
@dataclass
class PartialStatus:
    exists: bool = False
    n_done: int = 0
    n_total: int | None = None
    missing_layers: list[int] = field(default_factory=list)
    mtime: float | None = None
    unparseable: bool = False

    @property
    def complete(self) -> bool:
        return (not self.unparseable) and is_complete(self.n_done, self.n_total)


def read_partial(path: str) -> PartialStatus:
    """Read a single ``D1_*_lag{N}.json`` partial and report completeness."""
    if not os.path.exists(path):
        return PartialStatus(exists=False)
    try:
        mtime = os.path.getmtime(path)
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return PartialStatus(exists=True, unparseable=True,
                             mtime=os.path.getmtime(path)
                             if os.path.exists(path) else None)
    results = data.get("results", {})
    n_total = data.get("n_layers_total") or None
    done_idx = {int(k) for k in results}
    missing = (
        [i for i in range(n_total) if i not in done_idx]
        if isinstance(n_total, int) and n_total > 0 else []
    )
    return PartialStatus(
        exists=True, n_done=len(done_idx), n_total=n_total,
        missing_layers=missing, mtime=mtime,
    )


def query_inflight(model: str) -> set[tuple[int, int]]:
    """Return the set of ``(array_idx, lag)`` cells with a running or pending
    D1 job for ``model``. One job per ``(sub, lag)`` covers both the main and
    neg halves, so the test is cell-level.

    Read partials BEFORE calling this (safe race ordering): if a job finishes
    between the partial read and the squeue snapshot, the cell is seen as
    incomplete + not-in-flight and is conservatively resubmitted - a resume
    from the (now-complete) checkpoint is a cheap no-op, never data loss.
    """
    user = os.environ.get("USER", "")
    try:
        out = subprocess.run(
            ["squeue", "--me", "-h", "-o", "%j|%t|%i"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        if user:
            try:
                out = subprocess.run(
                    ["squeue", "-u", user, "-h", "-o", "%j|%t|%i"],
                    capture_output=True, text=True, check=True,
                ).stdout
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("WARNING: squeue unavailable - treating ALL cells as "
                      "not-in-flight (conservative; may resubmit live cells).",
                      file=sys.stderr)
                return set()
        else:
            return set()

    inflight: set[tuple[int, int]] = set()
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        name, _state, jobid = parts[0], parts[1], parts[2]
        lag = job_lag(name, model)
        if lag is None:
            continue
        # jobid is "<A>_<arrayspec>" for arrays, or "<A>" for a non-array job.
        array_part = jobid.split("_", 1)[1] if "_" in jobid else ""
        for idx in parse_array_spec(array_part):
            inflight.add((idx, lag))
    return inflight


# ── Scan ───────────────────────────────────────────────────────────────────
@dataclass
class CellRow:
    sub_idx: int
    sub_id: str
    lag: int
    main: PartialStatus
    neg: PartialStatus
    inflight: bool

    def state(self, target: str) -> str:
        gate_incomplete = (not self.main.complete) or (
            target == "both" and not self.neg.complete
        )
        if not gate_incomplete:
            return "complete"
        return "in-flight" if self.inflight else "ORPHAN"


def scan(scratch_dir: str, parc: str, stimulus: str, model: str) -> list[CellRow]:
    inflight = None  # filled after partials are read (race ordering)
    rows_pre: list[tuple[int, str, int, PartialStatus, PartialStatus]] = []
    for sub_idx, sub_id in enumerate(SUBJECTS):
        pdir = os.path.join(
            scratch_dir, "output", "08d_transformer_depth", parc, sub_id,
            f"{stimulus}_{model}", "partials",
        )
        for lag in LAGS:
            main = read_partial(os.path.join(pdir, f"D1_D1_main_lag{lag}.json"))
            neg = read_partial(
                os.path.join(pdir, f"D1_D1_neg_control_lag{lag}.json"))
            rows_pre.append((sub_idx, sub_id, lag, main, neg))
    inflight = query_inflight(model)
    return [
        CellRow(sub_idx, sub_id, lag, main, neg,
                inflight=(sub_idx, lag) in inflight)
        for (sub_idx, sub_id, lag, main, neg) in rows_pre
    ]


def _fmt_partial(p: PartialStatus) -> str:
    if not p.exists:
        return "  -  "
    if p.unparseable:
        return "BADJSON"
    total = p.n_total if p.n_total is not None else "?"
    return f"{p.n_done:>2}/{total}"


def print_table(rows: list[CellRow], target: str) -> None:
    print(f"\n{'sub':<7} {'lag':>3} {'main':>7} {'neg':>7} {'state':<10} "
          f"{'main_missing / mtime-age':<28}")
    print("-" * 72)
    import time
    now = time.time()
    for r in sorted(rows, key=lambda x: (x.sub_idx, x.lag)):
        st = r.state(target)
        detail = ""
        if st != "complete" and r.main.exists and not r.main.complete:
            miss = r.main.missing_layers
            miss_s = (",".join(map(str, miss[:6])) + ("…" if len(miss) > 6 else "")
                      if miss else "")
            age = (f"{(now - r.main.mtime) / 3600:.1f}h"
                   if r.main.mtime else "?")
            detail = f"miss[{miss_s}] age={age}"
        print(f"{r.sub_id:<7} {r.lag:>3} {_fmt_partial(r.main):>7} "
              f"{_fmt_partial(r.neg):>7} {st:<10} {detail:<28}")


def orphan_specs(rows: list[CellRow], target: str) -> dict[int, list[int]]:
    """Map lag -> sorted array indices of ORPHAN cells (incomplete + not
    in-flight), grouped for per-lag sbatch arrays."""
    specs: dict[int, list[int]] = {}
    for r in rows:
        if r.state(target) == "ORPHAN":
            specs.setdefault(r.lag, []).append(r.sub_idx)
    return {lag: sorted(idxs) for lag, idxs in sorted(specs.items())}


def build_sbatch_cmd(lag: int, arr: list[int], model: str, stimulus: str,
                     cfg: dict) -> list[str]:
    name = f"08d_D1f_lag{lag}_{model}"
    arr_s = ",".join(map(str, arr))
    export = (
        f"ALL,ANALYSES=D1,LAGS={lag},MODEL={model},STIMULUS={stimulus},"
        f"VT={cfg['vt']},N_PERMS={cfg['n_perms']},N_JOBS={cfg['n_jobs']},PERLAGS="
    )
    return [
        "sbatch", "--parsable",
        f"--job-name={name}",
        f"--output=logs/{name}_%A_%a.out",
        f"--error=logs/{name}_%A_%a.err",
        f"--time={cfg['time']}",
        f"--mem={cfg['mem']}",
        f"--cpus-per-task={cfg['cpus']}",
        f"--partition={cfg['partition']}",
        f"--exclude={cfg['exclude']}",
        "--requeue",
        f"--array={arr_s}",
        f"--export={export}",
        cfg["script"],
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--parcellation", default=DEFAULT_PARC)
    ap.add_argument("--stimulus", default=DEFAULT_STIMULUS)
    ap.add_argument("--target", choices=["main", "both"], default="main",
                    help="Gate resubmission on MAIN completeness only (default) "
                         "or on MAIN and NEG. Neg status is always reported.")
    ap.add_argument("--submit", action="store_true",
                    help="Submit orphan arrays. Default is dry-run (report only).")
    args = ap.parse_args()

    scratch_dir = os.getenv("SCRATCH_DIR")
    if scratch_dir is None:
        print("SCRATCH_DIR must be set in the environment / .env", file=sys.stderr)
        return 2

    rows = scan(scratch_dir, args.parcellation, args.stimulus, args.model)
    print_table(rows, args.target)

    specs = orphan_specs(rows, args.target)
    n_complete = sum(1 for r in rows if r.state(args.target) == "complete")
    n_inflight = sum(1 for r in rows if r.state(args.target) == "in-flight")
    n_orphan = sum(len(v) for v in specs.values())
    print(f"\nSummary ({args.model}, target={args.target}): "
          f"{n_complete} complete, {n_inflight} in-flight, {n_orphan} ORPHAN "
          f"of {len(rows)} cells.")

    if not specs:
        print("No orphaned cells. Nothing to submit.")
        return 0

    print(f"\nOrphan arrays (lag -> array indices): "
          f"{ {lag: idxs for lag, idxs in specs.items()} }")
    cmds = [build_sbatch_cmd(lag, arr, args.model, args.stimulus, SUBMIT_DEFAULTS)
            for lag, arr in specs.items()]

    if not args.submit:
        print("\n[DRY-RUN] would submit:")
        for c in cmds:
            print("  " + " ".join(c))
        print("\nRe-run with --submit to launch.")
        return 0

    print("\nSubmitting orphan arrays:")
    for lag, c in zip(specs, cmds):
        try:
            jid = subprocess.run(c, capture_output=True, text=True,
                                 check=True).stdout.strip()
            print(f"  lag={lag} array={specs[lag]} -> {jid}")
        except subprocess.CalledProcessError as e:
            print(f"  lag={lag} FAILED: {e.stderr.strip()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
