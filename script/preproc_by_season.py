#!/usr/bin/env python3
"""preproc_by_season.py - fMRIPrep release and slice timing of the analysed Friends runs (Table S7).

The released CNeuroMod Friends derivatives were produced with three fMRIPrep
releases (20.2.3, 20.2.5, 20.2.6), and only some 20.2.3 sessions received
slice-timing correction (STC). Preprocessing therefore changes with season in
most participants. This script

1. tabulates, for every analysed run (the keys of each participant's
   decoded_states.pkl), its season, session, fMRIPrep release and STC status
   (Table S7A), and
2. tests, within each season that mixed preprocessing, whether mean state
   occupancy differs between the two run groups (Table S7B).

Release comes from each session's fMRIPrep report ("fMRIPrep version: X").
STC comes from the session report for 20.2.3 ("Slice timing correction:
Applied / Not applied"; a 20.2.3 report covers only its own session) and from
each run's sidecar field SliceTimingCorrected for 20.2.5 and 20.2.6. A run
acquired twice uses the session the analysis used (utils.common.pick_run_session).

Statistic: L1 distance between the two groups' mean fractional-occupancy
vectors over all states. Nulls, all within one participant and season:
  N1  run-label permutation (10,000 draws). Ignores session structure, so it is
      reported with a calibration on uniform seasons and is not used for inference.
  N2  exact session-label permutation: preprocessing is constant within a
      session, so every assignment of the A label to the same number of
      sessions is enumerated; p = share of assignments with L1 >= observed,
      floor = 1 / number of assignments.
  N3  order-preserving: every contiguous block of k runs (k = size of group A),
      in acquisition order, inside the same participant's season groups with
      uniform preprocessing and at least 2k runs; the share of blocks whose
      block-versus-rest L1 >= observed.

Inputs:
    $DATA_DIR/cneuromod.processed/fmriprep/friends/{sub}_{ses}.html and
        {sub}/{ses}/func/{sub}_{ses}_task-{run}_space-fsLR_den-91k_bold.json
    $SCRATCH_DIR/output/04_combined_hdphmm/{parc}/{sub}/final/vt0.95/decoded_states.pkl
    $SCRATCH_DIR/output/02_parcel_ts_avg/{parc}/{sub}/{sub}_{ses}_task-{run}_parcel_avg.npy (session of each run)
    $SCRATCH_DIR/output/05a_recurrence_analysis/{parc}/{sub}/vt0.95/recurrence_summary.json (check)
Outputs (--out_dir):
    runs.tsv        one row per analysed run
    contrasts.json  one entry per within-season contrast
    calibration.json  N1 on first-12-runs vs rest of uniform seasons

Usage:
    python script/preproc_by_season.py --out_dir <dir>
"""
import argparse
import collections
import glob
import itertools
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from utils.common import pick_run_session  # noqa: E402

PARC = "atlas-4S156Parcels"
SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"]
N_STATES = 50
FO_THRESHOLD = 0.02  # 05a fo_active_threshold, used only for the 05a check
N_PERM = 10000
SEED = 20261102

# (participant, season, group A, group B, kind); labels are "<release>/<STC|noSTC>"
CONTRASTS = [
    ("sub-01", 3, "20.2.3/STC", "20.2.3/noSTC", "STC (version fixed)"),
    ("sub-01", 4, "20.2.3/STC", "20.2.5/noSTC", "STC+version"),
    ("sub-02", 5, "20.2.3/STC", "20.2.5/noSTC", "STC+version"),
    ("sub-03", 5, "20.2.3/STC", "20.2.5/noSTC", "STC+version"),
    ("sub-05", 3, "20.2.3/STC", "20.2.5/noSTC", "STC+version"),
    ("sub-01", 1, "20.2.5/noSTC", "20.2.3/noSTC", "version 20.2.3 vs 20.2.5"),
    ("sub-01", 2, "20.2.5/noSTC", "20.2.3/noSTC", "version 20.2.3 vs 20.2.5"),
    ("sub-01", 3, "20.2.5/noSTC", "20.2.3/noSTC", "version 20.2.3 vs 20.2.5"),
    ("sub-01", 6, "20.2.6/noSTC", "20.2.5/noSTC", "version 20.2.5 vs 20.2.6"),
    ("sub-03", 5, "20.2.6/noSTC", "20.2.5/noSTC", "version 20.2.5 vs 20.2.6"),
    ("sub-03", 6, "20.2.6/noSTC", "20.2.5/noSTC", "version 20.2.5 vs 20.2.6"),
    ("sub-05", 5, "20.2.6/noSTC", "20.2.5/noSTC", "version 20.2.5 vs 20.2.6"),
    ("sub-05", 6, "20.2.6/noSTC", "20.2.5/noSTC", "version 20.2.5 vs 20.2.6"),
    ("sub-06", 6, "20.2.6/noSTC", "20.2.5/noSTC", "version 20.2.5 vs 20.2.6"),
]

_RUN_RE = re.compile(r"s(\d+)e(\d+)([a-d])")


# ---------------------------------------------------------------- run table

def parse_report(text):
    """(release, n 'Applied', n 'Not applied') from an fMRIPrep session report."""
    m = re.search(r"fMRIPrep version: ([0-9.]+)", text)
    return (m.group(1) if m else None,
            text.count("Slice timing correction: Applied"),
            text.count("Slice timing correction: Not applied"))


def stc_status(release, n_applied, n_not, sidecar_value):
    """'STC' / 'noSTC', or a string naming why it could not be decided.

    20.2.3 status comes from the session report; later releases record it per
    run in the sidecar (SliceTimingCorrected).
    """
    if release == "20.2.3":
        if n_applied > 0 and n_not == 0:
            return "STC"
        if n_not > 0 and n_applied == 0:
            return "noSTC"
        return "mixed?"
    return {False: "noSTC", True: "STC"}.get(sidecar_value, str(sidecar_value))


def build_run_table(fmriprep_dir, output_dir):
    reports = {}
    for f in glob.glob(os.path.join(fmriprep_dir, "sub-0*_ses-*.html")):
        sub, ses = os.path.basename(f)[:-5].split("_")
        with open(f, errors="ignore") as fh:
            reports[(sub, ses)] = parse_report(fh.read())
    rows = []
    for sub in SUBJECTS:
        with open(os.path.join(output_dir, "04_combined_hdphmm", PARC, sub, "final", "vt0.95",
                               "decoded_states.pkl"), "rb") as fh:
            decoded = pickle.load(fh)
        files = glob.glob(os.path.join(output_dir, "02_parcel_ts_avg", PARC, sub,
                                       f"{sub}_ses-*_task-s*_parcel_avg.npy"))
        files_by_run = collections.defaultdict(list)
        for f in files:
            m = re.match(rf"{sub}_(ses-\d+)_task-(s\d+e\d+[a-d])_", os.path.basename(f))
            files_by_run[m.group(2)].append(f)
        for run in decoded:
            path = pick_run_session(sub, run, files_by_run[run], kind="parcel_ts")
            ses = re.search(r"(ses-\d+)", os.path.basename(path)).group(1)
            release, n_app, n_not = reports.get((sub, ses), (None, 0, 0))
            sidecar = None
            if release != "20.2.3":
                sc = os.path.join(fmriprep_dir, sub, ses, "func",
                                  f"{sub}_{ses}_task-{run}_space-fsLR_den-91k_bold.json")
                if os.path.exists(sc):
                    with open(sc) as fh:
                        sidecar = json.load(fh).get("SliceTimingCorrected", "missing")
                else:
                    sidecar = "nofile"
            m = _RUN_RE.match(run)
            rows.append(dict(sub=sub, run=run, season=int(m.group(1)), episode=int(m.group(2)),
                             part=m.group(3), ses=int(ses.split("-")[1]), version=release,
                             stc=stc_status(release, n_app, n_not, sidecar), n_tr=len(decoded[run])))
    return rows


def write_tsv(rows, path):
    keys = list(rows[0])
    with open(path, "w") as fh:
        fh.write("\t".join(keys) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[k]) for k in keys) + "\n")


# ---------------------------------------------------------------- statistics

def l1_from_mask(F, mask):
    return float(np.abs(F[mask].mean(0) - F[~mask].mean(0)).sum())


def l1_batch(F, masks):
    """L1 between group means for each row of a (n, runs) boolean mask array."""
    m = masks.astype(float)
    a = m @ F / m.sum(1, keepdims=True)
    b = (1 - m) @ F / (1 - m).sum(1, keepdims=True)
    return np.abs(a - b).sum(1)


def js_divergence(p, q):
    mid = 0.5 * (p + q)

    def kl(a, b):
        nz = a > 0
        return np.sum(a[nz] * np.log2(a[nz] / b[nz]))
    return float(0.5 * kl(p, mid) + 0.5 * kl(q, mid))


def session_null(F, units, is_a_unit):
    """Exact N2: every assignment of the A label to the same number of session units.

    units: per-run session key; is_a_unit: {unit: bool}. Returns (null L1 array, n assignments).
    """
    uniq = sorted(set(units))
    n_a = sum(1 for u in uniq if is_a_unit[u])
    uidx = np.array([uniq.index(u) for u in units])
    combos = list(itertools.combinations(range(len(uniq)), n_a))
    masks = np.stack([np.isin(uidx, c) for c in combos])
    return l1_batch(F, masks), len(combos)


def block_reference(F_all, groups, k):
    """N3 reference: L1 of every contiguous k-run block vs the rest of each uniform group of >= 2k runs.

    groups: {key: list of row indices in acquisition order}. Returns (values, first-block values by key).
    """
    ref, first = [], []
    for g, gi in groups.items():
        if len(gi) < 2 * k:
            continue
        Fg = F_all[gi]
        for start in range(0, len(gi) - k + 1):
            m = np.zeros(len(gi), bool)
            m[start:start + k] = True
            v = l1_from_mask(Fg, m)
            ref.append(v)
            if start == 0:
                first.append((g, v))
    return np.array(ref), first


def fractional_occupancy(decoded, runs):
    return np.stack([np.bincount(decoded[r], minlength=N_STATES) / len(decoded[r]) for r in runs])


def run_contrasts(rows, output_dir, rng):
    data = {}
    for sub in sorted({r["sub"] for r in rows}):
        with open(os.path.join(output_dir, "04_combined_hdphmm", PARC, sub, "final", "vt0.95",
                               "decoded_states.pkl"), "rb") as fh:
            decoded = pickle.load(fh)
        with open(os.path.join(output_dir, "05a_recurrence_analysis", PARC, sub, "vt0.95",
                               "recurrence_summary.json")) as fh:
            summary = json.load(fh)
        rr = sorted([r for r in rows if r["sub"] == sub],
                    key=lambda r: (r["ses"], r["season"], r["episode"], r["part"]))
        F = fractional_occupancy(decoded, [r["run"] for r in rr])
        data[sub] = dict(rows=rr, F=F)
        # Check: the 05a season-specificity index recomputed from these occupancies.
        active = (F > FO_THRESHOLD).astype(float)
        sea = np.array([r["season"] for r in rr])
        rec = np.stack([active[sea == k].mean(0) for k in sorted(set(sea))])
        diff = float(np.abs((rec.max(0) - rec.min(0)) - np.array(summary["specificity_index"])).max())
        print(f"[check] {sub}: max |specificity - 05a| = {diff:.2e}")
        if diff > 1e-12:
            raise RuntimeError(f"{sub}: occupancy does not reproduce the 05a specificity index ({diff:.2e})")

    results = []
    for sub, season, la, lb, kind in CONTRASTS:
        d = data[sub]
        idx = [i for i, r in enumerate(d["rows"]) if r["season"] == season and r["lab"] in (la, lb)]
        rr = [d["rows"][i] for i in idx]
        F = d["F"][idx]
        mask = np.array([r["lab"] == la for r in rr])
        obs = l1_from_mask(F, mask)
        null1 = l1_batch(F, np.stack([rng.permutation(mask) for _ in range(N_PERM)]))
        p1 = (np.sum(null1 >= obs - 1e-12) + 1) / (N_PERM + 1)
        units = [(r["season"], r["ses"]) for r in rr]
        labs_by_unit = collections.defaultdict(set)
        for r, u in zip(rr, units):
            labs_by_unit[u].add(r["lab"])
        if any(len(v) > 1 for v in labs_by_unit.values()):
            raise RuntimeError(f"{sub} S{season}: preprocessing varies within a session")
        null2, n2 = session_null(F, units, {u: la in v for u, v in labs_by_unit.items()})
        p2 = float(np.mean(null2 >= obs - 1e-12))
        k = int(mask.sum())
        groups = collections.defaultdict(list)
        for i, r in enumerate(d["rows"]):
            groups[(r["season"], r["lab"])].append(i)
        ref, first = block_reference(d["F"], groups, k)
        results.append(dict(
            sub=sub, season=season, A=la, B=lb, kind=kind, nA=k, nB=int((~mask).sum()),
            units_A=sum(1 for v in labs_by_unit.values() if la in v), units=len(labs_by_unit),
            L1=obs, JS=js_divergence(F[mask].mean(0), F[~mask].mean(0)),
            null1_mean=float(null1.mean()), p1=float(p1),
            n2=n2, p2=p2, floor2=1 / n2,
            n3=int(len(ref)), ref_median=float(np.median(ref)) if len(ref) else None,
            p3=float(np.mean(ref >= obs)) if len(ref) else None,
            first=[(f"S{g[0]} {g[1]}", round(v, 3)) for g, v in first],
            p3f=float(np.mean([v >= obs for _, v in first])) if first else None))
    return data, results


def calibrate_n1(data, rng, k=12, n_perm=2000, min_runs=40):
    """N1 on the first k runs vs the rest of each uniform season group (no preprocessing contrast)."""
    out = {}
    for sub in sorted(data):
        d = data[sub]
        groups = collections.defaultdict(list)
        for i, r in enumerate(d["rows"]):
            groups[(r["season"], r["lab"])].append(i)
        cells = []
        for g, gi in sorted(groups.items()):
            if len(gi) < min_runs:
                continue
            F = d["F"][gi]
            m = np.zeros(len(gi), bool)
            m[:k] = True
            obs = l1_from_mask(F, m)
            null = l1_batch(F, np.stack([rng.permutation(m) for _ in range(n_perm)]))
            cells.append(dict(season=g[0], label=g[1], L1=obs,
                              p=float((np.sum(null >= obs) + 1) / (n_perm + 1))))
        out[sub] = cells
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True)
    a = p.parse_args(argv)
    load_dotenv()
    data_dir, scratch = os.environ.get("DATA_DIR"), os.environ.get("SCRATCH_DIR")
    if not data_dir or not scratch:
        raise SystemExit("DATA_DIR and SCRATCH_DIR must be set (environment or .env)")
    fmriprep_dir = os.path.join(data_dir, "cneuromod.processed", "fmriprep", "friends")
    output_dir = os.path.join(scratch, "output")
    os.makedirs(a.out_dir, exist_ok=True)

    rows = build_run_table(fmriprep_dir, output_dir)
    bad = sorted({r["stc"] for r in rows} - {"STC", "noSTC"})
    if bad:
        raise RuntimeError(f"undecided slice-timing status for some runs: {bad}")
    write_tsv(rows, os.path.join(a.out_dir, "runs.tsv"))
    tally = collections.Counter((r["version"], r["stc"]) for r in rows)
    print(f"{len(rows)} runs: " + ", ".join(f"{v}/{s} {n}" for (v, s), n in sorted(tally.items())))

    for r in rows:
        r["lab"] = f"{r['version']}/{r['stc']}"
    rng = np.random.default_rng(SEED)
    data, results = run_contrasts(rows, output_dir, rng)
    with open(os.path.join(a.out_dir, "contrasts.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    with open(os.path.join(a.out_dir, "calibration.json"), "w") as fh:
        json.dump(calibrate_n1(data, rng), fh, indent=1)
    for c in results:
        print(f"{c['sub']} S{c['season']} {c['A']} vs {c['B']}: L1 {c['L1']:.3f}, "
              f"N2 p {c['p2']:.3f} (floor {c['floor2']:.3g}), N3 {c['p3']:.3f} ({c['n3']})")


if __name__ == "__main__":
    main()
