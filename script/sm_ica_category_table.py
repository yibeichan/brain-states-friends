#!/usr/bin/env python
"""Supplement table: ICA-HMM correspondence by HMM-state taxonomy category.

Pure post-hoc AGGREGATION of the saved sm_ica_states results -- no ICA re-run,
no re-matching, no new statistics. For each subject, takes the K_active "all"
state-set matched pairs (ICA components Hungarian-matched to the full active HMM
repertoire), labels each matched HMM state by its summary_category (from the 05e
state_flags.csv), and tabulates spatial (Tier-1) and temporal (Tier-2)
correspondence per category.

FDR note: spatial_q / tier2_q are read straight from the JSON, where BH-FDR was
applied within the full active-set family at that K. The per-category "surviving"
counts are therefore descriptive subsets of that single within-subject family
(not separately re-corrected per category).

Outputs (to sm_ica_states/{parcellation}/):
  category_correspondence_table.csv  -- long format, one row per (subject, category)
  category_correspondence_table.md   -- supplement-ready markdown
"""
import os
import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
SCRATCH_DIR = os.getenv("SCRATCH_DIR")

SHORT = {
    "eligible_for_content_analysis": "Content-eligible",
    "run_onset_anchored": "Run-onset",
    "season_temporal": "Season",
    "drift_anchored": "Drift",
    "low_confidence": "Low-confidence",
    "unused": "Unused",
    "rare": "Rare",
}
# Display order: substantive states first, then low-support / timing-confounded.
CAT_ORDER = ["Content-eligible", "Run-onset", "Season", "Drift",
             "Low-confidence", "Unused", "Rare"]


def _arr(x):
    return np.array([np.nan if v is None else v for v in x], dtype=float)


def build_table(parcellation, vt):
    ica_base = os.path.join(SCRATCH_DIR, "output", "sm_ica_states", parcellation)
    flag_base = os.path.join(SCRATCH_DIR, "output",
                             "05e_temporal_trend_a4", parcellation)
    rows = []
    skipped = []
    for sdir in sorted(glob.glob(os.path.join(ica_base, "sub-*"))):
        sid = os.path.basename(sdir)
        summ_path = os.path.join(sdir, "ica_match_summary.json")
        if not os.path.exists(summ_path):
            skipped.append(sid)  # reported loudly after the loop
            continue
        with open(summ_path) as f:
            d = json.load(f)
        K = d["K_active"]
        if str(K) not in d["by_K"]:
            raise SystemExit(f"{sid}: K_active={K} not in by_K "
                             f"({list(d['by_K'])}); run the K_active grid first.")
        a = d["by_K"][str(K)]["state_sets"]["all"]
        flags_path = os.path.join(flag_base, sid, f"vt{vt}", "state_flags.csv")
        if not os.path.exists(flags_path):
            raise SystemExit(
                f"{sid}: missing {flags_path} (05e_a4 state_flags); cannot label "
                "categories. Run 05e_a4 for this subject or exclude it.")
        flags = pd.read_csv(flags_path)
        cat = dict(zip(flags["state"].astype(int), flags["summary_category"]))

        ids = [int(s) for s in a["hmm_state_ids"]]
        r = _arr(a["matched_r"])
        sq = _arr(a["spatial_q"])
        # SIGNED Tier-2 rho: the test is one-sided for positive sign-aligned
        # correspondence, so a negative rho is a directional failure -- averaging
        # |rho| would inflate the magnitude by counting failures as successes.
        rho = _arr(a["tier2_rho"])
        tq = _arr(a["tier2_q"])
        # Schema-drift guard: all per-pair arrays must align with hmm_state_ids.
        n_pairs = len(ids)
        if not all(len(x) == n_pairs for x in (r, sq, rho, tq)):
            raise SystemExit(
                f"{sid} K={K}: misaligned 'all' arrays "
                f"(ids={n_pairs}, r={len(r)}, sq={len(sq)}, rho={len(rho)}, "
                f"tq={len(tq)}) -- schema drift in ica_match_summary.json.")
        labels = np.array([SHORT.get(cat.get(s, "?"), cat.get(s, "?"))
                           for s in ids])
        # Every matched state must carry a known category, else denominators
        # are silently biased (missing state or drifted summary_category name).
        unknown = sorted({lab for lab in labels if lab not in CAT_ORDER})
        if unknown:
            raise SystemExit(
                f"{sid}: matched states with unknown/missing category "
                f"{unknown} (not in {CAT_ORDER}); check state_flags.csv "
                "coverage and the SHORT label map.")
        for c in CAT_ORDER:
            sel = labels == c
            n = int(sel.sum())
            if n == 0:
                continue
            sp_q = sq[sel]
            tp_rho = rho[sel]
            tp_q = tq[sel]
            sp_tested = int(np.isfinite(sp_q).sum())
            tp_tested = int(np.isfinite(tp_q).sum())
            # Tier-2 excludes low-occupancy states (min_occ guard), so some
            # categories (e.g. Unused/Rare) have no temporal test -> report NaN
            # rather than averaging an empty slice. Signed mean (see above).
            t_mean = (round(float(np.nanmean(tp_rho)), 3)
                      if tp_tested > 0 else float("nan"))
            rows.append({
                "subject": sid,
                "K_active": K,
                "category": c,
                "n_states": n,
                # n<3 categories (Season, Rare, often Unused) are point estimates
                # over 1-2 states -- descriptive only, not a distribution.
                "small_n": n < 3,
                "spatial_mean_r": round(float(np.nanmean(r[sel])), 3),
                "spatial_sig": int(np.nansum(sp_q < 0.05)),
                "spatial_tested": sp_tested,
                "temporal_mean_rho": t_mean,
                "temporal_sig": int(np.nansum(tp_q < 0.05)),
                "temporal_tested": tp_tested,
            })
    if skipped:
        print(f"WARNING: {len(skipped)} subject(s) missing ica_match_summary.json "
              f"and EXCLUDED from the table: {skipped}", file=sys.stderr)
    return pd.DataFrame(rows)


def to_markdown(df):
    lines = [
        "| Subject | K_active | Category | n states | Spatial mean r | "
        "Spatial sig (q<.05) | Temporal mean rho (signed) | "
        "Temporal sig (q<.05) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, x in df.iterrows():
        if x.temporal_tested > 0:
            t_mean = f"{x.temporal_mean_rho:.3f}"
            t_sig = f"{x.temporal_sig}/{x.temporal_tested}"
        else:
            t_mean = t_sig = "–"  # no Tier-2 test (low-occupancy guard)
        n_disp = f"{x.n_states}†" if x.small_n else f"{x.n_states}"
        cat_disp = (f"{x.category}‡"
                    if x.category in ("Low-confidence", "Unused", "Rare")
                    else x.category)
        lines.append(
            f"| {x.subject} | {x.K_active} | {cat_disp} | {n_disp} | "
            f"{x.spatial_mean_r:.3f} | {x.spatial_sig}/{x.spatial_tested} | "
            f"{t_mean} | {t_sig} |")
    lines += [
        "",
        "Temporal mean rho is signed (one-sided test for positive sign-aligned "
        "correspondence); negative values indicate directional mismatch, not "
        "magnitude. Spatial q is BH-FDR within each subject's full active-set "
        "family; per-category counts are descriptive subsets, not re-corrected. "
        "Tier-2 p-values are conditional on the Tier-1 spatial match; "
        "low-occupancy states are excluded from the temporal test (–).",
        "† n<3 states: point estimate over 1-2 states, descriptive only.",
        "‡ Low-confidence / Unused / Rare are low-support or quality-flagged "
        "(sub-HRF / sparse) categories; spatial matches there reflect "
        "recoverability of the state's mean map within the shared PC subspace, "
        "NOT independent biological or content validation -- the state flags "
        "stand.",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parcellation", default="atlas-4S156Parcels")
    p.add_argument("--vt", default="0.95")
    a = p.parse_args()
    if SCRATCH_DIR is None:
        raise SystemExit("SCRATCH_DIR must be set in .env")
    df = build_table(a.parcellation, a.vt)
    out_dir = os.path.join(SCRATCH_DIR, "output", "sm_ica_states", a.parcellation)
    csv_path = os.path.join(out_dir, "category_correspondence_table.csv")
    md_path = os.path.join(out_dir, "category_correspondence_table.md")
    # Outputs may already be read-only git-annex symlinks (after a datalad save);
    # write a temp then os.replace so the rename overwrites the symlink without
    # touching annex content (PermissionError otherwise).
    def _atomic_write(path, write_fn):
        tmp = path + ".tmp"
        write_fn(tmp)
        os.replace(tmp, path)
    _atomic_write(csv_path, lambda p: df.to_csv(p, index=False))
    _atomic_write(md_path,
                  lambda p: open(p, "w").write(to_markdown(df) + "\n"))
    print(df.to_string(index=False))
    print(f"\nwrote {csv_path}\nwrote {md_path}")


if __name__ == "__main__":
    main()
