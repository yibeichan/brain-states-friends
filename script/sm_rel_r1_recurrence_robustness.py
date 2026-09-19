#!/usr/bin/env python3
"""sm_rel_r1_recurrence_robustness.py - threshold sensitivity and occupancy confound of recurrence.

The recurrence score is the fraction of Friends runs in which a state's
fractional occupancy (FO) exceeds 0.02 (about 14 s of a 12-min run). Two
questions about that construct, both about analyses already in the paper:

1. Threshold sensitivity. Recompute recurrence at FO thresholds 0.01, 0.02
   (reference), 0.03, 0.05 and report (a) rank stability of the ordering,
   (b) how many states cross the active line (recurrence > 0) and the
   content-eligibility line (recurrence >= 0.10 among states carrying no
   other exclusion flag), and (c) the range statistics quoted in Results.
   Exclusion flags (sub-HRF, run-onset-anchored, drift-anchored) do not
   depend on the FO threshold, so eligibility churn is computable from
   recurrence alone.
2. Occupancy confound. Spearman correlations among recurrence, the model's
   stationary distribution pi, and mean FO across Friends runs, over active
   states. Recurrence is known to track pi closely; this makes it a
   reproducible artifact the Methods can cite.

Faithfulness gates: recurrence at 0.02 must equal recurrence_scores.npy; the
recomputed category of every flag-free state at 0.02 must equal its
summary_category in state_flags.csv; pi must match stationary_distribution.npy.

Inputs (frozen, under $SCRATCH_DIR/output):
    05a_recurrence_analysis/{parc}/{sub}/vt{vt}/{fractional_occupancy.pkl, recurrence_scores.npy}
    05e_temporal_trend_a4/{parc}/{sub}/vt{vt}/state_flags.csv
    06b_transition_structure/{parc}/{sub}/vt{vt}/stationary_distribution.npy
    04_combined_hdphmm/{parc}/{sub}/final/vt{vt}/best_model.pkl
Output:
    {SCRATCH_DIR}/output/sm_rel_r1_recurrence_robustness/{parc}/{sub}/vt{vt}/
        recurrence_robustness_summary.json, recurrence_by_threshold.npy (thresholds x K)
"""
import os
import sys
import json
import pickle
import logging
import argparse
import platform
from pathlib import Path

import numpy as np
import scipy
import pandas as pd
from scipy.stats import spearmanr
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from utils.jax_free_model_io import _load_model_no_jax
from utils.stats import safe_float
from utils.stationary import stationary_distribution

load_dotenv()
SCRATCH_DIR = os.getenv("SCRATCH_DIR")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

THRESHOLDS = (0.01, 0.02, 0.03, 0.05)
REFERENCE_THRESHOLD = 0.02
ELIGIBILITY_RECURRENCE = 0.10
# Categories whose assignment depends on recurrence alone (no exclusion flag).
FLAG_FREE_CATEGORIES = ("eligible_for_content_analysis", "rare", "unused")


def recurrence_at(fo, n_states, threshold):
    """Fraction of runs in which FO > threshold, per state (main-pipeline definition)."""
    if len(fo) == 0:
        return np.zeros(n_states)
    counts = np.zeros(n_states)
    for vec in fo.values():
        counts += (np.asarray(vec) > threshold).astype(float)
    return counts / len(fo)


def mean_fo(fo):
    """Mean fractional occupancy across runs (runs weighted equally)."""
    return np.mean(np.vstack([fo[k] for k in sorted(fo)]), axis=0)


def recurrence_category(rec_value):
    """Category of a flag-free state from its recurrence alone."""
    if rec_value <= 0:
        return "unused"
    if rec_value < ELIGIBILITY_RECURRENCE:
        return "rare"
    return "eligible_for_content_analysis"


def churn(rec_ref, rec_alt, flag_free):
    """States crossing the active and eligibility lines between two thresholds."""
    rec_ref = np.asarray(rec_ref); rec_alt = np.asarray(rec_alt); flag_free = np.asarray(flag_free, bool)
    active_changed = int(np.sum((rec_ref > 0) != (rec_alt > 0)))
    elig_ref = flag_free & (rec_ref >= ELIGIBILITY_RECURRENCE)
    elig_alt = flag_free & (rec_alt >= ELIGIBILITY_RECURRENCE)
    return {
        "active_changed": active_changed,
        "eligible_changed": int(np.sum(elig_ref != elig_alt)),
        "n_eligible_ref": int(elig_ref.sum()),
        "n_eligible_alt": int(elig_alt.sum()),
        "gained_eligible": np.flatnonzero(elig_alt & ~elig_ref).astype(int).tolist(),
        "lost_eligible": np.flatnonzero(elig_ref & ~elig_alt).astype(int).tolist(),
    }


def rank_stability(rec_ref, rec_alt):
    """Spearman between two recurrence vectors over the reference active set and over states active in both."""
    rec_ref = np.asarray(rec_ref); rec_alt = np.asarray(rec_alt)
    ref_active = rec_ref > 0
    both = ref_active & (rec_alt > 0)

    def _rho(mask):
        if mask.sum() < 3:
            return None
        return safe_float(spearmanr(rec_ref[mask], rec_alt[mask]).statistic)

    return {"rho_reference_active": _rho(ref_active), "rho_both_active": _rho(both),
            "n_reference_active": int(ref_active.sum()), "n_both_active": int(both.sum())}


def range_stats(rec):
    """Range statistics over active states (the numbers quoted in Results para 2)."""
    a = np.asarray(rec)[np.asarray(rec) > 0]
    if a.size == 0:
        return {"n_active": 0, "min": None, "max": None, "p10": None, "p90": None}
    return {"n_active": int(a.size), "min": float(a.min()), "max": float(a.max()),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90))}


def load_inputs(sub_id, parcellation, vt):
    """Frozen main-pipeline inputs for one subject."""
    base = os.path.join(SCRATCH_DIR, "output")
    rdir = os.path.join(base, "05a_recurrence_analysis", parcellation, sub_id, f"vt{vt}")
    with open(os.path.join(rdir, "fractional_occupancy.pkl"), "rb") as f:
        fo = pickle.load(f)
    recurrence = np.load(os.path.join(rdir, "recurrence_scores.npy"))
    flags = pd.read_csv(os.path.join(base, "05e_temporal_trend_a4", parcellation, sub_id,
                                     f"vt{vt}", "state_flags.csv")).sort_values("state")
    saved_pi = np.load(os.path.join(base, "06b_transition_structure", parcellation, sub_id,
                                    f"vt{vt}", "stationary_distribution.npy"))
    model_path = os.path.join(base, "04_combined_hdphmm", parcellation, sub_id,
                              "final", f"vt{vt}", "best_model.pkl")
    model = _load_model_no_jax(model_path)
    return {"fo": fo, "recurrence": recurrence, "flags": flags, "saved_pi": saved_pi,
            "transmat": np.asarray(model.transmat_), "n_states": int(model.n_components),
            "paths": {"recurrence_dir": rdir, "model": model_path}}


def run_subject(sub_id, parcellation, vt, out_dir):
    """Threshold sweep, churn, and occupancy-confound correlations for one subject."""
    inp = load_inputs(sub_id, parcellation, vt)
    fo, recurrence, flags, K = inp["fo"], inp["recurrence"], inp["flags"], inp["n_states"]
    if len(flags) != K:
        raise RuntimeError(f"{sub_id}: state_flags.csv has {len(flags)} rows, model has {K} states")

    # Gate 1: the reference threshold reproduces the published recurrence exactly.
    rec_ref = recurrence_at(fo, K, REFERENCE_THRESHOLD)
    rec_delta = float(np.max(np.abs(rec_ref - recurrence)))
    if rec_delta != 0.0:
        raise RuntimeError(f"{sub_id}: recurrence at {REFERENCE_THRESHOLD} deviates from saved by {rec_delta:.2e}")

    # Gate 2: flag-free categories at the reference threshold match 05e_a4.
    cats = flags["summary_category"].to_numpy()
    flag_free = np.isin(cats, FLAG_FREE_CATEGORIES)
    # 05e assigns "unused" before any exclusion flag, so an unused-at-0.02 state may
    # carry sub-HRF/run-onset/drift flags that its category hides. Masking those
    # states out of flag_free (via `(cats == "unused") & raw_flagged`) was tried and
    # reverted: it left every thresholds[*].churn_vs_reference and occupancy_confound
    # value unchanged, but it did shift n_flag_free_states (34->31 for sub-02 alone),
    # so it is not applied here pending a decision on how n_flag_free_states is used
    # downstream. summary_category is treated as authoritative as-is.
    mismatches = [{"state": int(s), "saved": str(cats[s]), "recomputed": recurrence_category(rec_ref[s])}
                  for s in np.flatnonzero(flag_free) if recurrence_category(rec_ref[s]) != cats[s]]
    if mismatches:
        raise RuntimeError(f"{sub_id}: {len(mismatches)} flag-free states disagree with state_flags.csv: {mismatches[:3]}")

    # Gate 3: pi over active states matches 06b.
    active = np.flatnonzero(recurrence > 0)
    pi = stationary_distribution(inp["transmat"], active)
    if inp["saved_pi"].shape != pi.shape:
        raise RuntimeError(f"{sub_id}: pi shape {pi.shape} vs saved {inp['saved_pi'].shape}")
    pi_delta = float(np.max(np.abs(inp["saved_pi"] - pi)))
    if pi_delta > 1e-6:
        raise RuntimeError(f"{sub_id}: recomputed pi deviates from saved by {pi_delta:.2e}")

    by_threshold, matrix = {}, []
    for t in THRESHOLDS:
        rec_t = recurrence_at(fo, K, t)
        matrix.append(rec_t)
        by_threshold[f"{t:.2f}"] = {
            "threshold_fo": t,
            "approx_seconds_of_12min_run": round(t * 12 * 60, 1),
            "range": range_stats(rec_t),
            "rank_stability_vs_reference": rank_stability(rec_ref, rec_t),
            "churn_vs_reference": churn(rec_ref, rec_t, flag_free),
        }

    mfo = mean_fo(fo)
    confound = {
        "n_active": int(active.size),
        "rho_recurrence_pi": safe_float(spearmanr(recurrence[active], pi).statistic),
        "rho_recurrence_mean_fo": safe_float(spearmanr(recurrence[active], mfo[active]).statistic),
        "rho_pi_mean_fo": safe_float(spearmanr(pi, mfo[active]).statistic),
    }

    summary = {
        "sub_id": sub_id, "parcellation": parcellation, "vt": float(vt),
        "n_states_total": K, "n_runs": len(fo),
        "reference_threshold": REFERENCE_THRESHOLD, "eligibility_recurrence": ELIGIBILITY_RECURRENCE,
        "flag_free_categories": list(FLAG_FREE_CATEGORIES),
        "n_flag_free_states": int(flag_free.sum()),
        "gate": {"recurrence_max_abs_delta": rec_delta, "category_mismatches": mismatches,
                 "stationary_max_abs_delta": pi_delta},
        "thresholds": by_threshold,
        "occupancy_confound": confound,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "pandas": pd.__version__},
        "inputs": inp["paths"],
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "recurrence_robustness_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    np.save(os.path.join(out_dir, "recurrence_by_threshold.npy"), np.vstack(matrix))
    logger.info("%s: rho(rec,pi)=%.3f rho(rec,meanFO)=%.3f | rank stability vs 0.02: %s | eligible churn: %s",
                sub_id, confound["rho_recurrence_pi"], confound["rho_recurrence_mean_fo"],
                {k: v["rank_stability_vs_reference"]["rho_reference_active"] for k, v in by_threshold.items()},
                {k: v["churn_vs_reference"]["eligible_changed"] for k, v in by_threshold.items()})
    return summary


def build_parser():
    p = argparse.ArgumentParser(description="Recurrence threshold sensitivity and occupancy confound.")
    p.add_argument("--sub_id", required=True)
    p.add_argument("--parcellation", default="atlas-4S156Parcels")
    p.add_argument("--vt", default="0.95")
    p.add_argument("--out_dir", default=None)
    return p


def main():
    if SCRATCH_DIR is None:
        raise ValueError("SCRATCH_DIR must be set in environment or .env file")
    a = build_parser().parse_args()
    vt = f"{float(a.vt):.2f}"
    out_dir = a.out_dir or os.path.join(SCRATCH_DIR, "output", "sm_rel_r1_recurrence_robustness",
                                        a.parcellation, a.sub_id, f"vt{vt}")
    run_subject(a.sub_id, a.parcellation, vt, out_dir)


if __name__ == "__main__":
    main()
