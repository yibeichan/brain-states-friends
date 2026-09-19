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
